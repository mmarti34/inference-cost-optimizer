"""
Auto-grade production responses using LLM. Fire-and-forget from public API; never blocks response.
"""
import asyncio
import json
import logging
import random
import time
from typing import Any, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)


def _model_to_provider(model: str) -> str:
    """Map grading model name to provider for router lookup."""
    if not model:
        return "openai"
    m = (model or "").lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m:
        return "openai"
    if "gemini" in m:
        return "gemini"
    return "openai"


def get_enabled_auto_graded_metrics_sync(org_id: str, endpoint_slug: str) -> list[dict]:
    """Return enabled auto_graded_metrics for this org/endpoint. Empty endpoint_slugs = all endpoints."""
    try:
        r = (
            supabase.table("auto_graded_metrics")
            .select("id, name, key, metric_type, rubric, categories, direction, grading_model, sample_rate, endpoint_slugs")
            .eq("org_id", org_id)
            .eq("enabled", True)
            .execute()
        )
        rows = list(r.data or [])
        out = []
        for row in rows:
            slugs = row.get("endpoint_slugs") or []
            if not slugs or endpoint_slug in slugs:
                out.append(row)
        return out
    except Exception as e:
        logger.warning("auto_graded_metrics fetch failed: %s", e)
        return []


def _call_grading_llm_sync(org_id: str, model: str, prompt: str, prompt_id: str) -> dict:
    """Sync call to LLM for grading. Routes to the correct provider (openai, anthropic, gemini) based on model."""
    provider = _model_to_provider(model)
    m = (model or "gpt-4o-mini").strip()
    if provider == "openai":
        from routers.openai_router import handle_prompt
        from routers.openai_router import PromptPayload
        payload = PromptPayload(org_id=org_id, provider=provider, model=m, prompt=prompt, prompt_id=prompt_id)
        return handle_prompt(payload)
    if provider == "anthropic":
        from routers.anthropic_router import handle_prompt
        from routers.anthropic_router import PromptPayload
        payload = PromptPayload(user_id="", org_id=org_id, provider=provider, model=m, prompt=prompt)
        out = handle_prompt(payload)
        return {"response": out.get("response", ""), "input_tokens": out.get("input_tokens", 0), "output_tokens": out.get("output_tokens", 0)}
    if provider == "gemini":
        from routers.gemini_router import handle_prompt
        from routers.gemini_router import PromptPayload
        payload = PromptPayload(org_id=org_id, provider=provider, model=m, prompt=prompt, prompt_id=prompt_id)
        out = handle_prompt(payload)
        return {"response": out.get("response", ""), "input_tokens": out.get("input_tokens", 0), "output_tokens": out.get("output_tokens", 0)}
    from routers.openai_router import handle_prompt
    from routers.openai_router import PromptPayload
    payload = PromptPayload(org_id=org_id, provider="openai", model="gpt-4o-mini", prompt=prompt, prompt_id=prompt_id)
    return handle_prompt(payload)


def _estimate_grading_cost_sync(prompt_tokens: int, output_tokens: int, model: str) -> float:
    try:
        from utils.pricing import get_pricing
        provider = _model_to_provider(model or "gpt-4o-mini")
        pricing = get_pricing(provider, (model or "gpt-4o-mini").strip())
        return (prompt_tokens * pricing.get("input", 0) + output_tokens * pricing.get("output", 0)) / 1000
    except Exception:
        return 0.0


def insert_auto_grade_result_sync(
    org_id: str,
    request_log_id: str,
    metric_id: str,
    metric_key: str,
    score: Optional[float] = None,
    binary_result: Optional[bool] = None,
    category_result: Optional[str] = None,
    reasoning: str = "",
    grading_latency_ms: Optional[int] = None,
    grading_cost: Optional[float] = None,
) -> None:
    try:
        supabase.table("auto_grade_results").insert({
            "org_id": org_id,
            "request_log_id": request_log_id,
            "metric_id": metric_id,
            "metric_key": metric_key,
            "score": score,
            "binary_result": binary_result,
            "category_result": category_result,
            "reasoning": (reasoning or "")[:500],
            "grading_latency_ms": grading_latency_ms,
            "grading_cost": grading_cost,
        }).execute()
    except Exception as e:
        logger.warning("auto_grade_results insert failed: %s", e)


class GradingError(RuntimeError):
    """The LLM judge could not produce a usable grade (call failed or reply was unparseable)."""


def build_grading_prompt(
    metric_type: str,
    rubric: str,
    input_text: str,
    output: str,
    categories: Optional[list] = None,
    expected_output: Optional[str] = None,
) -> str:
    """Build the judge prompt for one metric type. Single source of the rubric wording."""
    metric_type = (metric_type or "score").lower()
    rubric = (rubric or "").strip() or "Evaluate the response."
    inp_preview = (input_text or "")[:500]
    out_preview = (output or "")[:1000]
    expected_block = ""
    if expected_output:
        expected_block = f"\nReference answer (for comparison, not a required verbatim match): {str(expected_output)[:1000]}\n"

    header = (
        "You are evaluating an AI system's output.\n\n"
        f"Input from user: {inp_preview}\n"
        f"AI system output: {out_preview}\n"
        f"{expected_block}\n"
        f"{rubric}\n\n"
    )
    if metric_type == "binary":
        return header + 'Respond with ONLY a JSON object: {"result": true or false, "reasoning": "<one sentence>"}'
    if metric_type == "category":
        cats = categories or []
        cats_str = ", ".join(f'"{c}"' for c in cats) if cats else '"positive", "neutral", "negative"'
        return header + (
            f"Choose one category from: [{cats_str}]\n"
            'Respond with ONLY a JSON object: {"category": "<chosen category>", "reasoning": "<one sentence>"}'
        )
    # "score" and any unknown type both grade on the 1-5 scale.
    return header + 'Respond with ONLY a JSON object: {"score": <number 1-5>, "reasoning": "<one sentence>"}'


def _parse_grade_reply(reply: str) -> dict:
    reply_clean = (reply or "").strip().strip("`").strip()
    if reply_clean.startswith("json"):
        reply_clean = reply_clean[4:].strip()
    try:
        parsed = json.loads(reply_clean)
    except (ValueError, TypeError) as e:
        raise GradingError(f"judge reply was not valid JSON: {reply_clean[:200]}") from e
    if not isinstance(parsed, dict):
        raise GradingError(f"judge reply was not a JSON object: {reply_clean[:200]}")
    return parsed


def grade_output_sync(
    org_id: str,
    *,
    metric_type: str,
    rubric: str,
    output: str,
    input_text: str,
    model: Optional[str] = None,
    categories: Optional[list] = None,
    expected_output: Optional[str] = None,
    prompt_id: str = "grade",
) -> dict:
    """
    Run the LLM judge once and return a structured grade.

    This is the ONLY LLM judge in the backend. Both production auto-grading
    (auto_grade_results) and eval-gate model_graded checks (eval_run_results)
    go through it, so the rubric wording, provider routing and cost accounting
    stay identical between them.

    Returns {metric_type, score, binary_result, category_result, reasoning,
             grading_latency_ms, grading_cost, grader_provider, grader_model}.
    Raises GradingError when the judge could not be called or its reply could
    not be parsed — callers decide whether that is a failure or a skip.

    NOTE: the result is an LLM's opinion. It is a quality signal of provenance
    'llm_judge' and must never be recorded or presented as a business outcome
    or as a deterministic check.
    """
    metric_type = (metric_type or "score").lower()
    grading_model = (model or "gpt-4o-mini").strip() or "gpt-4o-mini"
    grader_provider = _model_to_provider(grading_model)
    grade_prompt = build_grading_prompt(
        metric_type, rubric, input_text, output,
        categories=categories, expected_output=expected_output,
    )

    start = time.perf_counter()
    try:
        result = _call_grading_llm_sync(org_id, grading_model, grade_prompt, prompt_id=prompt_id)
    except Exception as e:
        raise GradingError(f"judge call failed ({grader_provider}/{grading_model}): {e}") from e
    latency_ms = int((time.perf_counter() - start) * 1000)

    reply = (result.get("response") or result.get("output") or "").strip()
    parsed = _parse_grade_reply(reply)
    cost = _estimate_grading_cost_sync(
        result.get("input_tokens") or 0,
        result.get("output_tokens") or 0,
        grading_model,
    )

    score = None
    binary_result = None
    category_result = None
    if metric_type == "binary":
        binary_result = bool(parsed.get("result", False))
    elif metric_type == "category":
        category_result = str(parsed.get("category", ""))[:200]
    else:
        try:
            score = float(parsed.get("score", 0))
        except (TypeError, ValueError) as e:
            raise GradingError(f"judge returned a non-numeric score: {parsed.get('score')!r}") from e

    return {
        "metric_type": metric_type,
        "score": score,
        "binary_result": binary_result,
        "category_result": category_result,
        "reasoning": str(parsed.get("reasoning", ""))[:500],
        "grading_latency_ms": latency_ms,
        "grading_cost": cost,
        "grader_provider": grader_provider,
        "grader_model": grading_model,
    }


def _grade_one_sync(
    request_log_id: str,
    org_id: str,
    metric: dict,
    output: str,
    input_text: str,
) -> None:
    """Grade one metric (sync) and store it in auto_grade_results."""
    key = metric.get("key") or ""
    metric_id = metric.get("id")
    try:
        graded = grade_output_sync(
            org_id,
            metric_type=(metric.get("metric_type") or "score"),
            rubric=(metric.get("rubric") or ""),
            output=output or "",
            input_text=input_text or "",
            model=metric.get("grading_model"),
            categories=metric.get("categories") or [],
            prompt_id=f"grade-{key}-{request_log_id}",
        )
        insert_auto_grade_result_sync(
            org_id=org_id,
            request_log_id=request_log_id,
            metric_id=str(metric_id),
            metric_key=key,
            score=graded["score"],
            binary_result=graded["binary_result"],
            category_result=graded["category_result"],
            reasoning=graded["reasoning"],
            grading_latency_ms=graded["grading_latency_ms"],
            grading_cost=graded["grading_cost"],
        )
    except Exception as e:
        logger.error("Auto-grading failed for metric %s: %s", key, e, exc_info=False)


async def grade_response(
    request_log_id: str,
    org_id: str,
    endpoint_slug: str,
    output: str,
    input_text: str,
) -> None:
    """Run all enabled auto-graded metrics for this response. Fire-and-forget; never raises."""
    metrics = get_enabled_auto_graded_metrics_sync(org_id, endpoint_slug)
    if not metrics:
        return
    loop = asyncio.get_event_loop()
    for metric in metrics:
        sample_rate = float(metric.get("sample_rate") or 100)
        if sample_rate < 100 and random.random() * 100 > sample_rate:
            continue
        try:
            await loop.run_in_executor(
                None,
                _grade_one_sync,
                request_log_id,
                org_id,
                metric,
                output or "",
                input_text or "",
            )
        except Exception as e:
            logger.error("Auto-grading task failed for %s: %s", metric.get("key"), e)
