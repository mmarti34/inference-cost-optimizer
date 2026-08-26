"""
Direct Inference — OpenAI-compatible passthrough to a real provider model.

This is the engine behind mode 1 of ``POST /v1/chat/completions``.  A customer
points an existing OpenAI client at OptiML::

    client = OpenAI(base_url="https://api.optiml.one/v1", api_key="<optiml_service_key>")
    client.chat.completions.create(model="gpt-4o", messages=[...])

…and the request is executed against the customer's OWN provider key (from
``api_keys`` via ``api_key_cache``), with OptiML observing cost, latency,
workload identity and strategy.  There is no Studio workflow and no deployment
behind a direct request — see ``routers/openai_compat.py`` for the mode rule.

WHY THIS DOES NOT REUSE ``routers/*_router.py::handle_prompt``
--------------------------------------------------------------
Those adapters take a single flat ``prompt`` string, inject their own system
message, and (for tools) run an *agentic server-side loop* that executes tools
on OptiML's side.  OpenAI chat-completions semantics are the opposite on both
counts: the caller owns the full ``messages`` array, and ``tool_calls`` are
returned TO THE CALLER to execute.  Reusing them would silently change the
customer's application semantics.  We reuse the pieces that are genuinely
shared instead: the provider base URLs those adapters use, the org provider-key
cache, ``utils.pricing``, and ``provider_resilience``.

DIALECTS
--------
Two wire dialects cover all nine providers:

* ``openai``    — the request body is forwarded (near-)verbatim to an
  OpenAI-compatible ``/chat/completions``.  The provider is the authority on
  what it supports; its errors are propagated, never swallowed.  This is
  deliberate: forwarding is honest, whereas a hand-maintained allow-list would
  reject parameters providers added last week.
* ``anthropic`` — native Messages API translation (Anthropic has no
  first-class OpenAI chat-completions surface we are willing to depend on).
  Because we translate, anything we cannot express is rejected LOUDLY with the
  offending field named; nothing is silently dropped.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import httpx

from utils.pricing import get_pricing, get_provider_for_model

logger = logging.getLogger(__name__)

# ── Provider wire configuration ──────────────────────────────────────────────
# Base URLs mirror the ones already used by routers/*_router.py so direct
# inference and workflow execution hit exactly the same endpoints.
OPENAI_DIALECT_BASES: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "mistral": "https://api.mistral.ai/v1",
    # Provider-maintained OpenAI compatibility surfaces.
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "cohere": "https://api.cohere.ai/compatibility/v1",
}

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

#: Every provider id OptiML can dispatch direct inference to.  A ``model``
#: string whose first path segment is one of these is treated as an explicit
#: provider qualification (``anthropic/claude-sonnet-4-5``).
KNOWN_PROVIDERS: tuple[str, ...] = tuple(OPENAI_DIALECT_BASES) + ("anthropic",)

#: Providers that document ``stream_options.include_usage``.  Sending it to a
#: provider that does not know the field is a 400, so we only add it here.
_STREAM_USAGE_OPT_IN = {"openai", "groq", "together", "fireworks", "deepseek"}

#: Fields the Anthropic translation genuinely cannot express.  Anything listed
#: here produces a 400 naming the field rather than being dropped on the floor.
_ANTHROPIC_UNSUPPORTED = (
    "response_format",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "parallel_tool_calls",
)

#: OptiML-internal keys stripped from the body before it reaches a provider.
#: ``metadata`` is NOT stripped — OpenAI defines it and providers accept it —
#: but the ``optiml`` entry inside it is removed by ``strip_optiml_metadata``.
_OPTIML_BODY_KEYS = ("optiml",)

#: Params we never forward because they are OptiML transport concerns.
_NON_FORWARDED = {"model", "messages", "stream", "metadata", "optiml"}

DEFAULT_TIMEOUT_S = 600.0
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class _NonRetryable(Exception):
    """
    Internal marker: carry a :class:`DirectInferenceError` past the retry layer.

    ``provider_resilience`` decides retryability by inspecting the exception, and
    it treats anything with ``status_code`` 429 as transient — correct for a
    workflow step OptiML owns, WRONG for a passthrough proxy.  On a passthrough,
    the caller's own OpenAI client is the authority on rate-limit backoff: if we
    silently burn 3 seconds retrying their 429, their client then retries on top
    of that, and a fast actionable error becomes a compounding hang in *their*
    application.  Wrapping 4xx in this marker makes it non-retryable without
    changing the shared retry policy for anyone else.
    """

    def __init__(self, err: "DirectInferenceError") -> None:
        super().__init__(err.message)
        self.err = err


class DirectInferenceError(Exception):
    """An error that must be rendered in the OpenAI error envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        err_type: str = "invalid_request_error",
        code: Optional[str] = None,
        param: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.code = code
        self.param = param


@dataclass(frozen=True)
class ResolvedModel:
    """Where a direct-inference ``model`` string actually goes."""

    provider: str
    model: str  # provider-native model id (prefix stripped)
    requested: str  # exactly what the caller sent
    dialect: str  # "openai" | "anthropic"
    api_base: str
    explicit_provider: bool  # True when the caller wrote "<provider>/<model>"


@dataclass
class CompletionResult:
    """One completed direct-inference call, in OpenAI shape plus attribution."""

    body: dict[str, Any]
    finish_reason: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    provider_latency_ms: int
    cost_usd: Optional[float]
    cost_estimated: bool
    pricing_source: str
    tokens_known: bool
    tool_call_count: int = 0

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


# ─────────────────────────────────────────────────────────────────────────────
# Model resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_direct_model(model: str) -> ResolvedModel:
    """
    Resolve a bare (non-``optiml/``) model string to a provider + model.

    Resolution is ordered and deterministic:

    1. ``<known_provider>/<rest>``  →  that provider, ``rest`` as the model id.
       Only the exact provider ids in :data:`KNOWN_PROVIDERS` qualify, so
       ``meta-llama/Llama-3.3-70B-Instruct-Turbo`` is NOT read as a provider
       prefix and still resolves to Together.  A Together-hosted model whose
       own namespace collides with a provider id (``openai/gpt-oss-120b``) must
       be written fully qualified: ``together/openai/gpt-oss-120b``.
    2. ``utils.pricing.get_provider_for_model`` — the shared registry, then its
       documented prefix heuristics.

    Raises ``DirectInferenceError(400, code="model_not_found")`` when neither
    step resolves.  It NEVER falls back to a workflow lookup: see the mode rule.
    """
    requested = (model or "").strip()
    if not requested:
        raise DirectInferenceError(
            400, "model is required.", code="model_required", param="model"
        )

    if "/" in requested:
        head, rest = requested.split("/", 1)
        if head.lower() in KNOWN_PROVIDERS and rest.strip():
            return _build_resolved(head.lower(), rest.strip(), requested, True)
        # Fireworks' own model namespace. Checked here because the shared
        # registry's generic '"/" means Together' heuristic would otherwise
        # claim any unlisted Fireworks model id.
        if requested.lower().startswith("accounts/fireworks/"):
            return _build_resolved("fireworks", requested, requested, False)

    provider = get_provider_for_model(requested)
    if provider and provider.lower() in KNOWN_PROVIDERS:
        return _build_resolved(provider.lower(), requested, requested, False)

    raise DirectInferenceError(
        400,
        (
            f"Unknown model '{requested}'. OptiML treated it as direct inference "
            "because it is not prefixed with the reserved 'optiml/' namespace. "
            "Qualify the provider explicitly (e.g. 'openai/"
            f"{requested}') to route it, or use 'optiml/<endpoint_slug>' to call "
            "an OptiML deployed workflow."
        ),
        code="model_not_found",
        param="model",
    )


def _build_resolved(provider: str, model: str, requested: str, explicit: bool) -> ResolvedModel:
    if provider == "anthropic":
        return ResolvedModel(
            provider="anthropic",
            model=model,
            requested=requested,
            dialect="anthropic",
            api_base=ANTHROPIC_BASE,
            explicit_provider=explicit,
        )
    base = OPENAI_DIALECT_BASES.get(provider)
    if not base:
        raise DirectInferenceError(
            400,
            f"Provider '{provider}' is not available for direct inference.",
            code="provider_not_supported",
            param="model",
        )
    return ResolvedModel(
        provider=provider,
        model=model,
        requested=requested,
        dialect="openai",
        api_base=base,
        explicit_provider=explicit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pricing (delegated — never duplicated)
# ─────────────────────────────────────────────────────────────────────────────
def price_call(
    provider: str,
    model: str,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
) -> tuple[Optional[float], bool, str]:
    """
    Return ``(cost_usd, estimated, pricing_source)`` using ``utils.pricing``.

    ``cost_usd`` is None when we have no token counts at all — we do not invent
    spend.  ``estimated`` is True when the model has no real price in the
    registry; callers MUST surface that flag rather than presenting the number
    as measured.
    """
    if prompt_tokens is None and completion_tokens is None:
        try:
            source = str(get_pricing(provider, model).get("source") or "default")
            estimated = bool(get_pricing(provider, model).get("estimated"))
        except Exception:
            source, estimated = "default", True
        return None, estimated, source
    try:
        pricing = get_pricing(provider, model)
    except Exception:
        return None, True, "default"
    cost = (
        (prompt_tokens or 0) * float(pricing["input"])
        + (completion_tokens or 0) * float(pricing["output"])
    ) / 1000.0
    return cost, bool(pricing.get("estimated")), str(pricing.get("source") or "default")


# ─────────────────────────────────────────────────────────────────────────────
# Request assembly
# ─────────────────────────────────────────────────────────────────────────────
def strip_optiml_metadata(body: dict[str, Any]) -> dict[str, Any]:
    """Remove OptiML-only fields so nothing OptiML-specific reaches a provider."""
    clean = {k: v for k, v in body.items() if k not in _OPTIML_BODY_KEYS}
    meta = clean.get("metadata")
    if isinstance(meta, dict):
        pruned = {
            k: v
            for k, v in meta.items()
            if k != "optiml" and not str(k).lower().startswith("optiml_")
        }
        if pruned:
            clean["metadata"] = pruned
        else:
            clean.pop("metadata", None)
    return clean


def forwardable_params(body: dict[str, Any]) -> dict[str, Any]:
    """Every caller-supplied parameter that should reach the provider verbatim."""
    clean = strip_optiml_metadata(body)
    return {k: v for k, v in clean.items() if k not in _NON_FORWARDED and v is not None}


def get_org_provider_key(org_id: str, provider: str) -> str:
    """
    The customer's own provider key.  Direct inference spends the customer's
    provider budget, exactly as their application did before OptiML.
    """
    from api_key_cache import get_provider_api_key

    try:
        return get_provider_api_key(org_id, provider)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status == 404:
            raise DirectInferenceError(
                400,
                (
                    f"No {provider} API key is configured for this organization. "
                    f"Add one in OptiML settings before sending {provider} traffic "
                    "through direct inference."
                ),
                code="provider_key_missing",
                param="model",
            ) from exc
        raise DirectInferenceError(
            500,
            f"Could not load the {provider} API key for this organization.",
            err_type="api_error",
            code="provider_key_error",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Error propagation
# ─────────────────────────────────────────────────────────────────────────────
def _provider_error(provider: str, status: int, raw_body: str) -> DirectInferenceError:
    """
    Surface the provider's own status and message in the OpenAI error shape.

    We never collapse a provider 401/429/400 into a 500: the caller's client
    library relies on the status to retry, back off, or fail fast.
    """
    message = (raw_body or "").strip()
    err_type = "api_error"
    code: Optional[str] = None
    param: Optional[str] = None
    try:
        parsed = json.loads(raw_body)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            message = str(err.get("message") or message)
            err_type = str(err.get("type") or err_type)
            code = err.get("code")
            param = err.get("param")
        elif isinstance(err, str):
            message = err
        elif isinstance(parsed, dict) and parsed.get("message"):
            message = str(parsed["message"])
    except (ValueError, TypeError):
        pass

    if status == 401:
        err_type = "authentication_error"
    elif status == 403:
        err_type = "permission_error"
    elif status == 429:
        err_type = "rate_limit_error"
    elif 400 <= status < 500 and err_type == "api_error":
        err_type = "invalid_request_error"

    return DirectInferenceError(
        status,
        f"{provider}: {message[:1500] or 'provider returned an error'}",
        err_type=err_type,
        code=code if isinstance(code, str) else None,
        param=param if isinstance(param, str) else None,
    )


def _transport_error(provider: str, exc: Exception) -> DirectInferenceError:
    if isinstance(exc, httpx.TimeoutException):
        return DirectInferenceError(
            504,
            f"{provider}: request timed out.",
            err_type="api_error",
            code="provider_timeout",
        )
    return DirectInferenceError(
        502,
        f"{provider}: could not reach the provider ({type(exc).__name__}).",
        err_type="api_error",
        code="provider_unreachable",
    )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-dialect passthrough
# ─────────────────────────────────────────────────────────────────────────────
def _openai_dialect_payload(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = dict(params)
    payload["model"] = resolved.model
    payload["messages"] = messages
    if stream:
        payload["stream"] = True
        if resolved.provider in _STREAM_USAGE_OPT_IN and "stream_options" not in payload:
            payload["stream_options"] = {"include_usage": True}
    return payload


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _openai_complete(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    api_key: str,
    *,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    payload = _openai_dialect_payload(resolved, messages, params, stream=False)
    url = f"{resolved.api_base.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=_openai_headers(api_key), json=payload)
    except Exception as exc:  # network / DNS / TLS / timeout
        raise _transport_error(resolved.provider, exc) from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if response.status_code >= 400:
        raise _provider_error(resolved.provider, response.status_code, response.text)
    try:
        return response.json(), latency_ms
    except ValueError as exc:
        raise DirectInferenceError(
            502,
            f"{resolved.provider}: returned a non-JSON response.",
            err_type="api_error",
            code="provider_bad_response",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic native translation
# ─────────────────────────────────────────────────────────────────────────────
def _reject_unsupported_for_anthropic(params: dict[str, Any]) -> None:
    for name in _ANTHROPIC_UNSUPPORTED:
        if params.get(name) is not None:
            raise DirectInferenceError(
                400,
                (
                    f"'{name}' is not supported by Anthropic through OptiML direct "
                    "inference and OptiML will not silently ignore it. Remove the "
                    "field, or send this request to a provider that supports it."
                ),
                code="param_not_supported_by_provider",
                param=name,
            )
    n = params.get("n")
    if n is not None and int(n) != 1:
        raise DirectInferenceError(
            400,
            "Anthropic returns a single completion; 'n' must be 1.",
            code="param_not_supported_by_provider",
            param="n",
        )
    temperature = params.get("temperature")
    if temperature is not None and float(temperature) > 1.0:
        raise DirectInferenceError(
            400,
            (
                "Anthropic accepts temperature in [0, 1]; OpenAI allows [0, 2]. "
                f"Received {temperature}. Rescale it rather than having OptiML "
                "guess an equivalent."
            ),
            code="param_out_of_range_for_provider",
            param="temperature",
        )


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                out.append(block)
        return "\n".join(p for p in out if p)
    return str(content)


def _anthropic_image_block(image_url: Any) -> dict[str, Any]:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    url = str(url or "")
    if url.startswith("data:"):
        try:
            header, b64 = url.split(",", 1)
            media_type = header.split(":", 1)[1].split(";", 1)[0]
            base64.b64decode(b64, validate=True)
        except (ValueError, IndexError, binascii.Error) as exc:
            raise DirectInferenceError(
                400,
                "Malformed data: URL in an image_url content part.",
                code="invalid_image_url",
                param="messages",
            ) from exc
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise DirectInferenceError(
        400,
        "image_url must be an http(s) URL or a data: URL.",
        code="invalid_image_url",
        param="messages",
    )


def _anthropic_content_blocks(content: Any) -> Any:
    """OpenAI multi-part content → Anthropic content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    blocks: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": str(part.get("text") or "")})
        elif ptype == "image_url":
            blocks.append(_anthropic_image_block(part.get("image_url")))
        else:
            raise DirectInferenceError(
                400,
                (
                    f"Content part type '{ptype}' cannot be translated to Anthropic. "
                    "OptiML will not drop it silently."
                ),
                code="content_part_not_supported",
                param="messages",
            )
    return blocks or ""


def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    """
    Split an OpenAI messages array into Anthropic ``(system, messages)``.

    ``tool`` role messages become ``tool_result`` blocks on a user turn, and an
    assistant message carrying ``tool_calls`` becomes ``tool_use`` blocks —
    which is what makes a customer's existing multi-turn tool loop keep working.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role") or "user").lower()
        if role in ("system", "developer"):
            text = _text_of(msg.get("content"))
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": str(msg.get("tool_call_id") or ""),
                "content": _text_of(msg.get("content")),
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _text_of(msg.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for call in msg.get("tool_calls") or []:
                fn = (call or {}).get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (ValueError, TypeError):
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(call.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "input": args if isinstance(args, dict) else {},
                    }
                )
            out.append({"role": "assistant", "content": blocks or ""})
            continue

        out.append({"role": "user", "content": _anthropic_content_blocks(msg.get("content"))})

    system: Any = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _anthropic_tools(tools: Any) -> list[dict[str, Any]]:
    out = []
    for tool in tools or []:
        fn = (tool or {}).get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "name": str(name),
                "description": str(fn.get("description") or ""),
                "input_schema": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return out


def _anthropic_tool_choice(tool_choice: Any) -> Optional[dict[str, Any]]:
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": str(name)}
    raise DirectInferenceError(
        400,
        f"tool_choice value {tool_choice!r} cannot be translated to Anthropic.",
        code="param_not_supported_by_provider",
        param="tool_choice",
    )


_ANTHROPIC_STOP_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _anthropic_payload(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    stream: bool,
) -> dict[str, Any]:
    _reject_unsupported_for_anthropic(params)
    system, translated = _anthropic_messages(messages)

    max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
    payload: dict[str, Any] = {
        "model": resolved.model,
        "messages": translated,
        "max_tokens": int(max_tokens) if max_tokens else ANTHROPIC_DEFAULT_MAX_TOKENS,
    }
    if system:
        payload["system"] = system
    if params.get("temperature") is not None:
        payload["temperature"] = float(params["temperature"])
    if params.get("top_p") is not None:
        payload["top_p"] = float(params["top_p"])
    stop = params.get("stop")
    if stop:
        payload["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    tools = _anthropic_tools(params.get("tools"))
    if tools:
        payload["tools"] = tools
        choice = _anthropic_tool_choice(params.get("tool_choice"))
        if choice:
            payload["tool_choice"] = choice
    if stream:
        payload["stream"] = True
    return payload


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }


def _anthropic_to_openai_body(
    data: dict[str, Any],
    resolved: ResolvedModel,
    *,
    request_id: str,
    created: int,
) -> tuple[dict[str, Any], str, int]:
    """Anthropic Messages response → OpenAI chat.completion body."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )

    finish = _ANTHROPIC_STOP_REASON.get(str(data.get("stop_reason") or ""), "stop")
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = data.get("usage") or {}
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": resolved.requested,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
        },
    }
    return body, finish, len(tool_calls)


def _anthropic_complete(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    api_key: str,
    *,
    timeout: float,
) -> tuple[dict[str, Any], int]:
    payload = _anthropic_payload(resolved, messages, params, stream=False)
    url = f"{resolved.api_base.rstrip('/')}/messages"
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=_anthropic_headers(api_key), json=payload)
    except Exception as exc:
        raise _transport_error("anthropic", exc) from exc
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if response.status_code >= 400:
        raise _provider_error("anthropic", response.status_code, response.text)
    try:
        return response.json(), latency_ms
    except ValueError as exc:
        raise DirectInferenceError(
            502,
            "anthropic: returned a non-JSON response.",
            err_type="api_error",
            code="provider_bad_response",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public: non-streaming
# ─────────────────────────────────────────────────────────────────────────────
def complete(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    org_id: str,
    *,
    request_id: str,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> CompletionResult:
    """
    Execute one non-streaming direct-inference call.

    Retries and the per-attempt deadline come from ``provider_resilience`` — the
    same policy workflow execution uses — with ONE deliberate narrowing: any 4xx
    the provider returns is passed straight through instead of being retried.
    See :class:`_NonRetryable` for why a proxy must not absorb the caller's 429.
    Transport failures and 5xx still retry, because those are invisible to the
    caller and genuinely transient.
    """
    from provider_resilience import call_with_resilience

    api_key = get_org_provider_key(org_id, resolved.provider)
    created = int(time.time())

    def attempt() -> tuple[dict[str, Any], int]:
        try:
            if resolved.dialect == "anthropic":
                return _anthropic_complete(resolved, messages, params, api_key, timeout=timeout)
            return _openai_complete(resolved, messages, params, api_key, timeout=timeout)
        except DirectInferenceError as err:
            if 400 <= err.status_code < 500:
                raise _NonRetryable(err) from err
            raise

    try:
        data, latency_ms = call_with_resilience(
            attempt,
            attempt_timeout=timeout,
            context_label=f"direct_inference {resolved.provider}/{resolved.model}",
        )
    except _NonRetryable as marker:
        raise marker.err from None

    if resolved.dialect == "anthropic":
        body, finish, tool_calls = _anthropic_to_openai_body(
            data, resolved, request_id=request_id, created=created
        )
        usage = body["usage"]
        prompt_tokens: Optional[int] = usage["prompt_tokens"]
        completion_tokens: Optional[int] = usage["completion_tokens"]
    else:
        body = dict(data)
        # Echo the model string the caller asked for; providers sometimes
        # return a dated variant, which breaks naive client-side equality.
        body.setdefault("id", request_id)
        body["object"] = "chat.completion"
        body.setdefault("created", created)
        body["model"] = resolved.requested
        choices = body.get("choices") or []
        first = choices[0] if choices else {}
        finish = str(first.get("finish_reason") or "stop")
        tool_calls = len(((first.get("message") or {}).get("tool_calls")) or [])
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

    tokens_known = prompt_tokens is not None or completion_tokens is not None
    cost, estimated, source = price_call(
        resolved.provider, resolved.model, prompt_tokens, completion_tokens
    )

    return CompletionResult(
        body=body,
        finish_reason=finish,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider_latency_ms=latency_ms,
        cost_usd=cost,
        cost_estimated=estimated,
        pricing_source=source,
        tokens_known=tokens_known,
        tool_call_count=tool_calls,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: streaming
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StreamAccounting:
    """Mutable totals a stream fills in as it runs, read once it completes."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    provider_latency_ms: int = 0
    tool_call_count: int = 0
    error: Optional[DirectInferenceError] = None
    chunks: int = 0

    @property
    def tokens_known(self) -> bool:
        return self.prompt_tokens is not None or self.completion_tokens is not None


def _sse(data: Any) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _error_chunk(err: DirectInferenceError) -> str:
    return _sse(
        {
            "error": {
                "message": err.message,
                "type": err.err_type,
                "param": err.param,
                "code": err.code,
            }
        }
    )


def stream(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    org_id: str,
    *,
    request_id: str,
    accounting: StreamAccounting,
    include_usage: bool,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Iterator[str]:
    """
    Yield real OpenAI-format SSE (``chat.completion.chunk`` … ``[DONE]``).

    Errors that happen mid-stream cannot change the HTTP status — the 200 is
    already on the wire — so they are emitted as a terminal ``error`` SSE event
    (the same convention OpenAI uses) and recorded on ``accounting`` for the
    caller to log.  Nothing is fabricated: if the provider never reports usage,
    no usage block is emitted and ``accounting.tokens_known`` stays False.
    """
    created = int(time.time())
    try:
        api_key = get_org_provider_key(org_id, resolved.provider)
    except DirectInferenceError as err:
        accounting.error = err
        yield _error_chunk(err)
        yield "data: [DONE]\n\n"
        return

    try:
        if resolved.dialect == "anthropic":
            gen = _stream_anthropic(
                resolved, messages, params, api_key,
                request_id=request_id, created=created,
                accounting=accounting, include_usage=include_usage, timeout=timeout,
            )
        else:
            gen = _stream_openai_dialect(
                resolved, messages, params, api_key,
                request_id=request_id, created=created,
                accounting=accounting, include_usage=include_usage, timeout=timeout,
            )
        for chunk in gen:
            accounting.chunks += 1
            yield chunk
    except DirectInferenceError as err:
        accounting.error = err
        yield _error_chunk(err)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("direct inference stream failed")
        err = DirectInferenceError(
            500, f"Stream failed: {type(exc).__name__}", err_type="api_error"
        )
        accounting.error = err
        yield _error_chunk(err)

    yield "data: [DONE]\n\n"


def _stream_openai_dialect(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    api_key: str,
    *,
    request_id: str,
    created: int,
    accounting: StreamAccounting,
    include_usage: bool,
    timeout: float,
) -> Iterator[str]:
    payload = _openai_dialect_payload(resolved, messages, params, stream=True)
    url = f"{resolved.api_base.rstrip('/')}/chat/completions"
    t0 = time.perf_counter()

    with httpx.Client(timeout=timeout) as client:
        try:
            ctx = client.stream("POST", url, headers=_openai_headers(api_key), json=payload)
        except Exception as exc:
            raise _transport_error(resolved.provider, exc) from exc
        with ctx as response:
            if response.status_code >= 400:
                response.read()
                raise _provider_error(resolved.provider, response.status_code, response.text)
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                except ValueError:
                    continue

                usage = parsed.get("usage")
                if isinstance(usage, dict):
                    if usage.get("prompt_tokens") is not None:
                        accounting.prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        accounting.completion_tokens = int(usage["completion_tokens"])
                    if not parsed.get("choices"):
                        # Usage-only trailer. Forward only if the caller asked.
                        if include_usage:
                            parsed["model"] = resolved.requested
                            yield _sse(parsed)
                        continue
                    if not include_usage:
                        parsed.pop("usage", None)

                for choice in parsed.get("choices") or []:
                    if choice.get("finish_reason"):
                        accounting.finish_reason = str(choice["finish_reason"])
                    calls = (choice.get("delta") or {}).get("tool_calls") or []
                    for call in calls:
                        if call.get("id"):
                            accounting.tool_call_count += 1

                parsed["object"] = "chat.completion.chunk"
                parsed["model"] = resolved.requested
                parsed.setdefault("id", request_id)
                parsed.setdefault("created", created)
                yield _sse(parsed)

    accounting.provider_latency_ms = int((time.perf_counter() - t0) * 1000)


def _chunk(
    request_id: str, created: int, model: str, delta: dict[str, Any], finish: Optional[str] = None
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
    }


def _stream_anthropic(
    resolved: ResolvedModel,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    api_key: str,
    *,
    request_id: str,
    created: int,
    accounting: StreamAccounting,
    include_usage: bool,
    timeout: float,
) -> Iterator[str]:
    """Anthropic SSE event stream → OpenAI ``chat.completion.chunk`` stream."""
    payload = _anthropic_payload(resolved, messages, params, stream=True)
    url = f"{resolved.api_base.rstrip('/')}/messages"
    model = resolved.requested
    t0 = time.perf_counter()

    tool_index = -1
    emitted_role = False

    with httpx.Client(timeout=timeout) as client:
        try:
            ctx = client.stream("POST", url, headers=_anthropic_headers(api_key), json=payload)
        except Exception as exc:
            raise _transport_error("anthropic", exc) from exc
        with ctx as response:
            if response.status_code >= 400:
                response.read()
                raise _provider_error("anthropic", response.status_code, response.text)

            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                etype = event.get("type")

                if etype == "message_start":
                    usage = (event.get("message") or {}).get("usage") or {}
                    if usage.get("input_tokens") is not None:
                        accounting.prompt_tokens = int(usage["input_tokens"])
                    if not emitted_role:
                        emitted_role = True
                        yield _sse(_chunk(request_id, created, model, {"role": "assistant"}))

                elif etype == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        tool_index += 1
                        accounting.tool_call_count += 1
                        yield _sse(
                            _chunk(
                                request_id, created, model,
                                {
                                    "tool_calls": [
                                        {
                                            "index": tool_index,
                                            "id": str(block.get("id") or ""),
                                            "type": "function",
                                            "function": {
                                                "name": str(block.get("name") or ""),
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                            )
                        )

                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        yield _sse(
                            _chunk(request_id, created, model, {"content": delta.get("text") or ""})
                        )
                    elif dtype == "input_json_delta":
                        yield _sse(
                            _chunk(
                                request_id, created, model,
                                {
                                    "tool_calls": [
                                        {
                                            "index": max(tool_index, 0),
                                            "function": {
                                                "arguments": delta.get("partial_json") or ""
                                            },
                                        }
                                    ]
                                },
                            )
                        )

                elif etype == "message_delta":
                    stop_reason = (event.get("delta") or {}).get("stop_reason")
                    usage = event.get("usage") or {}
                    if usage.get("output_tokens") is not None:
                        accounting.completion_tokens = int(usage["output_tokens"])
                    finish = _ANTHROPIC_STOP_REASON.get(str(stop_reason or ""), "stop")
                    accounting.finish_reason = finish
                    yield _sse(_chunk(request_id, created, model, {}, finish))

                elif etype == "error":
                    err = event.get("error") or {}
                    raise DirectInferenceError(
                        502,
                        f"anthropic: {err.get('message') or 'stream error'}",
                        err_type="api_error",
                        code=str(err.get("type") or "provider_stream_error"),
                    )

    accounting.provider_latency_ms = int((time.perf_counter() - t0) * 1000)

    if include_usage and accounting.tokens_known:
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": accounting.prompt_tokens or 0,
                    "completion_tokens": accounting.completion_tokens or 0,
                    "total_tokens": (accounting.prompt_tokens or 0)
                    + (accounting.completion_tokens or 0),
                },
            }
        )
