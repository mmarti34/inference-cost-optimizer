"""
Per-component token accounting for an assembled prompt.

WHY THIS EXISTS
---------------
`optimization.candidates.PromptCompressionGenerator` refuses to run, and its
first stated blocker is this:

    "A measured token breakdown per step attributable to the static prompt vs
     the injected context vs the user input. `node_results` records totals, not
     that split, so nothing today can say which part to cut."

This module is that breakdown, and nothing more. It answers "what is actually
in this prompt, and how much of it is the same on every request" from
MEASUREMENT. It proposes no change, writes nothing, and executes no model.

WHAT IT MEASURES
----------------
For one step of a workload, over a set of recorded replay cases, it reproduces
the prompt the runtime would assemble and splits it into components:

    system_instructions        data.systemInstructions | data.system_prefix
    task_description           data.taskDescription | data.task, with {{input}}
                               blanked out so the user's text is not counted
                               inside the template
    context_source:<label>     one per contextConfig source, as collected and
                               packaged by context_runtime
    user_input                 the case's own input text

The split follows `workflow_runtime` (the `ai-step` branch) and
`context_runtime` (`_collect_sources` -> `_package_context`) step for step, and
CALLS those functions rather than re-implementing them, because a decomposition
that does not match what the runtime assembles describes a prompt that is never
sent.

HONESTY RULES THIS MODULE OBEYS
-------------------------------
1.  A token count is a real tokenizer's output or it is None. There is no
    chars/4 anywhere in this file. `tokens is None` means NOT MEASURED, and
    `tokenizer is None` accompanies it every time.

2.  The tokenizer must be the one the model in use actually uses. `tiktoken`
    resolves that for OpenAI-family models (gpt-4.1 and gpt-4o -> o200k_base,
    gpt-3.5-turbo -> cl100k_base). For a model with no available tokenizer —
    Anthropic, Gemini, Mistral, any self-hosted id — tokens are None on every
    component. That is a correct answer, not a failure, and it is why
    `tokenizer` is per-component rather than assumed globally.

3.  `static` is MEASURED, never inferred from provenance. A component is static
    only if its text was byte-identical in EVERY profiled case and present in
    all of them. A "static system prompt" that interpolates a customer name is
    measured as varying, because it is. A component whose text could not be
    observed at all is never reported static.

4.  `reducible_tokens` is a FLOOR. It sums only components measured static that
    are not the user's input, and it is None the moment one of those could not
    be tokenized. It is the amount known to be present on every request; the
    true reducible amount can only be larger.

5.  Read-only. Every database access here is a SELECT. In particular
    `context_runtime._maybe_use_kb_consolidation` is deliberately NOT called:
    it bumps a hit counter, which is a write. See `_consolidation_may_apply`
    for how its effect is accounted for without performing it.

WHAT DOES NOT ADD UP, AND WHY THAT IS CORRECT
---------------------------------------------
* Component chars do not sum to the assembled prompt's chars. The joiners the
  runtime inserts ("\\n\\n" between blocks, "\\n\\n---\\n\\n" between context
  sources) belong to no component and are not attributed to one.

* Component tokens do not sum to `total_tokens`. BPE is not additive across a
  concatenation boundary: tokenizing two pieces separately and tokenizing their
  concatenation legitimately differ. `total_tokens` is therefore measured by
  tokenizing the assembled prompt itself, which is the number the provider
  actually bills, rather than by adding the parts up.

* `total_tokens` is None whenever the assembled prompt was not fully observed —
  an unresolvable context source, an LLM summarization step this module refuses
  to run, or a possible SM consolidation substitution. An undercount presented
  as a measurement would be worse than no number.

KNOWN LIMITS
------------
* Only the `ai-step` and `model` node types are profiled. A `model` node's
  prompt is nothing but the previous output (see `workflow_runtime` line ~1877),
  so it decomposes to user_input alone with zero reducible tokens — reported
  rather than skipped, because "nothing to cut here" is a finding.

* A `previous_node_output` context source cannot be resolved without executing
  the upstream node, which this module will not do. Such a source is reported
  with tokens=None and chars=0 (meaning NOT OBSERVED, not "empty"), and it
  forces `total_tokens` to None.

* `deployment_id` is not threaded through to `_collect_sources`, so live asset
  content is measured. If a promoted deployment pinned an asset snapshot whose
  content differs, the context components describe the live text.

* Cases are profiled with the FIRST step's view of the world: `input_text` is
  taken as the previous output. For a step deeper in the graph the real
  previous output is an upstream node's result, which is not available without
  executing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from supabase_client import supabase

logger = logging.getLogger(__name__)

# Component names. `context_source:` is a prefix, completed with the source's
# configured label.
COMPONENT_SYSTEM_INSTRUCTIONS = "system_instructions"
COMPONENT_TASK_DESCRIPTION = "task_description"
COMPONENT_USER_INPUT = "user_input"
CONTEXT_SOURCE_PREFIX = "context_source:"

#: Node types whose prompt this module knows how to decompose.
_AI_STEP = "ai-step"
_MODEL = "model"
PROFILABLE_NODE_TYPES = (_AI_STEP, _MODEL)

#: Upper bound on cases tokenized per profile. Profiling is pure CPU, but a
#: workload with thousands of replay cases should not turn a candidate-
#: generation call into a minute of BPE. `n_cases` reports what was SAMPLED and
#: `coverage` is measured against that, so the number stays interpretable.
MAX_SAMPLE_CASES = 200

#: The runtime's own joiners, from workflow_runtime / context_runtime.
_BLOCK_JOINER = "\n\n"
_CONTEXT_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Public shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentTokens:
    """
    One measured piece of an assembled prompt.

    tokens
        A real tokenizer's count, or None meaning NOT MEASURED. Never an
        estimate. None arises when no tokenizer exists for the model, or when
        the component's final text was not observed.
    chars
        Length of the text this component contributed, measured. For a
        component that was not observed at all this is 0, which means "no text
        seen", not "contributed nothing" — read it together with tokens=None.
        Where cases varied, this is the mean over the cases the component
        appeared in, rounded; for a static component every case agreed.
    tokenizer
        Name of the encoding used, e.g. "o200k_base". None whenever tokens is
        None, always.
    static
        Measured: the text was byte-identical across every profiled case and
        present in all of them. False also covers "could not be measured",
        which keeps an unobserved component out of `reducible_tokens`.
    """

    component: str
    source_type: Optional[str]
    tokens: Optional[int]
    chars: int
    tokenizer: Optional[str]
    static: bool


@dataclass(frozen=True)
class ContextProfile:
    """
    A workload step's prompt, decomposed and measured.

    total_tokens
        The assembled prompt tokenized as a whole — what the provider bills.
        None when the prompt was not fully observed or no tokenizer applies.
        NOT the sum of the components; see the module docstring.
    reducible_tokens
        A floor: tokens measured static and not the user's input. None when any
        such component could not be tokenized.
    coverage
        Profiled cases / sampled cases. Below 1.0 some cases could not be
        reproduced, and `static` was decided on the ones that could.
    """

    workload_id: str
    step_id: str
    n_cases: int
    components: tuple[ComponentTokens, ...]
    total_tokens: Optional[int]
    reducible_tokens: Optional[int]
    tokenizer: Optional[str]
    coverage: float


# ---------------------------------------------------------------------------
# Tokenizer resolution
# ---------------------------------------------------------------------------

def resolve_encoding(model: Optional[str], provider: Optional[str] = None):
    """
    The tokenizer the named model actually uses, or (None, None).

    Looked up, never guessed. `tiktoken.encoding_for_model` carries OpenAI's own
    model->encoding table (including prefix rules, so gpt-4.1-mini resolves the
    same way gpt-4.1 does); an id it does not know raises and we return None
    rather than defaulting to cl100k_base and calling the result a measurement.

    Provider is accepted but not used to choose an encoding: gpt-4o served
    through Azure tokenizes identically to gpt-4o served through OpenAI. It is
    part of the signature so a future provider-specific tokenizer has a place to
    land.

    Returns (encoder, encoding_name). Both None when nothing applies. Monkey-
    patchable: tests substitute this to exercise the measured paths without a
    network round trip for the BPE files.
    """
    if not model:
        return None, None
    try:
        import tiktoken
    except Exception:
        # Not installed. Every token count in this profile becomes None, which
        # is the honest result — see the module docstring.
        logger.info("tiktoken unavailable; context profile will report tokens=None")
        return None, None

    try:
        enc = tiktoken.encoding_for_model(str(model).strip())
    except KeyError:
        return None, None
    except Exception as exc:
        # The BPE file is fetched on first use. No network, no tokenizer.
        logger.info(
            "Could not load tokenizer for model %r: %s", model, type(exc).__name__
        )
        return None, None

    try:
        enc.encode("")
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("Tokenizer for %r unusable: %s", model, type(exc).__name__)
        return None, None

    return enc, getattr(enc, "name", None)


def _count(enc, text: Optional[str]) -> Optional[int]:
    """Tokens in `text`, or None if there is no tokenizer or no observed text."""
    if enc is None or text is None:
        return None
    try:
        return len(enc.encode(text, disallowed_special=()))
    except Exception:  # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------------------
# Internals: one case, decomposed
# ---------------------------------------------------------------------------

@dataclass
class _Piece:
    """A component as observed on ONE case. `text is None` means not observed."""

    name: str
    source_type: Optional[str]
    text: Optional[str]
    chars: int


def _apply_variables(template: str, variables: Optional[dict], prev_output: Optional[str] = None) -> str:
    """
    The runtime's own single-pass interpolation.

    Imported lazily: `workflow_runtime` pulls in the whole application at import
    time, and `optimization.candidates` imports this module. Re-implementing the
    substitution here would let the accounting drift away from the execution it
    claims to describe, so this raises rather than approximating if the runtime
    cannot be imported.
    """
    from workflow_runtime import _apply_variables as _runtime_apply_variables

    return _runtime_apply_variables(template, variables, prev_output)


def _consolidation_may_apply(
    org_id: str, candidates: list[dict], variables: Optional[dict]
) -> bool:
    """
    Would `context_runtime._maybe_use_kb_consolidation` have substituted a
    consolidated summary for the knowledge-asset sources on this case?

    That function REPLACES raw KB text with a shorter consolidation at runtime.
    Measuring the raw text and calling it reducible would then overstate what is
    there — breaking the floor guarantee — so its applicability has to be known.

    It cannot simply be called: it bumps a hit counter, and this module does not
    write. So only the read-only lookup is performed, and the answer is
    deliberately CONSERVATIVE — the runtime additionally requires a fresh
    version hash and a real saving before substituting, and this returns True
    without checking either. Over-detecting marks the KB components unmeasured,
    which shrinks `reducible_tokens`. A floor that is too low stays a floor.
    """
    asset_ids = [
        c["source_ref"]
        for c in candidates
        if c.get("source_type") == "knowledge_asset" and c.get("source_ref")
    ]
    if not asset_ids:
        return False
    try:
        from synthetic_mind.memory_store import get_kb_consolidation

        rec = get_kb_consolidation(
            org_id,
            asset_ids,
            min_confidence=0.7,
            scope_value=(variables or {}).get("_sm_scope_value"),
        )
        return rec is not None
    except Exception:
        # SM absent or unreachable: the runtime's own except-branch serves the
        # raw candidates in exactly that situation, so raw is what runs.
        return False


def _context_pieces(
    node: dict,
    case_variables: Optional[dict],
    org_id: str,
) -> tuple[list[_Piece], bool]:
    """
    The contextConfig sources, decomposed the way `context_runtime` packages
    them.

    Returns (pieces, complete). `complete` is False when the assembled prompt
    could not be fully reconstructed — a source that did not resolve, an LLM
    summarization this module refuses to perform, or a possible SM consolidation
    substitution. It is what forces `total_tokens` to None.
    """
    from context_runtime import _collect_sources

    data = node.get("data") or {}
    config = data.get("contextConfig")
    if not isinstance(config, dict) or not config.get("enabled"):
        return [], True

    sources_config = [s for s in (config.get("sources") or []) if isinstance(s, dict)]
    if not sources_config:
        return [], True

    # `context` is {} on purpose: resolving a previous_node_output source means
    # executing the upstream node, which this module will not do.
    candidates = _collect_sources(sources_config, {}, org_id, deployment_id=None)

    packaging = config.get("packaging") or {}
    include_labels = packaging.get("includeSourceLabels", False)
    strategy = packaging.get("strategy", "concat")
    max_chars = packaging.get("maxChars")

    # Sections, byte for byte as _package_context builds them.
    sections: list[str] = []
    for c in candidates:
        text = c["raw_text"]
        if include_labels and c.get("label"):
            text = f"## {c['label']}\n{text}"
        sections.append(text)

    joined_len = sum(len(s) for s in sections) + max(0, len(sections) - 1) * len(
        _CONTEXT_SEPARATOR
    )

    complete = True
    consolidation_risk = _consolidation_may_apply(org_id, candidates, case_variables)
    if consolidation_risk:
        complete = False

    # Truncation. _package_context cuts the JOINED text, so a section's
    # surviving span is derived from its offset in that join.
    summarized = False
    survivors: list[Optional[str]] = list(sections)
    if max_chars and joined_len > max_chars:
        if strategy == "summarize":
            # The runtime calls gpt-4o-mini here. Running it would spend money
            # and produce a different summary than the one that ran in
            # production anyway, so the packaged text is simply not observed.
            summarized = True
            complete = False
            survivors = [None] * len(sections)
        else:
            offset = 0
            survivors = []
            for section in sections:
                keep = max(0, min(len(section), max_chars - offset))
                survivors.append(section[:keep])
                offset += len(section) + len(_CONTEXT_SEPARATOR)

    pieces: list[_Piece] = []
    used_names: dict[str, int] = {}
    for candidate, section, survivor in zip(candidates, sections, survivors):
        label = candidate.get("label") or candidate.get("source_type") or "source"
        name = f"{CONTEXT_SOURCE_PREFIX}{label}"
        # Two sources may carry the same label. Disambiguate rather than let one
        # silently absorb the other's tokens.
        seen = used_names.get(name, 0)
        used_names[name] = seen + 1
        if seen:
            name = f"{name}#{seen + 1}"

        if summarized or consolidation_risk:
            # Text entered packaging but its final form was not observed. chars
            # records what went IN; tokens stays None because what came out is
            # unknown.
            pieces.append(
                _Piece(name, candidate.get("source_type"), None, len(section))
            )
        else:
            pieces.append(
                _Piece(
                    name,
                    candidate.get("source_type"),
                    survivor,
                    len(survivor or ""),
                )
            )

    # A configured source that produced no candidate did not vanish from the
    # prompt in production — it vanished from our view of it. Report it.
    resolved = len(candidates)
    if resolved < len(sources_config):
        complete = False
        for src in _unresolved(sources_config, candidates):
            label = src.get("label") or src.get("type") or "source"
            name = f"{CONTEXT_SOURCE_PREFIX}{label}"
            seen = used_names.get(name, 0)
            used_names[name] = seen + 1
            if seen:
                name = f"{name}#{seen + 1}"
            pieces.append(_Piece(name, src.get("type"), None, 0))

    return pieces, complete


def _unresolved(sources_config: list[dict], candidates: list[dict]) -> list[dict]:
    """
    Configured sources with no matching candidate.

    Matched on (type, ref) and consumed once each, so two identical sources of
    which only one resolved are handled correctly.
    """
    remaining: list[tuple[str, Any]] = [
        (c.get("source_type"), c.get("source_ref")) for c in candidates
    ]
    out: list[dict] = []
    for src in sources_config:
        key = (src.get("type"), src.get("assetId") or src.get("nodeId"))
        if key in remaining:
            remaining.remove(key)
        else:
            out.append(src)
    return out


def _decompose_case(node: dict, case: dict, org_id: str) -> tuple[list[_Piece], Optional[str]]:
    """
    One case, split into components, plus the assembled prompt.

    The assembly mirrors `workflow_runtime`'s ai-step branch exactly, including
    the order in which system instructions and context are prepended, because
    the assembled string is what `total_tokens` is measured on. It returns None
    for that string when any component was not observed.
    """
    data = node.get("data") or {}
    node_type = (node.get("type") or "").lower()
    variables = case.get("variables") if isinstance(case.get("variables"), dict) else None
    prev = case.get("input_text") or ""
    if not isinstance(prev, str):
        prev = str(prev)

    if node_type == _MODEL:
        # workflow_runtime passes the previous output straight through as the
        # prompt. No template, no system block, no context. Nothing is static.
        return [_Piece(COMPONENT_USER_INPUT, None, prev, len(prev))], prev

    # ---- task template ---------------------------------------------------
    task_raw = data.get("taskDescription") or data.get("task") or "Respond to the user."
    if not isinstance(task_raw, str):
        task_raw = str(task_raw)

    prompt_text = _apply_variables(task_raw, variables, prev_output=prev)

    # The user's text is either interpolated INTO the template or appended
    # after it. Either way it is accounted once, as `user_input`, and the
    # template is measured with {{input}} blanked so the two do not overlap.
    input_occurrences = task_raw.count("{{input}}")
    task_scaffold = _apply_variables(task_raw, variables, prev_output="")

    input_appended = False
    if input_occurrences == 0 and prev and prev.strip():
        prompt_text = prompt_text + _BLOCK_JOINER + prev
        input_appended = True

    pieces: list[_Piece] = []

    # ---- context ---------------------------------------------------------
    ctx_pieces, ctx_complete = _context_pieces(node, variables, org_id)
    ctx_config = data.get("contextConfig") or {}
    injection = (ctx_config.get("injection") or {}) if isinstance(ctx_config, dict) else {}
    location = injection.get("location", "prepend_to_system")

    observed_ctx = [p.text for p in ctx_pieces if p.text is not None]
    ctx_text: Optional[str] = None
    if ctx_pieces and ctx_complete:
        ctx_text = _CONTEXT_SEPARATOR.join(observed_ctx)
        # context_runtime returns None when the packaged text is blank, and the
        # runtime then injects nothing at all.
        if not ctx_text.strip():
            ctx_text = None

    if ctx_text is not None:
        if location == "append_to_prompt":
            prompt_text = prompt_text + _BLOCK_JOINER + ctx_text
        elif location == "prepend_to_prompt":
            prompt_text = ctx_text + _BLOCK_JOINER + prompt_text

    # ---- system instructions --------------------------------------------
    sys_raw = (data.get("systemInstructions") or data.get("system_prefix") or "")
    if not isinstance(sys_raw, str):
        sys_raw = str(sys_raw)
    sys_raw = sys_raw.strip()
    sys_text = ""
    if sys_raw:
        # The runtime strips FIRST, then interpolates. Order matters when a
        # variable value has leading whitespace.
        sys_text = _apply_variables(sys_raw, variables)
        prompt_text = sys_text + _BLOCK_JOINER + prompt_text
        pieces.append(
            _Piece(COMPONENT_SYSTEM_INSTRUCTIONS, None, sys_text, len(sys_text))
        )

    if ctx_text is not None and location == "prepend_to_system":
        prompt_text = ctx_text + _BLOCK_JOINER + prompt_text

    pieces.extend(ctx_pieces)
    pieces.append(
        _Piece(COMPONENT_TASK_DESCRIPTION, None, task_scaffold, len(task_scaffold))
    )

    # ---- user input ------------------------------------------------------
    if input_appended or input_occurrences:
        contributed = len(prev) * max(1, input_occurrences)
        if input_occurrences > 1:
            # Interpolated more than once. The chars are exact arithmetic on a
            # measured length, but a per-occurrence token count is not separable
            # from the surrounding template, so it is not claimed.
            pieces.append(_Piece(COMPONENT_USER_INPUT, None, None, contributed))
        else:
            pieces.append(_Piece(COMPONENT_USER_INPUT, None, prev, contributed))

    assembled: Optional[str] = prompt_text if ctx_complete else None
    return pieces, assembled


# ---------------------------------------------------------------------------
# Graph access (read-only)
# ---------------------------------------------------------------------------

def _load_graph(org_id: str, workload: dict, workflow_id: str) -> Optional[dict]:
    """
    The configuration currently in force for this workload.

    Mirrors `optimization.benchmark._load_baseline_graph`: the promoted
    deployment is what production serves, and the workflow draft is the
    fallback. Deliberately duplicated rather than imported, because
    `optimization.candidates` imports this module and `benchmark` imports
    `candidates` — importing it here would close that cycle.
    """
    try:
        resp = (
            supabase.table("workflow_deployments")
            .select("id, workflow_id, org_id, version, endpoint_slug, graph_json, status")
            .eq("org_id", org_id)
            .eq("workflow_id", workflow_id)
            .eq("status", "promoted")
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if rows and rows[0].get("graph_json"):
            return rows[0]["graph_json"]
    except Exception as exc:
        logger.warning(
            "context_accounting deployment lookup failed: %s", type(exc).__name__
        )

    try:
        resp = (
            supabase.table("workflows")
            .select("id, org_id, graph_json")
            .eq("id", workflow_id)
            .eq("org_id", org_id)
            .limit(1)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if rows:
            return rows[0].get("graph_json") or {"nodes": [], "edges": []}
    except Exception as exc:
        logger.warning(
            "context_accounting workflow lookup failed: %s", type(exc).__name__
        )

    return None


def _select_node(graph: dict, step_id: Optional[str]) -> Optional[dict]:
    """
    The node to profile.

    With a step_id, exactly that node (step ids ARE node ids — see
    `optimization.strategy.from_graph_json`). Without one, the first profilable
    node in graph order, so repeated calls agree.
    """
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    if step_id is not None:
        for node in nodes:
            if str(node.get("id") or "") == str(step_id):
                if (node.get("type") or "").lower() in PROFILABLE_NODE_TYPES:
                    return node
                return None
        return None
    for node in nodes:
        if (node.get("type") or "").lower() in PROFILABLE_NODE_TYPES:
            return node
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile_workload_context(
    org_id: str,
    workload: dict,
    *,
    cases: list[dict],
    step_id: Optional[str] = None,
) -> Optional[ContextProfile]:
    """
    Decompose one step's prompt into measured components.

    `cases` are replay cases in the `golden_inputs` shape — the same rows
    `optimization.benchmark` replays, so the profile describes the customer's
    real recorded inputs rather than a synthetic sample. Only `input_text` and
    `variables` are read.

    Returns None when there is nothing to profile: no workflow behind the
    workload, no graph, no profilable node, no cases, or not a single case that
    could be reproduced. A profile with unmeasurable components is still
    RETURNED — with the measurable ones populated and `coverage` reflecting what
    actually happened — because "this much is measured and that part is not" is
    the useful answer.
    """
    if not cases:
        return None

    from optimization import workloads as workloads_mod

    workflow_id = workloads_mod.resolve_workflow_id(org_id, workload)
    if not workflow_id:
        # Direct-inference workloads have no graph. Nothing to decompose here.
        return None

    graph = _load_graph(org_id, workload, workflow_id)
    if not graph:
        return None

    node = _select_node(graph, step_id)
    if node is None:
        return None

    node_id = str(node.get("id") or "")
    data = node.get("data") or {}
    enc, encoding_name = resolve_encoding(
        data.get("modelName") or data.get("model"), data.get("provider")
    )

    sample = list(cases)[:MAX_SAMPLE_CASES]
    n_cases = len(sample)

    per_case: list[list[_Piece]] = []
    assembled_totals: list[int] = []
    any_assembly_unobserved = False

    for case in sample:
        if not isinstance(case, dict):
            continue
        try:
            pieces, assembled = _decompose_case(node, case, org_id)
        except Exception as exc:
            # A case that cannot be reproduced is DROPPED FROM COVERAGE, never
            # silently treated as if it had matched the others.
            logger.warning(
                "context profile: case %s not reproducible: %s",
                case.get("id"),
                type(exc).__name__,
            )
            continue
        per_case.append(pieces)
        if assembled is None:
            any_assembly_unobserved = True
        else:
            n = _count(enc, assembled)
            if n is None:
                any_assembly_unobserved = True
            else:
                assembled_totals.append(n)

    profiled = len(per_case)
    if profiled == 0:
        return None

    components = _aggregate(per_case, profiled, enc, encoding_name)

    total_tokens: Optional[int] = None
    if not any_assembly_unobserved and assembled_totals:
        total_tokens = round(sum(assembled_totals) / len(assembled_totals))

    reducible = _reducible(components)

    return ContextProfile(
        workload_id=str(workload.get("id") or ""),
        step_id=node_id,
        n_cases=n_cases,
        components=tuple(components),
        total_tokens=total_tokens,
        reducible_tokens=reducible,
        tokenizer=encoding_name,
        coverage=(profiled / n_cases) if n_cases else 0.0,
    )


def _aggregate(
    per_case: list[list[_Piece]],
    profiled: int,
    enc,
    encoding_name: Optional[str],
) -> list[ComponentTokens]:
    """
    Collapse per-case observations into one row per component.

    `static` is decided here, by comparing the observed text across cases — the
    only place it CAN be decided, since provenance says nothing about whether a
    template interpolates.
    """
    order: list[str] = []
    seen: dict[str, list[_Piece]] = {}
    for pieces in per_case:
        for piece in pieces:
            if piece.name not in seen:
                seen[piece.name] = []
                order.append(piece.name)
            seen[piece.name].append(piece)

    out: list[ComponentTokens] = []
    for name in order:
        observations = seen[name]
        source_type = next(
            (o.source_type for o in observations if o.source_type), None
        )
        texts = [o.text for o in observations]
        observed = [t for t in texts if t is not None]

        # Static requires: seen on every profiled case, observed on every one of
        # them, and byte-identical throughout.
        static = (
            len(observations) == profiled
            and len(observed) == profiled
            and all(t == observed[0] for t in observed)
        )

        if static:
            tokens = _count(enc, observed[0])
            chars = len(observed[0])
        elif observed and len(observed) == len(observations):
            counts = [_count(enc, t) for t in observed]
            tokens = (
                round(sum(c for c in counts if c is not None) / len(counts))
                if all(c is not None for c in counts)
                else None
            )
            chars = round(sum(o.chars for o in observations) / len(observations))
        else:
            # At least one case did not expose this component's text, so no
            # token count is claimed for it at all.
            tokens = None
            chars = round(sum(o.chars for o in observations) / len(observations))

        # user_input is never static in the sense that matters, and must never
        # be proposed for removal. It is measured like everything else, but the
        # flag is not allowed to invite cutting it.
        if name == COMPONENT_USER_INPUT:
            static = False

        out.append(
            ComponentTokens(
                component=name,
                source_type=source_type,
                tokens=tokens,
                chars=chars,
                tokenizer=(encoding_name if tokens is not None else None),
                static=static,
            )
        )
    return out


def _reducible(components: list[ComponentTokens]) -> Optional[int]:
    """
    Tokens present on EVERY request that are not the user's own input.

    A floor. None — not zero, not a partial sum — if any qualifying component
    could not be tokenized, because a partial sum would read as the whole and
    understate nothing visibly.
    """
    qualifying = [
        c for c in components if c.static and c.component != COMPONENT_USER_INPUT
    ]
    if not qualifying:
        return 0
    if any(c.tokens is None for c in qualifying):
        return None
    return sum(c.tokens or 0 for c in qualifying)
