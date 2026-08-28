"""
Execution strategies: HOW a workload should be attempted.

A Strategy is an ordered list of STEPS, each bound to an EXECUTOR. It is the
unit of comparison for a recommendation — baseline strategy vs candidate
strategy, never merely "model A vs model B". A strategy with a deterministic
pre-classifier, then a model, then a human approver is expressible today; only
model steps actually execute today, and steps whose executor cannot execute are
reported rather than silently dropped.

Runtime adapter
---------------
`from_graph_json` reads a workflow graph into a Strategy, and `apply_to_graph`
writes a Strategy back into a graph. Both are HONEST about what this codebase
can really change:

APPLICABLE today (verified against workflow_runtime.py / context_runtime.py):
    model            ai-step / model node  data.modelName
    provider         ai-step / model node  data.provider
    prompt           ai-step node          data.taskDescription | data.task,
                                           data.systemInstructions | data.system_prefix
    context_length   any node              data.contextConfig.packaging.maxChars
                                           and per-source data.contextConfig.sources[].maxChars
                                           (applied in context_runtime.py)
    fallback_chain   router node           data.strategy
                                           ('cheapest'|'fastest'|'balanced'|'fallback')

NOT applicable ON THIS SURFACE — and deliberately NOT faked:
    temperature,     workflow_runtime._execute_model_node builds a payload with
    max_tokens,      only (org_id, provider, model, prompt, prompt_id). None of
    top_p            these ever reaches a provider through a workflow graph;
                     anthropic_router hardcodes max_tokens=1024. Setting them on
                     a node would change the graph and change nothing about the
                     execution, producing a benchmark that "proves" a difference
                     that cannot exist. Refused at apply time.
                     *** They ARE applicable on the direct_inference surface, ***
                     where the OpenAI dialect forwards the body and the Anthropic
                     translation maps them explicitly. Applicability is scoped
                     per surface — see SURFACE_APPLICABLE_DIMENSIONS.
    caching          There is no cache node or cache layer in workflow_runtime.
    reasoning_effort No node or router passes a reasoning-effort parameter.
    retrieval,       Deferred: the existing retrieval code does not expose a
    reranking        safely-substitutable knob beyond maxChars.
    llm_call_count,  Structural rewrites. Representable in `dimensions` for a
    workflow_structure future generator, but this module will not synthesise a
    tool_selection,  restructured graph, because it cannot verify the rewrite
    deterministic_code preserves the workflow's contract.

Attempting to apply an unsupported dimension raises UnsupportedDimension. That
is the point: a dimension that cannot be applied must never reach a benchmark.

Context efficiency
------------------
`prompt` and `context_length` together express one deliberately narrow change:
the same model, the same provider, the same sampling parameters, the same tools
and the same output contract, carrying less context. Building such a variant and
PROVING it is only such a variant are both here — `with_context_budget` /
`with_prompt_text` construct one, `context_only_change` checks the strategies
and `graph_diff_outside_context` checks the applied graphs. The graph-level
check is the load-bearing one: a Strategy cannot express a node's tools or its
declared output schema at all, so only the graph comparison can say those did
not move.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from optimization import domain

# ---------------------------------------------------------------------------
# What Runtime can really change
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════════
# Applicability is a property of the EXECUTION SURFACE, not a global constant.
# ═══════════════════════════════════════════════════════════════════════════
#
# The same dimension can be real on one surface and fictional on another, and
# the honest answer is per-surface precision — not a single list that is wrong
# half the time.
#
# `temperature` is the clearest case:
#   * runtime          — DROPPED. workflow_runtime._execute_model_node builds a
#                        payload of only (org_id, provider, model, prompt,
#                        prompt_id). The value never reaches a provider, so
#                        varying it would change the graph, change nothing about
#                        the execution, and produce a benchmark that "proves" a
#                        difference that cannot exist.
#   * direct_inference — REAL. The OpenAI dialect forwards the body, and the
#                        Anthropic translation maps temperature/top_p/max_tokens
#                        explicitly (direct_inference.py). The value lands in the
#                        outbound payload.
#
# So on direct inference these are genuinely benchmarkable optimization
# dimensions, and a candidate generator may propose changing them.

SURFACE_RUNTIME = "runtime"
SURFACE_DIRECT_INFERENCE = "direct_inference"

#: Dimensions each surface can actually apply.
SURFACE_APPLICABLE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    SURFACE_RUNTIME: (
        "model",
        "provider",
        "prompt",
        "context_length",
        "fallback_chain",
    ),
    SURFACE_DIRECT_INFERENCE: (
        "model",
        "provider",
        "prompt",
        "temperature",
        "max_tokens",
        "top_p",
    ),
}

#: Why a dimension cannot be applied, per surface. Surfaced verbatim in errors
#: and API responses so a caller is never left guessing.
_SHARED_UNSUPPORTED = {
    "reasoning_effort": "No node, router or dialect passes a reasoning-effort parameter.",
    "retrieval": "No safely-substitutable retrieval knob beyond a character budget.",
    "reranking": "No reranking stage exists.",
    "llm_call_count": "Requires a structural rewrite; not synthesised here.",
    "workflow_structure": "Requires a structural rewrite; not synthesised here.",
    "tool_selection": "Tool bindings are not modelled as a substitutable step yet.",
    "deterministic_code": (
        "Requires a 'software' executor able to execute; executors of that type "
        "can be registered but cannot yet be executed."
    ),
}

SURFACE_UNSUPPORTED_DIMENSIONS: dict[str, dict[str, str]] = {
    SURFACE_RUNTIME: {
        **_SHARED_UNSUPPORTED,
        "temperature": (
            "workflow_runtime._execute_model_node sends only (org_id, provider, "
            "model, prompt, prompt_id) to the router; temperature never reaches a "
            "provider on this surface. It IS applicable on direct_inference."
        ),
        "max_tokens": (
            "Not plumbed through workflow_runtime; anthropic_router hardcodes "
            "max_tokens=1024. It IS applicable on direct_inference."
        ),
        "top_p": (
            "Not plumbed through workflow_runtime. It IS applicable on "
            "direct_inference."
        ),
        "caching": "There is no cache node or response cache in workflow_runtime.",
    },
    SURFACE_DIRECT_INFERENCE: {
        **_SHARED_UNSUPPORTED,
        "context_length": (
            "OptiML does not assemble context on the direct-inference surface — "
            "the caller sends the full message list — so there is no budget to "
            "vary. It IS applicable on runtime, via contextConfig.packaging.maxChars."
        ),
        "fallback_chain": (
            "There is no router node on this surface; a direct request targets one "
            "model. It IS applicable on runtime, via the router node's strategy."
        ),
        "caching": "No response cache sits in front of the direct-inference path.",
    },
}

#: Back-compatible module-level aliases. The runtime surface is the default
#: because `apply_to_graph` only ever operates on a workflow graph.
APPLICABLE_DIMENSIONS = SURFACE_APPLICABLE_DIMENSIONS[SURFACE_RUNTIME]
UNSUPPORTED_DIMENSIONS = SURFACE_UNSUPPORTED_DIMENSIONS[SURFACE_RUNTIME]


def applicable_dimensions(surface: str = SURFACE_RUNTIME) -> tuple[str, ...]:
    """Dimensions this surface can genuinely vary."""
    return SURFACE_APPLICABLE_DIMENSIONS.get(surface, ())


def unsupported_dimensions(surface: str = SURFACE_RUNTIME) -> dict[str, str]:
    """Dimensions this surface cannot vary, mapped to why."""
    return SURFACE_UNSUPPORTED_DIMENSIONS.get(surface, dict(_SHARED_UNSUPPORTED))


#: Strategy-config keys that carry a dimension value, so `_assert_supported`
#: can tell "this config sets temperature" from "this config has a node_type".
#: Strategy-config key holding DERIVED node facts a change must preserve. It is
#: deliberately absent from `_CONFIG_KEY_DIMENSION` (it configures nothing, so
#: it can never be refused as an unapplicable dimension), absent from the
#: fingerprint (it is derived, so it must not make two identical strategies look
#: different) and never written back by `apply_to_graph`.
CONFIG_KEY_INVARIANTS = "invariants"

#: Config keys that are DERIVED FROM the graph rather than chosen by a strategy.
#: Excluded from the fingerprint so that adding a derived fact — as
#: `invariants` was — cannot silently change the fingerprint of every strategy
#: in the system and break dedup against strategies recorded before it existed.
DERIVED_CONFIG_KEYS = (CONFIG_KEY_INVARIANTS,)

_CONFIG_KEY_DIMENSION = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "top_p": "top_p",
    "task_description": "prompt",
    "system_instructions": "prompt",
    "context": "context_length",
    "router_strategy": "fallback_chain",
    "caching": "caching",
    "reasoning_effort": "reasoning_effort",
}


LLM_NODE_TYPES = ("ai-step", "model")
ROUTER_NODE_TYPE = "router"

ROUTER_STRATEGIES = ("cheapest", "fastest", "balanced", "fallback")

STEP_ROLES = ("primary", "fallback", "verifier", "approver", "preprocessor", "router")


class UnsupportedDimension(ValueError):
    """Raised when a strategy asks for a change this runtime cannot make."""


class StrategyApplyError(ValueError):
    """Raised when a strategy cannot be applied to the given graph."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StrategyStep:
    """
    One step of work, bound to an executor.

    `executor_ref` names the executor structurally so a step can reference an
    executor that has not been registered in the `executors` table yet;
    `executor_id` links it once it has. Nothing about this shape is model
    specific — an agent step differs only in executor_type and config.
    """

    step_id: str
    order: int
    executor_type: str = "model"
    executor_ref: dict = field(default_factory=dict)
    executor_id: Optional[str] = None
    role: str = "primary"
    config: dict = field(default_factory=dict)
    on_failure: str = "fail"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "StrategyStep":
        return StrategyStep(
            step_id=str(d.get("step_id") or ""),
            order=int(d.get("order") or 0),
            executor_type=(d.get("executor_type") or "model"),
            executor_ref=d.get("executor_ref") or {},
            executor_id=(str(d["executor_id"]) if d.get("executor_id") else None),
            role=(d.get("role") or "primary"),
            config=d.get("config") or {},
            on_failure=(d.get("on_failure") or "fail"),
        )

    @property
    def is_executable_today(self) -> bool:
        """Only model steps can actually be executed by workflow_runtime."""
        return self.executor_type == "model"


@dataclass
class Strategy:
    """An ordered, multi-executor answer to 'how should this work be attempted?'"""

    steps: list[StrategyStep] = field(default_factory=list)
    surface: str = "runtime"
    surface_binding: dict = field(default_factory=dict)
    dimensions: list[str] = field(default_factory=list)

    # ---- serialisation ----

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "steps": [s.to_dict() for s in sorted(self.steps, key=lambda x: x.order)],
            "surface_binding": self.surface_binding,
            "dimensions": list(self.dimensions),
            "fingerprint": self.fingerprint(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Strategy":
        return Strategy(
            steps=[StrategyStep.from_dict(s) for s in (d.get("steps") or [])],
            surface=(d.get("surface") or "runtime"),
            surface_binding=d.get("surface_binding") or {},
            dimensions=list(d.get("dimensions") or []),
        )

    def fingerprint(self) -> str:
        """
        Stable hash of the semantically meaningful content, for deduping
        identical candidates produced by different generators.

        DERIVED_CONFIG_KEYS are excluded. A derived fact — the node's tool
        names, its declared output contract — is read off the graph rather than
        chosen by the strategy, so hashing it would make two strategies that
        propose exactly the same change look different, and would have shifted
        every fingerprint in the system on the day the derived key was added.
        """
        payload = {
            "surface": self.surface,
            "steps": [
                {
                    "step_id": s.step_id,
                    "order": s.order,
                    "executor_type": s.executor_type,
                    "executor_ref": s.executor_ref,
                    "role": s.role,
                    "config": {
                        k: v for k, v in (s.config or {}).items()
                        if k not in DERIVED_CONFIG_KEYS
                    },
                    "on_failure": s.on_failure,
                }
                for s in sorted(self.steps, key=lambda x: (x.order, x.step_id))
            ],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ---- descriptors ----

    @property
    def llm_call_count(self) -> int:
        return sum(1 for s in self.steps if s.executor_type == "model")

    @property
    def executor_types(self) -> list[str]:
        return sorted({s.executor_type for s in self.steps})

    @property
    def unexecutable_steps(self) -> list[StrategyStep]:
        """
        Steps whose executor kind cannot run in this runtime today. Reported,
        never silently dropped — a benchmark over a strategy with unexecutable
        steps would not be measuring the strategy.
        """
        return [s for s in self.steps if not s.is_executable_today]

    def step(self, step_id: str) -> Optional[StrategyStep]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None


# ---------------------------------------------------------------------------
# Runtime adapter: graph_json <-> Strategy
# ---------------------------------------------------------------------------

def _graph_shape_hash(graph: dict) -> str:
    """Hash of node types + edges: what the graph DOES, ignoring configuration."""
    nodes = sorted(
        (str(n.get("id")), (n.get("type") or "").lower())
        for n in (graph.get("nodes") or [])
        if n.get("id")
    )
    edges = sorted(
        (str(e.get("source")), str(e.get("target")))
        for e in (graph.get("edges") or [])
        if e.get("source") and e.get("target")
    )
    blob = json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _context_config_summary(data: dict) -> Optional[dict]:
    cfg = data.get("contextConfig")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    packaging = cfg.get("packaging") or {}
    sources = cfg.get("sources") or []
    src = [s for s in sources if isinstance(s, dict)]
    return {
        "enabled": True,
        "packaging_max_chars": packaging.get("maxChars"),
        "source_count": len(sources),
        "source_max_chars": [s.get("maxChars") for s in src],
        "mode": cfg.get("mode"),
        # Labels, types and required-ness are carried so a context-reduction
        # candidate can check its budget against the sources the workload
        # declares REQUIRED, using the measured per-source size from
        # optimization.context_accounting (whose component labels are
        # 'context_source:<label>'). They are DESCRIPTIVE: `_apply_context`
        # never writes them, and `diff_dimensions` never reads them, so adding
        # a source or changing its type can never be mistaken for a
        # context_length change.
        "source_labels": [s.get("label") or s.get("type") for s in src],
        "source_types": [s.get("type") for s in src],
        "source_required": [bool(s.get("required")) for s in src],
    }


#: Placeholder syntax the runtime actually interpolates. Must stay identical to
#: `workflow_runtime._apply_variables`, which is a strictly single-pass
#: `re.sub(r'\{\{(\w+)\}\}', ...)`. A placeholder this regex does not match is
#: not interpolated by the runtime either, so matching the runtime exactly is
#: what makes "every placeholder survived" a true statement rather than a
#: hopeful one.
_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def placeholders_in(*texts: Optional[str]) -> set[str]:
    """
    Every placeholder the runtime would interpolate in the given text(s).

    Used by the context-reduction static checks: a shorter prompt that has lost
    `{{input}}` does not execute the same workload, and that is discoverable
    here, before a provider request exists, rather than from a benchmark whose
    candidate arm silently ran on different information.
    """
    found: set[str] = set()
    for text in texts:
        if isinstance(text, str) and text:
            found.update(_PLACEHOLDER_RE.findall(text))
    return found


def _node_invariants(data: dict) -> dict:
    """
    What a strategy change must PRESERVE, read off the node itself.

    These are facts about the node, not choices a strategy makes, so they are
    excluded from the fingerprint (see `Strategy.fingerprint`) and are never
    applied back onto a graph. They exist so a candidate generator can prove —
    statically, against real node data — that a reduced prompt still references
    every bound tool and still carries the node's declared output contract.

    `tool_names` mirrors `workflow_runtime`'s own construction: a tool without a
    `name` is not sent to the provider, so it is not an invariant either.
    """
    tools = data.get("tools") or []
    tool_names = [
        str(t["name"]) for t in tools
        if isinstance(t, dict) and t.get("name")
    ]

    # The node's DECLARED output contract, if it declares one. Absent on most
    # nodes today; absence is recorded as an empty structure rather than as a
    # claim that the workload has no output contract, because a contract stated
    # in prose inside the prompt is still a contract — that half is detected
    # from the prompt text by the generator's own marker extraction.
    declared: dict[str, Any] = {}
    for key in ("outputSchema", "output_schema", "responseFormat", "response_format"):
        value = data.get(key)
        if value:
            declared[key] = value

    return {
        "tool_names": tool_names,
        "declared_output_contract": declared,
        "placeholders": sorted(
            placeholders_in(
                data.get("taskDescription"),
                data.get("task"),
                data.get("systemInstructions"),
                data.get("system_prefix"),
            )
        ),
    }


def from_graph_json(graph: dict, *, workflow_id: Optional[str] = None) -> Strategy:
    """
    Extract a Strategy from a workflow graph.

    Every node that performs work becomes a step. Node ids become step ids, so
    `apply_to_graph` can map a step back onto exactly the node it came from.
    Only configuration this codebase actually honours is captured — reading a
    `temperature` off a node and calling it strategy would imply it matters.
    """
    nodes = graph.get("nodes") or []
    steps: list[StrategyStep] = []
    order = 0

    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        node_type = (node.get("type") or "").lower()
        data = node.get("data") or {}

        if node_type in LLM_NODE_TYPES:
            config: dict[str, Any] = {
                "node_type": node_type,
            }
            if node_type == "ai-step":
                config["task_description"] = data.get("taskDescription") or data.get("task")
                config["system_instructions"] = (
                    data.get("systemInstructions") or data.get("system_prefix")
                )
            ctx = _context_config_summary(data)
            if ctx:
                config["context"] = ctx
            # DERIVED, never applied. See `_node_invariants` and the fingerprint
            # note: these describe the node, they do not configure it.
            config[CONFIG_KEY_INVARIANTS] = _node_invariants(data)

            steps.append(
                StrategyStep(
                    step_id=node_id,
                    order=order,
                    executor_type="model",
                    executor_ref={
                        "executor_type": "model",
                        "vendor": (data.get("provider") or "").strip().lower() or None,
                        "external_id": (data.get("modelName") or data.get("model") or "").strip()
                        or None,
                    },
                    role="primary",
                    config=config,
                    on_failure="fail",
                )
            )
            order += 1

        elif node_type == ROUTER_NODE_TYPE:
            steps.append(
                StrategyStep(
                    step_id=node_id,
                    order=order,
                    # A router is deterministic selection logic executed by
                    # OptiML, not a model call.
                    executor_type="software",
                    executor_ref={
                        "executor_type": "software",
                        "vendor": "optiml",
                        "external_id": "workflow_router",
                    },
                    role="router",
                    config={
                        "node_type": node_type,
                        "router_strategy": (
                            data.get("strategy") or data.get("primaryModel") or "balanced"
                        ),
                    },
                    on_failure="fail",
                )
            )
            order += 1

    return Strategy(
        steps=steps,
        surface="runtime",
        surface_binding={
            "kind": "workflow_graph",
            "workflow_id": workflow_id,
            "graph_shape_hash": _graph_shape_hash(graph),
            "node_count": len(nodes),
            "llm_call_count": sum(
                1 for n in nodes if (n.get("type") or "").lower() in LLM_NODE_TYPES
            ),
        },
        dimensions=[],
    )


def _assert_supported(config: dict, surface: str = SURFACE_RUNTIME) -> None:
    """
    Refuse a config that sets a dimension THIS surface cannot vary.

    Refusing is the point. A silently-ignored change would produce a candidate
    identical to the baseline and a benchmark that measures nothing while
    reporting a result.
    """
    unsupported = unsupported_dimensions(surface)
    for key in config:
        dimension = _CONFIG_KEY_DIMENSION.get(key, key)
        if dimension in unsupported:
            raise UnsupportedDimension(
                f"Cannot apply '{dimension}' on the '{surface}' surface: "
                f"{unsupported[dimension]}"
            )


def apply_to_graph(graph: dict, strategy: Strategy) -> dict:
    """
    Produce a candidate graph by applying a Strategy onto a baseline graph.

    Returns a NEW graph; the input is never mutated. Raises:
      UnsupportedDimension — the strategy asks for something this runtime
                             cannot actually do (see UNSUPPORTED_DIMENSIONS).
      StrategyApplyError   — a step references a node that is not in the graph,
                             or a value is outside the vocabulary the runtime
                             accepts.

    Failing loudly is deliberate. A silently-ignored change would produce a
    candidate graph identical to the baseline and a benchmark that measures
    nothing while reporting a result.
    """
    if strategy.surface != "runtime":
        raise StrategyApplyError(
            f"Strategy surface '{strategy.surface}' cannot be applied to a workflow graph."
        )

    out = copy.deepcopy(graph)
    nodes_by_id = {str(n.get("id")): n for n in (out.get("nodes") or []) if n.get("id")}

    for step in sorted(strategy.steps, key=lambda s: s.order):
        # Always the runtime surface here: this function writes a workflow graph.
        _assert_supported(step.config or {}, SURFACE_RUNTIME)

        node = nodes_by_id.get(step.step_id)
        if node is None:
            raise StrategyApplyError(
                f"Strategy step '{step.step_id}' does not match any node in the graph. "
                "The strategy was built from a different graph version."
            )

        if step.executor_type not in ("model", "software"):
            raise StrategyApplyError(
                f"Step '{step.step_id}' uses executor_type '{step.executor_type}', which "
                "cannot execute in the runtime surface. Registering such an executor is "
                "supported; executing one is not."
            )

        data = node.setdefault("data", {})
        node_type = (node.get("type") or "").lower()
        cfg = step.config or {}

        if step.executor_type == "model":
            if node_type not in LLM_NODE_TYPES:
                raise StrategyApplyError(
                    f"Step '{step.step_id}' is a model step but node type is '{node_type}'."
                )
            ref = step.executor_ref or {}
            if ref.get("external_id"):
                data["modelName"] = str(ref["external_id"]).strip()
            if ref.get("vendor"):
                data["provider"] = str(ref["vendor"]).strip()

            if node_type == "ai-step":
                if cfg.get("task_description") is not None:
                    data["taskDescription"] = str(cfg["task_description"])
                    # workflow_runtime reads taskDescription first, then task.
                    # Clear the stale alternate so the applied prompt is the
                    # one that actually executes.
                    data.pop("task", None)
                if cfg.get("system_instructions") is not None:
                    data["systemInstructions"] = str(cfg["system_instructions"])
                    data.pop("system_prefix", None)

            ctx = cfg.get("context")
            if isinstance(ctx, dict):
                _apply_context(data, ctx, step.step_id)

        elif step.role == "router":
            if node_type != ROUTER_NODE_TYPE:
                raise StrategyApplyError(
                    f"Step '{step.step_id}' is a router step but node type is '{node_type}'."
                )
            rs = cfg.get("router_strategy")
            if rs is not None:
                rs = str(rs).strip().lower()
                if rs not in ROUTER_STRATEGIES:
                    raise StrategyApplyError(
                        f"Unknown router strategy '{rs}'. Runtime accepts {ROUTER_STRATEGIES}."
                    )
                data["strategy"] = rs

    return out


def _apply_context(data: dict, ctx: dict, step_id: str) -> None:
    """
    Apply the context_length dimension.

    Only maxChars budgets are touched: those are read and enforced by
    context_runtime._package_context and _collect_sources. Sources are never
    added or removed here — that would change what information the workload has
    access to, which is a retrieval change, not a token-budget change.
    """
    existing = data.get("contextConfig")
    if not isinstance(existing, dict) or not existing.get("enabled"):
        raise StrategyApplyError(
            f"Step '{step_id}' sets a context budget but node has no enabled contextConfig."
        )

    if "packaging_max_chars" in ctx and ctx["packaging_max_chars"] is not None:
        budget = int(ctx["packaging_max_chars"])
        if budget <= 0:
            raise StrategyApplyError(
                f"Step '{step_id}': context packaging_max_chars must be > 0."
            )
        packaging = existing.setdefault("packaging", {})
        packaging["maxChars"] = budget

    source_budgets = ctx.get("source_max_chars")
    if isinstance(source_budgets, list) and source_budgets:
        sources = existing.get("sources") or []
        if len(source_budgets) != len(sources):
            raise StrategyApplyError(
                f"Step '{step_id}': {len(source_budgets)} source budgets for "
                f"{len(sources)} sources. The strategy was built from a different graph."
            )
        for src, budget in zip(sources, source_budgets):
            if budget is None or not isinstance(src, dict):
                continue
            src["maxChars"] = int(budget)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def diff_dimensions(baseline: Strategy, candidate: Strategy) -> list[str]:
    """
    Which optimization dimensions actually differ between two strategies.

    Only reports dimensions that were genuinely changed. An empty result means
    the candidate is the baseline, and a recommendation with no dimensions must
    never be benchmarked — there is nothing to measure.
    """
    changed: set[str] = set()

    base_by_id = {s.step_id: s for s in baseline.steps}
    cand_by_id = {s.step_id: s for s in candidate.steps}

    if set(base_by_id) != set(cand_by_id):
        changed.add("workflow_structure")

    for step_id, cand in cand_by_id.items():
        base = base_by_id.get(step_id)
        if base is None:
            continue

        b_ref, c_ref = base.executor_ref or {}, cand.executor_ref or {}
        if (b_ref.get("external_id") or None) != (c_ref.get("external_id") or None):
            changed.add("model")
        if (b_ref.get("vendor") or None) != (c_ref.get("vendor") or None):
            changed.add("provider")

        b_cfg, c_cfg = base.config or {}, cand.config or {}
        # Sampling parameters. Real on direct_inference, refused on runtime —
        # detecting a change here is correct on both, because the refusal
        # happens at apply time and must be able to see the difference.
        for key in ("temperature", "max_tokens", "top_p"):
            if b_cfg.get(key) != c_cfg.get(key):
                changed.add(key)
        if (b_cfg.get("task_description") or None) != (c_cfg.get("task_description") or None):
            changed.add("prompt")
        if (b_cfg.get("system_instructions") or None) != (c_cfg.get("system_instructions") or None):
            changed.add("prompt")
        if (b_cfg.get("router_strategy") or None) != (c_cfg.get("router_strategy") or None):
            changed.add("fallback_chain")

        b_ctx = b_cfg.get("context") or {}
        c_ctx = c_cfg.get("context") or {}
        if b_ctx.get("packaging_max_chars") != c_ctx.get("packaging_max_chars"):
            changed.add("context_length")
        if (b_ctx.get("source_max_chars") or []) != (c_ctx.get("source_max_chars") or []):
            changed.add("context_length")

    if baseline.llm_call_count != candidate.llm_call_count:
        changed.add("llm_call_count")

    # Preserve the canonical dimension ordering.
    return [d for d in domain.DIMENSIONS if d in changed]


# ---------------------------------------------------------------------------
# Context efficiency — building a variant, and PROVING it is only a variant
# ---------------------------------------------------------------------------
#
# The claim "same model, same provider, same sampling parameters, same tools,
# same output contract — only the context representation changed" is the whole
# value of this dimension, so it is made checkable in two places rather than
# asserted in a docstring:
#
#   `context_only_change`          at the STRATEGY level: no dimension other
#                                  than prompt/context_length differs.
#   `graph_diff_outside_context`   at the GRAPH level, after apply_to_graph:
#                                  no node field other than the prompt text and
#                                  the contextConfig budgets differs.
#
# The second is the stronger of the two, because it inspects what will actually
# be sent rather than what the strategy claims.

#: Node-data keys a context-reduction variant is permitted to differ on. Every
#: other key — modelName, provider, tools, temperature, outputSchema, anything a
#: future node grows — must be byte-identical between the two graphs.
CONTEXT_MUTABLE_NODE_KEYS = (
    "taskDescription",
    "task",
    "systemInstructions",
    "system_prefix",
    "contextConfig",
)

#: Within contextConfig, the ONLY paths a context-reduction variant may change.
#: Sources are never added, removed or retyped: that would change what
#: information the workload can see, which is a retrieval change wearing a
#: token-budget costume.
CONTEXT_MUTABLE_CONFIG_PATHS = (
    "packaging.maxChars",
    "sources[].maxChars",
)


def with_context_budget(
    baseline: "Strategy",
    step_id: str,
    *,
    packaging_max_chars: Optional[int] = None,
    source_max_chars: Optional[list] = None,
) -> "Strategy":
    """
    Clone the baseline with ONE step's context budget lowered.

    Nothing else is touched — not the model, not the provider, not a single
    character of the prompt. This is the cheapest and most defensible context
    variant there is: a pure budget change that the runtime already enforces in
    `context_runtime._package_context` and `_collect_sources`.

    Raises StrategyApplyError if the step has no enabled context configuration,
    because a budget change on a step that assembles no context would be a
    candidate identical to the baseline.
    """
    step = baseline.step(step_id)
    if step is None:
        raise StrategyApplyError(f"Strategy has no step '{step_id}'.")
    ctx = (step.config or {}).get("context")
    if not isinstance(ctx, dict) or not ctx.get("enabled"):
        raise StrategyApplyError(
            f"Step '{step_id}' has no enabled context configuration to budget."
        )

    new_ctx = dict(ctx)
    if packaging_max_chars is not None:
        new_ctx["packaging_max_chars"] = int(packaging_max_chars)
    if source_max_chars is not None:
        new_ctx["source_max_chars"] = list(source_max_chars)

    return _with_step_config(baseline, step_id, {"context": new_ctx})


def with_prompt_text(
    baseline: "Strategy",
    step_id: str,
    *,
    task_description: Optional[str] = None,
    system_instructions: Optional[str] = None,
) -> "Strategy":
    """
    Clone the baseline with ONE step's static prompt text replaced.

    `None` means "leave this field exactly as it was" — it is not a way to
    delete a prompt field, because deleting one is not a reduction, it is a
    different workload.
    """
    if baseline.step(step_id) is None:
        raise StrategyApplyError(f"Strategy has no step '{step_id}'.")
    patch: dict[str, Any] = {}
    if task_description is not None:
        patch["task_description"] = str(task_description)
    if system_instructions is not None:
        patch["system_instructions"] = str(system_instructions)
    if not patch:
        raise StrategyApplyError(
            f"Step '{step_id}': no prompt text supplied, so nothing would change."
        )
    return _with_step_config(baseline, step_id, patch)


def _with_step_config(baseline: "Strategy", step_id: str, patch: dict) -> "Strategy":
    """Clone a strategy, merging `patch` into exactly one step's config."""
    steps: list[StrategyStep] = []
    for s in baseline.steps:
        if s.step_id == step_id:
            steps.append(StrategyStep(
                step_id=s.step_id,
                order=s.order,
                executor_type=s.executor_type,
                executor_ref=copy.deepcopy(s.executor_ref or {}),
                executor_id=s.executor_id,
                role=s.role,
                config={**copy.deepcopy(s.config or {}), **patch},
                on_failure=s.on_failure,
            ))
        else:
            steps.append(copy.deepcopy(s))
    return Strategy(
        steps=steps,
        surface=baseline.surface,
        surface_binding=copy.deepcopy(baseline.surface_binding or {}),
        dimensions=[],
    )


def context_only_change(baseline: "Strategy", candidate: "Strategy") -> dict:
    """
    Is `candidate` a context-efficiency variant of `baseline`, and nothing else?

    Returns {"ok", "changed", "unexpected"}: `changed` is every dimension that
    differs, `unexpected` is the ones outside CONTEXT_REDUCTION_DIMENSIONS. A
    candidate with `ok=False` is not a context-reduction candidate — it is a
    different experiment, and measuring it under this name would attribute a
    model swap's cost saving to a prompt change.

    `ok=False` also when NOTHING changed: a variant identical to the baseline
    has nothing to measure.
    """
    changed = diff_dimensions(baseline, candidate)
    unexpected = [d for d in changed if d not in domain.CONTEXT_REDUCTION_DIMENSIONS]
    return {
        "ok": bool(changed) and not unexpected,
        "changed": changed,
        "unexpected": unexpected,
        "allowed": list(domain.CONTEXT_REDUCTION_DIMENSIONS),
    }


def _context_config_diff(base_cfg: Any, cand_cfg: Any) -> list[dict]:
    """Differences inside contextConfig, at path grain, excluding maxChars."""
    out: list[dict] = []
    if not isinstance(base_cfg, dict) or not isinstance(cand_cfg, dict):
        if base_cfg != cand_cfg:
            out.append({"path": "contextConfig", "kind": "shape_changed"})
        return out

    base_pack = {k: v for k, v in (base_cfg.get("packaging") or {}).items() if k != "maxChars"}
    cand_pack = {k: v for k, v in (cand_cfg.get("packaging") or {}).items() if k != "maxChars"}
    if base_pack != cand_pack:
        out.append({"path": "contextConfig.packaging", "kind": "changed"})

    for key in set(base_cfg) | set(cand_cfg):
        if key in ("packaging", "sources"):
            continue
        if base_cfg.get(key) != cand_cfg.get(key):
            out.append({"path": f"contextConfig.{key}", "kind": "changed"})

    base_src = base_cfg.get("sources") or []
    cand_src = cand_cfg.get("sources") or []
    if len(base_src) != len(cand_src):
        out.append({
            "path": "contextConfig.sources",
            "kind": "source_count_changed",
            "baseline": len(base_src),
            "candidate": len(cand_src),
        })
        return out
    for i, (b, c) in enumerate(zip(base_src, cand_src)):
        b_rest = {k: v for k, v in (b or {}).items() if k != "maxChars"}
        c_rest = {k: v for k, v in (c or {}).items() if k != "maxChars"}
        if b_rest != c_rest:
            out.append({"path": f"contextConfig.sources[{i}]", "kind": "changed"})
    return out


def graph_diff_outside_context(baseline_graph: dict, candidate_graph: dict) -> list[dict]:
    """
    Everything that differs between two graphs OTHER than prompt text and
    context budgets. An empty list is the proof that a candidate really is a
    context-efficiency variant.

    This is checked on the applied GRAPHS rather than on the strategies, because
    the graph is what executes. A strategy cannot express `tools` or a node's
    declared output schema at all, so a strategy-level comparison would be
    silent about exactly the fields a customer most needs held constant.
    """
    diffs: list[dict] = []

    base_nodes = {str(n.get("id")): n for n in (baseline_graph.get("nodes") or []) if n.get("id")}
    cand_nodes = {str(n.get("id")): n for n in (candidate_graph.get("nodes") or []) if n.get("id")}

    if set(base_nodes) != set(cand_nodes):
        diffs.append({
            "scope": "graph",
            "field": "nodes",
            "kind": "node_set_changed",
            "added": sorted(set(cand_nodes) - set(base_nodes)),
            "removed": sorted(set(base_nodes) - set(cand_nodes)),
        })

    if (baseline_graph.get("edges") or []) != (candidate_graph.get("edges") or []):
        diffs.append({"scope": "graph", "field": "edges", "kind": "changed"})

    for node_id in sorted(set(base_nodes) & set(cand_nodes)):
        b_node, c_node = base_nodes[node_id], cand_nodes[node_id]
        if (b_node.get("type") or "") != (c_node.get("type") or ""):
            diffs.append({
                "scope": "node", "node_id": node_id, "field": "type", "kind": "changed",
            })
        b_data = b_node.get("data") or {}
        c_data = c_node.get("data") or {}
        for key in sorted(set(b_data) | set(c_data)):
            if key == "contextConfig":
                diffs.extend({
                    "scope": "node", "node_id": node_id, "field": d["path"], "kind": d["kind"],
                } for d in _context_config_diff(b_data.get(key), c_data.get(key)))
                continue
            if key in CONTEXT_MUTABLE_NODE_KEYS:
                continue
            if b_data.get(key) != c_data.get(key):
                diffs.append({
                    "scope": "node", "node_id": node_id, "field": key, "kind": "changed",
                })

    return diffs


def unapplicable_dimensions(
    dimensions: list[str], surface: str = SURFACE_RUNTIME
) -> dict[str, str]:
    """
    Of the given dimensions, which cannot be applied ON THIS SURFACE, and why.

    A dimension unapplicable on one surface may be perfectly real on another —
    `temperature` is unapplicable on runtime and applicable on direct
    inference — so the surface is part of the question, not an afterthought.
    """
    applicable = applicable_dimensions(surface)
    unsupported = unsupported_dimensions(surface)
    return {
        d: unsupported.get(d, f"Not applicable on the '{surface}' surface.")
        for d in (dimensions or [])
        if d not in applicable
    }


# ---------------------------------------------------------------------------
# Direct Inference adapter
# ---------------------------------------------------------------------------
#
# A customer who changes one `base_url` and points production traffic at
# POST /v1/chat/completions has NO Studio workflow and NO deployment. That
# traffic must still feed the same Workload -> Strategy -> Attempt -> Cost ->
# Outcome architecture, so a Strategy must be able to describe "the customer's
# own current configuration" with nothing of OptiML's behind it.

def from_direct_inference_request(
    *,
    model: str,
    provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    step_id: str = "direct",
    workload_id: Optional[str] = None,
    extra_config: Optional[dict] = None,
) -> Strategy:
    """
    Build a baseline Strategy describing a direct-inference request.

    This is the customer's OWN configuration, observed rather than authored by
    OptiML: `surface='direct_inference'` and `surface_binding` records that
    there is no deployment. It is a legitimate baseline to benchmark against —
    the absence of an OptiML deployment does not make it less real.

    `temperature`, `max_tokens` and `top_p` are accepted as REAL strategy
    configuration here — unlike on the runtime surface, where they are refused
    because workflow_runtime drops them before the provider call.

    NOTE: `apply_to_graph` will refuse a strategy whose surface is not
    'runtime'. Applying a direct-inference candidate means re-issuing the
    request with different parameters, not rewriting a graph, and that
    application path belongs to the direct-inference surface itself.
    """
    config: dict[str, Any] = {"node_type": "direct_inference"}
    if system_prompt is not None:
        config["system_instructions"] = system_prompt

    # Sampling parameters are GENUINE strategy configuration on this surface:
    # the OpenAI dialect forwards them in the body and the Anthropic translation
    # maps temperature/top_p/max_tokens explicitly, so they reach the provider
    # and a change in them is really benchmarkable. Recording them as strategy
    # config (rather than as opaque request metadata) is what lets a future
    # generator propose a temperature or max_tokens change and have the
    # benchmark mean something.
    for key, value in (
        ("temperature", temperature),
        ("max_tokens", max_tokens),
        ("top_p", top_p),
    ):
        if value is not None:
            config[key] = value

    if extra_config:
        # Reject anything this SURFACE cannot actually vary, for the same reason
        # apply_to_graph does: a candidate that changes nothing would produce a
        # benchmark that measures nothing.
        _assert_supported(extra_config, SURFACE_DIRECT_INFERENCE)
        config.update(extra_config)

    _assert_supported(config, SURFACE_DIRECT_INFERENCE)

    return Strategy(
        steps=[
            StrategyStep(
                step_id=step_id,
                order=0,
                executor_type="model",
                executor_ref={
                    "executor_type": "model",
                    "vendor": (provider or "").strip().lower() or None,
                    "external_id": (model or "").strip() or None,
                },
                role="primary",
                config=config,
                on_failure="fail",
            )
        ],
        surface="direct_inference",
        surface_binding={
            "kind": "direct_inference",
            "workload_id": workload_id,
            # There is no workflow and no deployment. Recorded explicitly so a
            # consumer never goes looking for one.
            "workflow_id": None,
            "deployment_id": None,
            "llm_call_count": 1,
        },
        dimensions=[],
    )
