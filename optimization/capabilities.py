"""
DECLARED capability model for executors, and the ADAPTER layer over it.

Why this module exists
----------------------
The first real production benchmark dispatched 140 replay cases at `o1-mini`
and got a 100% error rate. Nothing was measured; the run produced provider
errors, one case at a time, 140 times. The cause was not exotic: the OpenAI
reasoning family does not accept `max_tokens` (it wants `max_completion_tokens`)
and does not accept `temperature`. Both facts were knowable BEFORE dispatch.
They were simply never asked.

So the fix is not a special case for o1. It is a place where "what a runtime
request needs" and "what a model family accepts" are both written down, and a
generic engine that compares them. Adding the next family — an Anthropic
translation quirk, a Gemini modality limit, a provider that renames `stop` — is
a DATA edit here, not a new branch in the benchmark loop.

Hard rules
----------
1. Everything in this module is a VENDOR CLAIM or a DECLARATION, never a
   measurement. Same boundary `optimization/executors.py` holds: a declared
   capability is a reason to run (or not run) a benchmark. It is never evidence
   about the customer's workload.

2. UNKNOWN IS NOT FALSE. A capability nothing has declared comes back
   `unknown`, and an `unknown` NEVER excludes a candidate. Treating silence as
   "unsupported" would fabricate an incompatibility, which is exactly the
   failure mode — an inflated funnel with invented buckets — this product
   exists to prevent. Only an explicit declaration can block.

3. NO NAME SNIFFING AT THE CALL SITE. There is no `model.startswith("o1")`
   anywhere outside the declaration table below, and inside the table the
   prefix is DATA consumed by one generic matcher. A caller asks
   `resolve_profile(vendor, model)` and reasons about the answer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: A declared capability is one of these. `unknown` is a first-class answer and
#: is the default for anything nobody has written down.
SUPPORT_YES = "supported"
SUPPORT_NO = "not_supported"
SUPPORT_UNKNOWN = "unknown"

#: How a request parameter fares against a family's declaration.
PARAM_SUPPORTED = "supported"
PARAM_RENAMED = "renamed"       # the family wants a different key for the same thing
PARAM_REJECTED = "rejected"     # the family refuses it outright
PARAM_UNKNOWN = "unknown"

#: Adapters. An adapter is a NAMED, declared transformation that can make an
#: otherwise-incompatible request executable. A family references one by name;
#: it never carries a lambda, so a declaration stays inspectable data.
#:
#: `lossless` is the load-bearing field. A rename preserves the customer's
#: intent exactly. DROPPING a parameter does not: a replay whose temperature was
#: silently discarded is no longer a comparison against the configuration the
#: customer actually runs, and reporting it as one would be a measurement of a
#: different workload wearing this workload's name. So a lossy adapter exists
#: and is declarable, but `adapt_request` refuses to apply it by default.
ADAPTER_RENAME = "rename_param"
ADAPTER_DROP = "drop_param"

ADAPTER_LOSSLESS = {
    ADAPTER_RENAME: True,
    ADAPTER_DROP: False,
}


# ---------------------------------------------------------------------------
# The declaration table — DATA, not control flow
# ---------------------------------------------------------------------------
#
# Each entry declares, for a family of models:
#
#   match          a structural rule. Fields are ANDed; a family matches only if
#                  every field it names matches. Adding `id_regex` or
#                  `api_style` here is a one-line change to `_matches`.
#   params         per-request-parameter rules, keyed by the OpenAI-dialect name
#                  OptiML's strategy config uses.
#   capabilities   coarse capability declarations (streaming, tools, ...).
#                  Anything omitted is `unknown` and cannot exclude anything.
#   provenance     where the claim came from, so a wrong entry is traceable.
#
# Entries are deliberately SHORT. A long speculative table would be a table of
# guesses, and a guessed incompatibility is worse than an unchecked one: the
# guess silently removes a candidate that might have won.

FAMILY_DECLARATIONS: list[dict] = [
    {
        "family_id": "openai_reasoning_o_series",
        "display_name": "OpenAI o-series reasoning models",
        "match": {"vendor": "openai", "id_prefixes": ["o1", "o3", "o4"]},
        "params": {
            # THE o1-mini INCIDENT, written down. A lossless rename, so a
            # candidate in this family is executable after adaptation rather
            # than ineligible.
            "max_tokens": {
                "support": PARAM_RENAMED,
                "rename_to": "max_completion_tokens",
                "adapter": ADAPTER_RENAME,
                "detail": "The o-series accepts max_completion_tokens only.",
            },
            # Rejected with NO adapter. Dropping it would change the sampling
            # regime the customer runs, so the honest outcome is ineligible —
            # not a quietly different experiment.
            "temperature": {
                "support": PARAM_REJECTED,
                "adapter": None,
                "detail": "Sampling temperature is fixed; the parameter is refused.",
            },
            "top_p": {
                "support": PARAM_REJECTED,
                "adapter": None,
                "detail": "Nucleus sampling is fixed; the parameter is refused.",
            },
        },
        "capabilities": {
            # Billed reasoning tokens are recorded as a FACT, never as a screen.
            # See optimization/eligibility.py: screening on this would have
            # excluded gpt-5-mini, the only verified win this product has.
            "emits_billed_reasoning_tokens": SUPPORT_YES,
            "system_message": SUPPORT_NO,
            "developer_message": SUPPORT_YES,
        },
        "provenance": "vendor_documentation",
        "last_verified": "2026-02-22",
    },
    {
        "family_id": "openai_gpt5",
        "display_name": "OpenAI GPT-5 family",
        "match": {"vendor": "openai", "id_prefixes": ["gpt-5"]},
        "params": {
            "max_tokens": {
                "support": PARAM_RENAMED,
                "rename_to": "max_completion_tokens",
                "adapter": ADAPTER_RENAME,
                "detail": "The GPT-5 family accepts max_completion_tokens only.",
            },
        },
        "capabilities": {
            "emits_billed_reasoning_tokens": SUPPORT_YES,
            "system_message": SUPPORT_YES,
        },
        "provenance": "vendor_documentation",
        "last_verified": "2026-02-22",
    },
    {
        "family_id": "anthropic_messages",
        "display_name": "Anthropic Messages API",
        "match": {"vendor": "anthropic"},
        "params": {
            # Mirrors direct_inference._ANTHROPIC_UNSUPPORTED, which already
            # refuses these at request time with a 400 rather than dropping
            # them. Declaring them here moves the same knowledge EARLIER, so a
            # benchmark never dispatches a request the translation layer will
            # refuse.
            **{
                name: {
                    "support": PARAM_REJECTED,
                    "adapter": None,
                    "detail": (
                        "The Anthropic translation cannot express this field and "
                        "OptiML refuses to silently ignore it."
                    ),
                }
                for name in (
                    "response_format", "logit_bias", "logprobs", "top_logprobs",
                    "presence_penalty", "frequency_penalty", "seed",
                    "parallel_tool_calls",
                )
            },
        },
        "capabilities": {
            "system_message": SUPPORT_YES,
            "streaming": SUPPORT_YES,
        },
        "provenance": "optiml_translation_layer:direct_inference.py",
        "last_verified": "2026-02-22",
    },
]


# ---------------------------------------------------------------------------
# The generic matcher
# ---------------------------------------------------------------------------

def _matches(rule: dict, vendor: str, model_id: str) -> Optional[int]:
    """
    Does this declaration's `match` rule apply? Returns its SPECIFICITY (the
    number of constraints it satisfied) or None.

    Specificity is what lets a narrow family override a broad one without the
    table needing an explicit priority column: `openai_reasoning_o_series`
    (vendor + prefix, specificity 2) beats a hypothetical vendor-wide OpenAI
    entry (specificity 1) on the parameters both declare.
    """
    v = (vendor or "").strip().lower()
    m = (model_id or "").strip().lower()
    score = 0

    want_vendor = rule.get("vendor")
    if want_vendor is not None:
        if v != str(want_vendor).strip().lower():
            return None
        score += 1

    prefixes = rule.get("id_prefixes")
    if prefixes:
        if not any(m.startswith(str(p).lower()) for p in prefixes):
            return None
        score += 1

    contains = rule.get("id_contains")
    if contains:
        if not any(str(c).lower() in m for c in contains):
            return None
        score += 1

    exact = rule.get("id_exact")
    if exact:
        if m not in {str(e).lower() for e in exact}:
            return None
        score += 1

    return score


@dataclass
class ModelProfile:
    """
    Everything DECLARED about one model. No measurement, ever.

    `families` names every declaration that matched, most specific last, so a
    surprising verdict can be traced back to the row that produced it.
    """

    vendor: str
    model_id: str
    params: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    #: Vendor-published facts merged in from shared/providers.json.
    context_window: Optional[int] = None
    api_style: Optional[str] = None
    known_to_catalog: bool = False

    def capability(self, name: str) -> str:
        """Declared support for a capability. `unknown` when nobody said."""
        return self.capabilities.get(name, SUPPORT_UNKNOWN)

    def param_rule(self, name: str) -> dict:
        """
        The declared rule for one request parameter.

        Absent means UNKNOWN, not "supported" and not "rejected". An unknown
        parameter is passed through untouched and recorded as unknown — the
        provider remains the authority on it.
        """
        return self.params.get(name) or {"support": PARAM_UNKNOWN}

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "model_id": self.model_id,
            "families": list(self.families),
            "capabilities": dict(self.capabilities),
            "declared_params": sorted(self.params),
            "context_window": self.context_window,
            "api_style": self.api_style,
            "known_to_catalog": self.known_to_catalog,
            "provenance": list(self.provenance),
            "data_class": "vendor_declaration",
        }


def _catalog_index() -> dict:
    """
    (vendor, model_id) -> vendor-published facts, from shared/providers.json.

    Built per call rather than cached: the catalog is small, and a stale cache
    here would mean a model that was added to the price sheet still reporting
    `model_not_available`.
    """
    from optimization import executors

    out: dict[tuple[str, str], dict] = {}
    for entry in executors.vendor_catalog():
        key = (
            str(entry.get("vendor") or "").strip().lower(),
            str(entry.get("external_id") or "").strip().lower(),
        )
        out[key] = entry
    return out


def resolve_profile(
    vendor: str,
    model_id: str,
    *,
    catalog_index: Optional[dict] = None,
) -> ModelProfile:
    """
    Merge every matching declaration with the vendor catalog into one profile.

    Merge order is specificity-ascending, so the narrowest declaration wins on
    any key two families both declare.
    """
    matched: list[tuple[int, dict]] = []
    for decl in FAMILY_DECLARATIONS:
        score = _matches(decl.get("match") or {}, vendor, model_id)
        if score is not None:
            matched.append((score, decl))
    matched.sort(key=lambda t: t[0])

    params: dict = {}
    capabilities: dict = {}
    families: list[str] = []
    provenance: list[str] = []
    for _score, decl in matched:
        params.update(decl.get("params") or {})
        capabilities.update(decl.get("capabilities") or {})
        families.append(decl["family_id"])
        prov = decl.get("provenance")
        if prov and prov not in provenance:
            provenance.append(prov)

    index = catalog_index if catalog_index is not None else _catalog_index()
    entry = index.get(
        (str(vendor or "").strip().lower(), str(model_id or "").strip().lower())
    )
    caps = (entry or {}).get("capabilities") or {}

    return ModelProfile(
        vendor=vendor,
        model_id=model_id,
        params=params,
        capabilities=capabilities,
        families=families,
        provenance=provenance or ["undeclared"],
        context_window=caps.get("context_window"),
        api_style=caps.get("api_style"),
        known_to_catalog=entry is not None,
    )


# ---------------------------------------------------------------------------
# The adapter engine
# ---------------------------------------------------------------------------

def _adapter_rename(params: dict, name: str, rule: dict) -> dict:
    target = rule.get("rename_to")
    if not target:
        raise ValueError(f"rename adapter for '{name}' declares no rename_to")
    out = dict(params)
    out[target] = out.pop(name)
    return out


def _adapter_drop(params: dict, name: str, rule: dict) -> dict:
    out = dict(params)
    out.pop(name, None)
    return out


ADAPTERS: dict[str, Callable[[dict, str, dict], dict]] = {
    ADAPTER_RENAME: _adapter_rename,
    ADAPTER_DROP: _adapter_drop,
}


@dataclass
class AdaptationResult:
    """
    Can this family execute this request, and at what cost to fidelity?

    `executable` False means the candidate is INELIGIBLE — there is a parameter
    the family refuses and no declared, lossless way around it. That is the
    o1-mini + temperature case, and it is a legitimate answer: the alternative
    is 140 provider errors.
    """

    executable: bool
    adapted_params: dict
    adaptations: list[dict] = field(default_factory=list)
    blockers: list[dict] = field(default_factory=list)
    unknown_params: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "executable": self.executable,
            "adaptations": list(self.adaptations),
            "blockers": list(self.blockers),
            "unknown_params": list(self.unknown_params),
        }


def adapt_request(
    profile: ModelProfile,
    request_params: dict,
    *,
    allow_lossy: bool = False,
) -> AdaptationResult:
    """
    Compare a REQUEST SHAPE against a family's declared parameter rules.

    `request_params` is only what the request ACTUALLY SETS. This matters more
    than it looks: on the runtime surface `workflow_runtime` never forwards
    `temperature` at all, so an o-series candidate there has nothing to reject
    and is executable after the `max_tokens` rename. The same model on the
    direct-inference surface, where the customer really does send
    `temperature`, is ineligible. Eligibility is a property of the (model,
    request) PAIR, not of the model.

    `allow_lossy` is off by default and no caller in the benchmark path turns it
    on. A lossy adaptation silently changes what is being measured.
    """
    adapted = dict(request_params or {})
    adaptations: list[dict] = []
    blockers: list[dict] = []
    unknown: list[str] = []

    for name in sorted(request_params or {}):
        if (request_params or {}).get(name) is None:
            continue  # not actually set
        rule = profile.param_rule(name)
        support = rule.get("support", PARAM_UNKNOWN)

        if support == PARAM_SUPPORTED:
            continue

        if support == PARAM_UNKNOWN:
            # Nobody declared anything. Pass it through and say so. Refusing
            # here would invent an incompatibility.
            unknown.append(name)
            continue

        adapter_name = rule.get("adapter")
        adapter = ADAPTERS.get(adapter_name) if adapter_name else None
        lossless = ADAPTER_LOSSLESS.get(adapter_name, False) if adapter_name else False

        if adapter is None or (not lossless and not allow_lossy):
            blockers.append({
                "param": name,
                "support": support,
                "declared_adapter": adapter_name,
                "lossless": lossless if adapter_name else None,
                "detail": rule.get("detail"),
                "families": list(profile.families),
            })
            continue

        adapted = adapter(adapted, name, rule)
        adaptations.append({
            "param": name,
            "adapter": adapter_name,
            "renamed_to": rule.get("rename_to"),
            "lossless": lossless,
            "detail": rule.get("detail"),
        })

    return AdaptationResult(
        executable=not blockers,
        adapted_params=adapted,
        adaptations=adaptations,
        blockers=blockers,
        unknown_params=unknown,
    )


def request_params_of_step(step: Any) -> dict:
    """
    The request parameters a strategy step ACTUALLY sets.

    Reads the same config keys `optimization.strategy` writes for the
    direct-inference surface. Structural keys (`node_type`,
    `system_instructions`, ...) are not request parameters and are excluded, so
    they can never be mistaken for one by a family rule.
    """
    config = dict(getattr(step, "config", None) or {})
    return {
        name: config[name]
        for name in _REQUEST_PARAM_KEYS
        if config.get(name) is not None
    }


#: Config keys that become provider request parameters. Kept explicit rather
#: than "everything not structural", so adding a strategy config key cannot
#: accidentally start being checked as a provider parameter.
_REQUEST_PARAM_KEYS = (
    "temperature",
    "max_tokens",
    "top_p",
    "response_format",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "parallel_tool_calls",
    "stop",
    "tools",
    "tool_choice",
    "stream",
)
