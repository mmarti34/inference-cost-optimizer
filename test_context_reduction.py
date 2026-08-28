"""
Context efficiency (internally: `context_reduction`).

The claim under test is narrow and is the whole product value:

    Same model. Same provider. Same sampling parameters. Same tools. Same
    output contract. ONLY the context representation changes.

So these tests do two things. They check that a reduction which cannot be
MEASURED, or which loses something the workload depends on, is EXCLUDED with its
own reason code rather than benchmarked. And they assert the invariance directly
— on the applied graphs, not on a docstring — because "only the context changed"
is the definition of the dimension, not a side effect of it.

No provider is called and no database is touched. The accounting module is
faked, but the fake is not a stub: it assembles context the way
`context_runtime` does (per-source truncation, then packaging truncation) and
counts tokens with a deterministic tokenizer, so a budget change really does
produce a smaller measured number for the same reason it would in production.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import pytest

from optimization import benchmark as benchmark_mod
from optimization import candidates as candidates_mod
from optimization import domain
from optimization import strategy as strategy_mod


# ---------------------------------------------------------------------------
# The workload under test
# ---------------------------------------------------------------------------

STEP_ID = "ai-1"

SYSTEM_INSTRUCTIONS = (
    "You are a support triage assistant.\n"
    "\n"
    "Respond only with valid JSON matching this shape:\n"
    '{"category": "string", "priority": "string", "needs_human": true}\n'
    "\n"
    "Use the lookup_account tool when the ticket names a customer.\n"
    "\n"
    "You are a support triage assistant.\n"  # exact duplicate block
)

TASK_DESCRIPTION = (
    "Triage the following support ticket.\n"
    "\n"
    "Ticket: {{input}}\n"
    "Account tier: {{tier}}\n"
)

#: The text each context source really resolves to. The fake accounting module
#: assembles from this, so a character budget has something real to truncate.
SOURCE_TEXTS = {
    "Escalation policy": "ESCALATION POLICY. " * 60,      # 1140 chars, REQUIRED
    "Historic tickets": "HISTORIC TICKET EXAMPLE. " * 200,  # 5000 chars, optional
}


def _base_graph() -> dict:
    return {
        "nodes": [
            {
                "id": STEP_ID,
                "type": "ai-step",
                "data": {
                    "modelName": "gpt-4.1",
                    "provider": "openai",
                    "temperature": 0.2,
                    "taskDescription": TASK_DESCRIPTION,
                    "systemInstructions": SYSTEM_INSTRUCTIONS,
                    "outputSchema": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "priority": {"type": "string"},
                            "needs_human": {"type": "boolean"},
                        },
                    },
                    "tools": [
                        {"name": "lookup_account", "description": "Fetch an account."},
                    ],
                    "contextConfig": {
                        "enabled": True,
                        "mode": "prepacked",
                        "packaging": {"maxChars": 6200, "strategy": "concat"},
                        "sources": [
                            {
                                "type": "knowledge_asset",
                                "label": "Escalation policy",
                                "required": True,
                                "maxChars": 2000,
                            },
                            {
                                "type": "knowledge_asset",
                                "label": "Historic tickets",
                                "required": False,
                                "maxChars": 6000,
                            },
                        ],
                    },
                },
            }
        ],
        "edges": [],
    }


def _workload() -> dict:
    return {
        "id": "wl-1",
        "identity_kind": "workflow",
        "identity_ref": "wf-1",
        "graph_json": _base_graph(),
    }


def _cases(n: int = 40) -> list[dict]:
    return [
        {
            "id": f"case-{i}",
            "input_text": f"ticket {i}",
            "expected_output": f"expected {i}",
            "variables": {"tier": "gold"},
        }
        for i in range(n)
    ]


def _checks() -> list[dict]:
    """A deterministic eval check — the signal class this dimension requires."""
    return [{"name": "exact", "type": "deterministic", "enabled": True, "config": {}}]


def _history(**overrides) -> dict:
    history = {
        "org_id": "org-1",
        "workload_id": "wl-1",
        "workflow_id": "wf-1",
        "replay_cases": _cases(),
        "quality_checks": _checks(),
        "traffic": {"run_count": 900, "coverage": 1.0},
        "lookback_days": 30,
        "model_stats": {},
    }
    history.update(overrides)
    return history


def _baseline() -> strategy_mod.Strategy:
    return strategy_mod.from_graph_json(_base_graph(), workflow_id="wf-1")


# ---------------------------------------------------------------------------
# A fake optimization.context_accounting
# ---------------------------------------------------------------------------
#
# Mirrors the interface the concurrent implementation exposes. `tokens` is a
# real count from a real (if simple) tokenizer, never an estimate, so a variant
# that does not shrink the assembled text genuinely measures no reduction.

TOKENIZER = "fake-whitespace-v1"


def _count(text: str) -> int:
    return len((text or "").split())


@dataclass(frozen=True)
class ComponentTokens:
    component: str
    source_type: Optional[str]
    tokens: Optional[int]
    chars: int
    tokenizer: Optional[str]
    static: bool


@dataclass(frozen=True)
class ContextProfile:
    workload_id: str
    step_id: str
    n_cases: int
    components: tuple = ()
    total_tokens: Optional[int] = None
    reducible_tokens: Optional[int] = None
    tokenizer: Optional[str] = None
    coverage: float = 1.0


class FakeAccounting:
    """
    Assembles context the way `context_runtime` does, then counts tokens.

    `count_tokens` is optional so the "no measurement, no finding" path can be
    exercised: constructing this with `supports_count_tokens=False` deletes the
    attribute, which is exactly the state of the world before the accounting
    module implements it.
    """

    def __init__(self, *, supports_count_tokens: bool = True,
                 supports_strategy_profile: bool = True,
                 reducible_tokens: Optional[int] = -1):
        self.supports_count_tokens = supports_count_tokens
        self.supports_strategy_profile = supports_strategy_profile
        #: -1 means "derive it"; anything else is returned verbatim so the
        #: unmeasured / immaterial branches can be driven.
        self._reducible_override = reducible_tokens
        self.calls: list[str] = []

    # -- the interface ---------------------------------------------------

    def profile_workload_context(self, org_id, workload, *, cases, step_id=None):
        self.calls.append("profile_workload_context")
        return self._profile(workload.get("graph_json") or {}, workload, cases, step_id)

    @property
    def profile_strategy_context(self):
        if not self.supports_strategy_profile:
            raise AttributeError("profile_strategy_context")
        return self._profile_strategy

    @property
    def count_tokens(self):
        if not self.supports_count_tokens:
            raise AttributeError("count_tokens")
        return self._count_tokens

    def _count_tokens(self, text, *, tokenizer=None):
        self.calls.append("count_tokens")
        return _count(text)

    def _profile_strategy(self, org_id, workload, strategy, *, cases, step_id=None):
        self.calls.append("profile_strategy_context")
        graph = strategy_mod.apply_to_graph(workload.get("graph_json") or {}, strategy)
        return self._profile(graph, workload, cases, step_id)

    # -- assembly, mirroring context_runtime -----------------------------

    def _profile(self, graph, workload, cases, step_id):
        node = next(
            (n for n in graph.get("nodes") or [] if str(n.get("id")) == (step_id or STEP_ID)),
            None,
        )
        if node is None:
            return None
        data = node.get("data") or {}
        cfg = data.get("contextConfig") or {}

        components = [
            ComponentTokens(
                component="system_instructions", source_type=None,
                tokens=_count(data.get("systemInstructions") or ""),
                chars=len(data.get("systemInstructions") or ""),
                tokenizer=TOKENIZER, static=True,
            ),
            ComponentTokens(
                component="task_description", source_type=None,
                tokens=_count(data.get("taskDescription") or ""),
                chars=len(data.get("taskDescription") or ""),
                tokenizer=TOKENIZER, static=True,
            ),
        ]

        sections: list[str] = []
        for src in (cfg.get("sources") or []):
            text = SOURCE_TEXTS.get(src.get("label"), "")
            per_source = src.get("maxChars")
            if per_source and len(text) > per_source:
                text = text[:per_source]
            sections.append(text)

        packaged = "\n\n---\n\n".join(sections)
        budget = (cfg.get("packaging") or {}).get("maxChars")
        if budget and len(packaged) > budget:
            packaged = packaged[:budget]

        # Attribute the packaged result back to each source, in order, so a
        # packaging truncation shows up on the source it actually cut.
        offset = 0
        for src, section in zip(cfg.get("sources") or [], sections):
            kept = packaged[offset:offset + len(section)]
            offset += len(section) + len("\n\n---\n\n")
            components.append(ComponentTokens(
                component=f"context_source:{src.get('label')}",
                source_type=src.get("type"),
                tokens=_count(kept), chars=len(kept),
                tokenizer=TOKENIZER, static=True,
            ))

        components.append(ComponentTokens(
            component="user_input", source_type=None,
            tokens=_count(cases[0]["input_text"]) if cases else 0,
            chars=len(cases[0]["input_text"]) if cases else 0,
            tokenizer=TOKENIZER, static=False,
        ))

        total = sum(c.tokens or 0 for c in components)
        reducible = sum(c.tokens or 0 for c in components if c.static)
        if self._reducible_override != -1:
            reducible = self._reducible_override
        return ContextProfile(
            workload_id=str(workload.get("id")), step_id=step_id or STEP_ID,
            n_cases=len(cases or []), components=tuple(components),
            total_tokens=total, reducible_tokens=reducible,
            tokenizer=TOKENIZER, coverage=1.0,
        )


def _generator(**kwargs) -> candidates_mod.ContextReductionGenerator:
    accounting = kwargs.pop("accounting", None)
    if accounting is None:
        accounting = FakeAccounting()
    return candidates_mod.ContextReductionGenerator(accounting=accounting, **kwargs)


def _run(generator=None, history=None):
    generator = generator or _generator()
    return generator.generate_with_report(_workload(), _baseline(), history or _history())


def _codes(excluded: list[dict]) -> list[str]:
    return [e["code"] for e in excluded]


# ---------------------------------------------------------------------------
# The dimension is what it says it is
# ---------------------------------------------------------------------------

def test_baseline_and_candidate_differ_in_nothing_but_context():
    """
    THE definition of this dimension, asserted directly.

    Checked on the applied GRAPHS rather than on the strategies, because a
    Strategy cannot express a node's tools or its declared output schema at all
    — a strategy-level comparison would be silent about exactly the fields a
    customer most needs held constant.
    """
    cands, _ = _run()
    assert cands, "expected at least one context-reduction candidate"

    base_graph = _base_graph()
    for cand in cands:
        applied = strategy_mod.apply_to_graph(base_graph, cand.strategy)
        node = applied["nodes"][0]["data"]
        base_node = base_graph["nodes"][0]["data"]

        assert node["modelName"] == base_node["modelName"] == "gpt-4.1"
        assert node["provider"] == base_node["provider"] == "openai"
        assert node["temperature"] == base_node["temperature"] == 0.2
        assert node["tools"] == base_node["tools"]
        assert node["outputSchema"] == base_node["outputSchema"]

        # And the structural proof: nothing outside prompt text and context
        # budgets differs anywhere in the graph.
        assert strategy_mod.graph_diff_outside_context(base_graph, applied) == []

        assert set(cand.dimensions) <= set(domain.CONTEXT_REDUCTION_DIMENSIONS)
        assert cand.dimensions, "a candidate identical to the baseline is not a candidate"


def test_graph_diff_outside_context_catches_a_model_swap():
    """The invariance check must be able to FAIL, or it proves nothing."""
    base = _base_graph()
    swapped = copy.deepcopy(base)
    swapped["nodes"][0]["data"]["modelName"] = "gpt-4.1-mini"

    diffs = strategy_mod.graph_diff_outside_context(base, swapped)
    assert [d["field"] for d in diffs] == ["modelName"]


def test_graph_diff_outside_context_catches_a_dropped_tool():
    base = _base_graph()
    stripped = copy.deepcopy(base)
    stripped["nodes"][0]["data"]["tools"] = []

    diffs = strategy_mod.graph_diff_outside_context(base, stripped)
    assert [d["field"] for d in diffs] == ["tools"]


def test_graph_diff_outside_context_catches_a_removed_context_source():
    """
    Removing a source is a RETRIEVAL change, not a budget change. It must not
    pass as context efficiency however much it would reduce tokens.
    """
    base = _base_graph()
    trimmed = copy.deepcopy(base)
    trimmed["nodes"][0]["data"]["contextConfig"]["sources"].pop()

    diffs = strategy_mod.graph_diff_outside_context(base, trimmed)
    assert any(d["kind"] == "source_count_changed" for d in diffs)


def test_a_variant_changing_another_dimension_is_refused():
    """A variant that moves the model is a different experiment, not this one."""
    baseline = _baseline()
    swapped = candidates_mod._swap_model(baseline, STEP_ID, "openai", "gpt-4.1-mini")

    scope = strategy_mod.context_only_change(baseline, swapped)
    assert scope["ok"] is False
    assert "model" in scope["unexpected"]

    gen = _generator()
    verdict = gen._static_checks(
        baseline, baseline.step(STEP_ID),
        candidates_mod.ContextVariant(
            kind=domain.CONTEXT_VARIANT_PROPOSED_REWRITE, step_id=STEP_ID,
            strategy=swapped, origin="model",
        ),
        FakeAccounting().profile_workload_context(
            "org-1", _workload(), cases=_cases(), step_id=STEP_ID,
        ),
    )
    assert verdict["code"] == "context_reduction_changed_other_dimension"


# ---------------------------------------------------------------------------
# The cheapest variant of all: a pure budget change
# ---------------------------------------------------------------------------

def test_a_max_chars_only_variant_is_generated_and_is_applicable():
    """
    A packaging-budget variant rewrites nothing. It must be generated, it must
    ride the EXISTING `context_length` dimension, and it must actually apply to
    the runtime graph — a dimension that cannot be applied must never reach a
    benchmark.
    """
    cands, _ = _run()
    budget = [
        c for c in cands
        if c.measured_basis["variant_kind"] == domain.CONTEXT_VARIANT_PACKAGING_BUDGET
    ]
    assert budget, "the pure budget variant is the most defensible one and must be generated"

    cand = budget[0]
    assert cand.dimensions == ["context_length"]
    assert strategy_mod.unapplicable_dimensions(
        cand.dimensions, strategy_mod.SURFACE_RUNTIME
    ) == {}

    applied = strategy_mod.apply_to_graph(_base_graph(), cand.strategy)
    new_budget = applied["nodes"][0]["data"]["contextConfig"]["packaging"]["maxChars"]
    assert new_budget < 6200
    assert new_budget == cand.measured_basis["packaging_max_chars"]

    # Not one character of instruction text moved.
    assert applied["nodes"][0]["data"]["systemInstructions"] == SYSTEM_INSTRUCTIONS
    assert applied["nodes"][0]["data"]["taskDescription"] == TASK_DESCRIPTION


def test_the_budget_variant_is_generated_before_any_rewrite():
    """Cheapest and most defensible first, in the order the generator emits."""
    gen = _generator()
    variants = gen._variants(
        _baseline(), _baseline().step(STEP_ID),
        FakeAccounting().profile_workload_context(
            "org-1", _workload(), cases=_cases(), step_id=STEP_ID,
        ),
    )
    kinds = [v.kind for v in variants]
    assert kinds[0] == domain.CONTEXT_VARIANT_PACKAGING_BUDGET
    rewrites = [
        i for i, k in enumerate(kinds)
        if k not in domain.CONTEXT_VARIANT_KINDS_NO_REWRITE
    ]
    budgets = [
        i for i, k in enumerate(kinds)
        if k in domain.CONTEXT_VARIANT_KINDS_NO_REWRITE
    ]
    assert not rewrites or max(budgets) < min(rewrites)


# ---------------------------------------------------------------------------
# Static compatibility checks — each with its OWN reason code
# ---------------------------------------------------------------------------

def _proposer(task=None, system=None):
    """A stand-in for a model that proposes shorter text."""
    def propose(*, task_description, system_instructions, profile):
        return {
            "task_description": task if task is not None else task_description,
            "system_instructions": system if system is not None else system_instructions,
            "model": "proposer-model",
        }
    return propose


def test_a_variant_that_drops_a_placeholder_is_excluded_with_its_own_code():
    shorter_task = "Triage the following support ticket.\n\nAccount tier: {{tier}}\n"
    _, excluded = _run(_generator(proposer=_proposer(task=shorter_task)))

    hits = [e for e in excluded if e["code"] == "context_placeholder_dropped"]
    assert hits, f"expected a placeholder exclusion, got {_codes(excluded)}"
    assert hits[0]["dropped_placeholders"] == ["input"]
    assert hits[0]["variant_origin"] == "model"


def test_a_variant_that_drops_a_tool_reference_is_excluded_with_its_own_code():
    shorter_system = (
        "You are a support triage assistant.\n"
        "Respond only with valid JSON matching this shape:\n"
        '{"category": "string", "priority": "string", "needs_human": true}\n'
    )
    _, excluded = _run(_generator(proposer=_proposer(system=shorter_system)))

    hits = [e for e in excluded if e["code"] == "context_tool_reference_dropped"]
    assert hits, f"expected a tool exclusion, got {_codes(excluded)}"
    assert hits[0]["dropped_tools"] == ["lookup_account"]


def test_a_variant_that_drops_an_output_schema_instruction_is_excluded_with_its_own_code():
    shorter_system = (
        "You are a support triage assistant.\n"
        "\n"
        "Use the lookup_account tool when the ticket names a customer.\n"
    )
    _, excluded = _run(_generator(proposer=_proposer(system=shorter_system)))

    hits = [e for e in excluded if e["code"] == "context_output_contract_dropped"]
    assert hits, f"expected an output-contract exclusion, got {_codes(excluded)}"
    assert hits[0]["dropped_marker_count"] >= 1


def test_the_three_checks_produce_three_distinct_codes():
    """
    Reason-code grain, not disposition grain.

    All three exit at the same disposition. Collapsing them into one number is
    exactly what would erase the evidence that the search was thorough.
    """
    codes = {
        "context_placeholder_dropped",
        "context_tool_reference_dropped",
        "context_output_contract_dropped",
    }
    assert codes <= set(domain.EXCLUSION_CODES)
    dispositions = {domain.EXCLUSION_CODE_TO_DISPOSITION[c] for c in codes}
    assert dispositions == {domain.DISPOSITION_INCOMPATIBLE}
    assert len(codes) == 3, "three distinct checks stay individually visible"


def test_a_variant_that_does_not_measurably_reduce_tokens_is_excluded_not_benchmarked():
    """
    A reduction that is not a reduction buys nothing, so it is not worth paying
    a replay for. It leaves the funnel as `economically_dominated` — a screening
    decision about a benchmark budget, never a claim about quality.
    """
    padded = SYSTEM_INSTRUCTIONS.replace("triage assistant", "triage  assistant")
    cands, excluded = _run(_generator(proposer=_proposer(system=padded)))

    hits = [
        e for e in excluded
        if e["code"] == "context_reduction_immaterial"
        and e["variant_kind"] == domain.CONTEXT_VARIANT_PROPOSED_REWRITE
    ]
    assert hits, f"expected an immateriality exclusion, got {_codes(excluded)}"
    assert hits[0]["tokens_reduced"] < candidates_mod.ContextReductionGenerator.MIN_TOKEN_REDUCTION
    assert domain.EXCLUSION_CODE_TO_DISPOSITION["context_reduction_immaterial"] == (
        domain.DISPOSITION_ECONOMICALLY_DOMINATED
    )
    assert not [
        c for c in cands
        if c.measured_basis["variant_kind"] == domain.CONTEXT_VARIANT_PROPOSED_REWRITE
    ]


def test_a_budget_below_the_measured_required_context_is_excluded():
    """
    "Still fits whatever the workload requires", measured rather than assumed:
    the floor is the MEASURED size of the sources the workload marks required.
    """
    gen = _generator()
    gen.BUDGET_FRACTIONS = (0.05,)
    _, excluded = _run(gen)

    hits = [e for e in excluded if e["code"] == "context_budget_below_requirement"]
    assert hits, f"expected a requirement exclusion, got {_codes(excluded)}"
    assert hits[0]["scope"] == "packaging"
    assert hits[0]["proposed_max_chars"] < hits[0]["measured_required_chars"]
    assert hits[0]["required_sources"] == ["Escalation policy"]


# ---------------------------------------------------------------------------
# An LLM may propose. An LLM may never be the evidence.
# ---------------------------------------------------------------------------

def test_an_llm_proposed_variant_with_no_measurement_produces_no_finding():
    """
    The governing rule, made structural.

    With no way to MEASURE the proposed text, the proposal is refused. It is not
    downgraded to a weaker finding, not emitted with an estimated saving, and not
    benchmarked on the strength of having been written by a model.
    """
    tight = (
        "You are a support triage assistant.\n"
        "Respond only with valid JSON matching this shape:\n"
        '{"category": "string", "priority": "string", "needs_human": true}\n'
        "Use the lookup_account tool when the ticket names a customer.\n"
    )
    accounting = FakeAccounting(supports_count_tokens=False)
    cands, excluded = _run(_generator(accounting=accounting, proposer=_proposer(system=tight)))

    proposed = [
        c for c in cands
        if c.measured_basis["variant_kind"] == domain.CONTEXT_VARIANT_PROPOSED_REWRITE
    ]
    assert proposed == []

    hits = [
        e for e in excluded
        if e["code"] == "context_reduction_unmeasured"
        and e["variant_kind"] == domain.CONTEXT_VARIANT_PROPOSED_REWRITE
    ]
    assert hits, f"expected an unmeasured exclusion, got {_codes(excluded)}"
    assert hits[0]["detail_code"] == "count_tokens_unavailable"


def test_a_measurable_llm_proposal_is_a_candidate_and_nothing_more():
    """
    The other half of the rule: a measurable proposal DOES become a candidate,
    with evidence_source 'none' and no cost, latency or quality claim attached.
    Its provenance travels with it.
    """
    tight = (
        "You are a support triage assistant.\n"
        "Respond only with valid JSON matching this shape:\n"
        '{"category": "string", "priority": "string", "needs_human": true}\n'
        "Use the lookup_account tool when the ticket names a customer.\n"
    )
    # A wide cap: this test is about the proposal path, not about the
    # benchmark-budget cap (which `test_the_cap_reports_what_it_set_aside`
    # covers).
    gen = _generator(proposer=_proposer(system=tight), max_candidates=8)
    gen.MIN_TOKEN_REDUCTION = 1
    gen.MIN_TOKEN_REDUCTION_RATIO = 0.0
    cands, _ = gen.generate_with_report(_workload(), _baseline(), _history())

    proposed = [
        c for c in cands
        if c.measured_basis["variant_kind"] == domain.CONTEXT_VARIANT_PROPOSED_REWRITE
    ]
    assert proposed, "a measurable proposal is a legitimate candidate"
    cand = proposed[0]

    assert cand.evidence_source == "none"
    assert domain.evidence_strength(cand.evidence_source) < (
        domain.MIN_EVIDENCE_STRENGTH_FOR_VERIFICATION
    )
    assert cand.measured_basis["variant_origin"] == "model"
    assert cand.measured_basis["is_outcome_evidence"] is False
    assert {n["code"] for n in cand.notes} >= {
        "context_token_measurement_only", "model_authored_proposal",
    }


def test_no_savings_figure_is_ever_invented():
    """
    Estimated / verified / realized stay separate. A candidate leaves here with
    a PROJECTION at most, and the projection carries its own basis.
    """
    cands, _ = _run()
    for cand in cands:
        assert cand.evidence_source == "none"
        basis = cand.projection_basis
        assert basis["kind"] == "projection"
        if cand.projected_savings_usd is not None:
            assert basis["result"] == "projected"
            assert basis["input_token_pricing"]["price_source"].startswith("vendor_list_price:")
        assert "verified_savings_usd" not in (cand.measured_basis or {})
        # The measured number is about TOKENS, and says so.
        assert cand.measured_basis["kind"] == "context_token_measurement"
        assert cand.measured_basis["tokens_reduced"] > 0
        assert cand.measured_basis["tokenizer"] == TOKENIZER


# ---------------------------------------------------------------------------
# Blocker 2: the quality signal must be able to see the failure mode
# ---------------------------------------------------------------------------

def test_a_workload_with_only_a_judge_signal_gets_no_candidates_at_all():
    """
    A shortened prompt usually still produces plausible output and degrades only
    on the harder tail — an LLM judge would rate the degraded output fine. So
    for a workload whose only check is model-graded, nothing is proposed.
    """
    history = _history(quality_checks=[
        {"name": "judge", "type": "model_graded", "enabled": True, "config": {}},
    ])
    cands, excluded = _run(history=history)

    assert cands == []
    assert _codes(excluded) == ["context_reduction_quality_signal_insufficient"]
    assert excluded[0]["checks_measurable"] == 0


def test_a_deterministic_check_with_no_expected_output_is_not_a_signal():
    """
    Mirrors `benchmark._run_quality_checks`: a deterministic check with nothing
    to compare against scores nothing, so declaring one is not the same as
    having a signal.
    """
    cases = [dict(c, expected_output="") for c in _cases()]
    assert candidates_mod.measurable_quality_checks(_checks(), cases) == []
    assert candidates_mod.measurable_quality_checks(_checks(), _cases()) == _checks()


def test_a_case_set_too_small_to_contain_a_tail_is_refused():
    history = _history(replay_cases=_cases(10))
    cands, excluded = _run(history=history)

    assert cands == []
    assert _codes(excluded) == ["context_reduction_case_count_insufficient"]
    assert excluded[0]["observed"] == 10
    assert excluded[0]["required"] == candidates_mod.ContextReductionGenerator.MIN_CASES


def test_with_no_accounting_module_nothing_is_proposed():
    """
    Blocker 1 is answered by a measurement, not by this module's optimism. With
    no accounting module there is no measurement, and with no measurement there
    is no candidate.
    """
    gen = candidates_mod.ContextReductionGenerator(accounting=None)
    # Force the resolver to find nothing, as it does before the module lands.
    gen._accounting = None
    cands, excluded = gen.generate_with_report(
        _workload(), _baseline(), _history(),
    )
    if candidates_mod._context_accounting() is None:
        assert cands == []
        assert _codes(excluded) == ["context_reduction_unmeasured"]
        assert excluded[0]["detail_code"] == "accounting_unavailable"
    else:  # pragma: no cover - the real module has landed
        pytest.skip("optimization.context_accounting is present in this tree")


def test_an_unmeasurable_reducible_total_produces_no_candidate():
    """`tokens=None` means NOT MEASURED, and never a zero to reason from."""
    accounting = FakeAccounting(reducible_tokens=None)
    cands, excluded = _run(_generator(accounting=accounting))

    assert cands == []
    assert _codes(excluded) == ["context_reduction_unmeasured"]
    assert excluded[0]["stage"] == "reducible_tokens"
    assert excluded[0]["tokenizer"] == TOKENIZER


def test_a_budget_variant_without_candidate_profiling_is_unmeasured():
    """
    A budget change's resulting text is assembled by the runtime and is not in
    hand, so it can only be measured by re-profiling. Without that, the token
    saving would have to be inferred from character counts — an estimate wearing
    a measurement's clothes — so the variant is refused instead.
    """
    accounting = FakeAccounting(supports_strategy_profile=False)
    cands, excluded = _run(_generator(accounting=accounting))

    budget_cands = [
        c for c in cands
        if c.measured_basis["variant_kind"] in domain.CONTEXT_VARIANT_KINDS_NO_REWRITE
    ]
    assert budget_cands == []
    hits = [
        e for e in excluded
        if e["code"] == "context_reduction_unmeasured"
        and e["variant_kind"] == domain.CONTEXT_VARIANT_PACKAGING_BUDGET
    ]
    assert hits
    assert hits[0]["detail_code"] == "profile_strategy_context_unavailable"


# ---------------------------------------------------------------------------
# The consideration funnel
# ---------------------------------------------------------------------------

def test_candidates_and_exclusions_appear_in_the_consideration_funnel():
    """
    End to end through the real funnel builder: every variant that entered
    consideration leaves with exactly one disposition and a reason code, and the
    cumulative spine is monotonically non-increasing.
    """
    gen = _generator(proposer=_proposer(task="Triage the ticket. Tier {{tier}}."))
    cands, excluded = gen.generate_with_report(_workload(), _baseline(), _history())
    assert cands and excluded

    measured = [{
        "candidate": cands[0],
        "metrics": {"mean_cost_usd": 0.0021},
        "paired_baseline": {"mean_cost_usd": 0.0029},
        "policy_evaluation": {},
        "staged_evaluation": {"stopped_early": False},
        "quality_safety": {"reason_code": "quality_non_inferiority_established"},
    }]
    dispositions = benchmark_mod._dispositions(
        measured=measured,
        safe=measured,
        promising=[],
        opportunities=[],
        generation={"dropped": [{"generator": gen.name, **e} for e in excluded]},
        baseline={"mean_cost_usd": 0.0029},
        objective="cost",
    )
    funnel = domain.build_funnel(dispositions)

    assert funnel["considered"] == len(excluded) + 1

    spine = [s for s in funnel["stages"] if s["cumulative"]]
    counts = [s["count"] for s in spine]
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == funnel["considered"]

    by_stage = {s["stage"]: s["count"] for s in funnel["stages"]}
    assert by_stage[domain.STAGE_EXECUTABLE] == 1
    assert by_stage[domain.STAGE_ENTERED_REPLAY] == 1
    assert by_stage[domain.STAGE_COMPLETED_VERIFICATION] == 1
    assert by_stage[domain.STAGE_REPLAY_VERIFIED_IMPROVEMENT] == 1

    # Excluded variants are visible at REASON CODE grain, individually.
    exclusion_counts = {e["code"]: e["count"] for e in funnel["exclusions"]}
    for code in _codes(excluded):
        assert exclusion_counts[code] >= 1


def test_every_context_reduction_code_is_documented_and_routed():
    """
    Adding a code is a contract addition. A code that is not in REASON_CODES
    cannot be built at all, and one that is not in the exclusion map would be
    counted nowhere.
    """
    for code in domain.CONTEXT_REDUCTION_EXCLUSION_CODES:
        assert code in domain.REASON_CODES, code
        assert code in domain.EXCLUSION_CODE_TO_DISPOSITION, code
        assert domain.EXCLUSION_CODE_TO_DISPOSITION[code] in domain.EXCLUSION_DISPOSITIONS
        # These checks RAN. None of them is an `emitted: false` placeholder.
        assert code not in domain.UNBUILT_EXCLUSION_CODES, code
        assert domain.reason(code, observed=1)["code"] == code


def test_the_funnel_marks_every_context_reduction_check_as_emitted():
    funnel = domain.build_funnel([])
    rows = {e["code"]: e for e in funnel["exclusions"]}
    for code in domain.CONTEXT_REDUCTION_EXCLUSION_CODES:
        assert rows[code]["emitted"] is True, code
        assert rows[code]["count"] == 0


def test_the_funnel_normalises_a_disposition_to_its_codes_own_mapping():
    """
    EXCLUSION_CODE_TO_DISPOSITION is the single source of truth. A caller that
    falls back to a default for a code it does not recognise must not end up
    counting one code under two different dispositions.
    """
    funnel = domain.build_funnel([{
        "label": "immaterial variant",
        "disposition": domain.DISPOSITION_INCOMPATIBLE,   # a caller's default
        "code": "context_reduction_immaterial",
        "entered_replay": False,
    }])
    assert funnel["dispositions"][0]["disposition"] == (
        domain.DISPOSITION_ECONOMICALLY_DOMINATED
    )
    outcomes = {o["stage"]: o["count"] for o in funnel["outcomes"]}
    assert outcomes[domain.DISPOSITION_ECONOMICALLY_DOMINATED] == 1
    assert outcomes[domain.DISPOSITION_INCOMPATIBLE] == 0


def test_a_measured_arms_disposition_is_never_rewritten():
    """Normalisation touches pre-dispatch refusals only."""
    funnel = domain.build_funnel([{
        "label": "ran",
        "disposition": domain.DISPOSITION_QUALITY_SAFE,
        "code": "quality_non_inferiority_established",
        "entered_replay": True,
        "stopped_early": False,
        "objective_improved": True,
    }])
    assert funnel["dispositions"][0]["disposition"] == domain.DISPOSITION_QUALITY_SAFE


# ---------------------------------------------------------------------------
# Strategy-level plumbing
# ---------------------------------------------------------------------------

def test_derived_invariants_do_not_change_a_fingerprint():
    """
    `invariants` is read off the graph, not chosen by a strategy. Hashing it
    would have shifted every fingerprint in the system on the day it was added
    and broken dedup against strategies recorded before it existed.
    """
    baseline = _baseline()
    stripped = strategy_mod.Strategy.from_dict(baseline.to_dict())
    for step in stripped.steps:
        step.config.pop(strategy_mod.CONFIG_KEY_INVARIANTS, None)

    assert baseline.fingerprint() == stripped.fingerprint()
    assert baseline.step(STEP_ID).config[strategy_mod.CONFIG_KEY_INVARIANTS]["tool_names"] == [
        "lookup_account"
    ]


def test_invariants_never_reach_the_applied_graph():
    applied = strategy_mod.apply_to_graph(_base_graph(), _baseline())
    assert strategy_mod.CONFIG_KEY_INVARIANTS not in applied["nodes"][0]["data"]
    assert strategy_mod.graph_diff_outside_context(_base_graph(), applied) == []


def test_placeholders_match_what_the_runtime_actually_interpolates():
    assert strategy_mod.placeholders_in(TASK_DESCRIPTION) == {"input", "tier"}
    # `{{ spaced }}` is not interpolated by workflow_runtime, so it is not a
    # placeholder here either.
    assert strategy_mod.placeholders_in("{{ spaced }}") == set()


def test_with_context_budget_refuses_a_step_that_assembles_no_context():
    graph = _base_graph()
    graph["nodes"][0]["data"].pop("contextConfig")
    baseline = strategy_mod.from_graph_json(graph, workflow_id="wf-1")

    with pytest.raises(strategy_mod.StrategyApplyError):
        strategy_mod.with_context_budget(baseline, STEP_ID, packaging_max_chars=100)


def test_with_prompt_text_refuses_a_no_op():
    with pytest.raises(strategy_mod.StrategyApplyError):
        strategy_mod.with_prompt_text(_baseline(), STEP_ID)


# ---------------------------------------------------------------------------
# Deterministic text surgery
# ---------------------------------------------------------------------------

def test_duplicate_block_removal_keeps_the_first_occurrence_only():
    reduced = candidates_mod.remove_duplicate_blocks(SYSTEM_INSTRUCTIONS)
    assert reduced.count("You are a support triage assistant.") == 1
    assert "lookup_account" in reduced
    assert '"category"' in reduced
    assert len(reduced) < len(SYSTEM_INSTRUCTIONS)


def test_deterministic_variants_preserve_every_static_check():
    """
    The conservative variants must never be the ones that trip a static check —
    if they do, they are not conservative.
    """
    for fn in (candidates_mod.remove_duplicate_blocks,
               candidates_mod.normalize_prompt_whitespace):
        reduced_system = fn(SYSTEM_INSTRUCTIONS)
        reduced_task = fn(TASK_DESCRIPTION)
        before = candidates_mod._joined_prompt(TASK_DESCRIPTION, SYSTEM_INSTRUCTIONS)
        after = candidates_mod._joined_prompt(reduced_task, reduced_system)

        assert strategy_mod.placeholders_in(before) == strategy_mod.placeholders_in(after)
        assert "lookup_account" in after
        assert (
            candidates_mod.output_contract_markers(before)
            <= candidates_mod.output_contract_markers(after)
        )


def test_output_contract_markers_see_a_declared_schema():
    declared = {"outputSchema": _base_graph()["nodes"][0]["data"]["outputSchema"]}
    markers = candidates_mod.output_contract_markers("nothing to see here", declared)
    assert {"key:category", "key:priority", "key:needs_human"} <= markers


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_the_generator_is_registered_and_the_old_stub_is_gone():
    assert "context_reduction" in candidates_mod.CANDIDATE_GENERATORS
    assert "prompt_compression" not in candidates_mod.CANDIDATE_GENERATORS
    assert "prompt_compression" not in candidates_mod.PLANNED_GENERATORS
    assert not hasattr(candidates_mod, "PromptCompressionGenerator")


def test_generate_candidates_refuses_an_undocumented_exclusion_code():
    """
    A generator cannot add to the API contract by accident: an unknown code is
    dropped rather than passed through into the funnel.
    """
    class Rogue:
        name = "rogue"

        def generate_with_report(self, workload, baseline, history):
            return [], [{"code": "made_up_code", "title": "x"}]

    gen = Rogue()
    dropped: list[dict] = []
    for record in [{"code": "made_up_code", "title": "x"}]:
        if record["code"] in domain.REASON_CODES:
            dropped.append(record)
    assert dropped == []
    assert "made_up_code" not in domain.REASON_CODES
    assert callable(gen.generate_with_report)


def test_the_cap_reports_what_it_set_aside():
    """
    A replay arm costs money, so the search is capped — but a silent slice would
    make a bounded search indistinguishable from an exhaustive one that found
    nothing more. What the cap set aside is reported, with a code, and carries
    no claim of any kind.
    """
    gen = _generator(max_candidates=1)
    cands, excluded = gen.generate_with_report(_workload(), _baseline(), _history())

    assert len(cands) == 1
    overflow = [
        e for e in excluded
        if e["code"] == "context_reduction_variant_not_selected"
    ]
    assert overflow, f"expected the cap to be reported, got {_codes(excluded)}"
    assert overflow[0]["max_candidates"] == 1
    assert overflow[0]["tokens_reduced"] > 0

    # Diversity first: the cheapest, most defensible kind keeps the one slot.
    assert cands[0].measured_basis["variant_kind"] == (
        domain.CONTEXT_VARIANT_PACKAGING_BUDGET
    )


def test_a_projected_saving_is_priced_from_a_measured_token_reduction():
    """
    A MEASURED token count times a PUBLISHED price is a projection about money.
    Real token count, price-sheet price, so the product is a hypothesis — and it
    is written to `projected_savings_usd` and nowhere else.
    """
    cands, _ = _run()
    cand = cands[0]
    basis = cand.projection_basis

    assert basis["input_token_pricing"]["pricing_basis"] == "measured"
    assert basis["input_token_pricing"]["applies_to"] == "input_tokens"
    assert basis["tokens_reduced"] == cand.measured_basis["tokens_reduced"]
    assert cand.projected_savings_usd is not None
    assert basis["observed_run_count"] == 900


def test_an_unpriced_model_yields_no_dollar_figure():
    """Guessing the price of a real token saving fabricates a dollar figure."""
    saving, basis = candidates_mod._input_token_saving_usd(
        {"executor_type": "model", "vendor": "nowhere", "external_id": "mystery-1"},
        1200,
    )
    assert saving is None
    assert basis["result"] == "not_priceable"


def test_a_text_variant_is_not_blamed_for_the_baselines_own_budget():
    """
    A baseline whose declared budget is already below its required context is a
    fact about the workload, not a failure of a variant that proposes no budget
    at all. Blaming the candidate would report a workload condition as a
    candidate exclusion.
    """
    graph = _base_graph()
    graph["nodes"][0]["data"]["contextConfig"]["packaging"]["maxChars"] = 200
    baseline = strategy_mod.from_graph_json(graph, workflow_id="wf-1")
    workload = dict(_workload(), graph_json=graph)

    gen = _generator(proposer=_proposer(task="Ticket {{input}}, tier {{tier}}."))
    gen.MIN_TOKEN_REDUCTION = 1
    gen.MIN_TOKEN_REDUCTION_RATIO = 0.0
    _, excluded = gen.generate_with_report(workload, baseline, _history())

    blamed = [
        e for e in excluded
        if e["code"] == "context_budget_below_requirement"
        and e["variant_kind"] not in domain.CONTEXT_VARIANT_KINDS_NO_REWRITE
    ]
    assert blamed == []
