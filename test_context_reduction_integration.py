"""
The two halves of context efficiency, composed.

`test_context_reduction.py` fakes the accounting module, and `test_context_
accounting.py` never sees a candidate. Both suites can pass while the pair does
not compose — which is exactly what happened: the generator asked the accounting
module for `count_tokens` and `profile_strategy_context`, neither existed, and
every budget variant was correctly but uselessly excluded as unmeasured.

So this file uses the REAL `optimization.context_accounting` with the REAL
tokenizer, and proves the thing the dimension is for: a maxChars budget change
produces a smaller MEASURED token count, through the same code path and the same
tokenizer as the baseline it is subtracted from.

Only database plumbing is stubbed — which graph a workload maps to, and what a
knowledge asset resolves to. Everything measured here is measured for real.
"""
from __future__ import annotations

import pytest

from optimization import context_accounting as acct
from optimization import strategy as strategy_mod

from test_context_reduction import (
    SOURCE_TEXTS,
    STEP_ID,
    _base_graph,
    _cases,
    _workload,
)

tiktoken = pytest.importorskip("tiktoken", reason="real tokenizer required")


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """Stub only the DB plumbing: graph lookup, asset text, consolidation."""
    monkeypatch.setattr(
        acct, "_load_workload_graph", lambda org_id, workload: workload["graph_json"]
    )
    monkeypatch.setattr(acct, "_consolidation_may_apply", lambda *a, **k: False)

    import context_runtime

    def _collect(sources_config, context, org_id, deployment_id=None):
        out = []
        for s in sources_config:
            label = s.get("label")
            text = SOURCE_TEXTS.get(label)
            if text is None:
                continue
            per_source = s.get("maxChars")
            if per_source:
                text = text[:per_source]
            out.append(
                {
                    "source_type": s.get("type"),
                    "label": label,
                    "raw_text": text,
                    "estimated_chars": len(text),
                }
            )
        return out

    monkeypatch.setattr(context_runtime, "_collect_sources", _collect)


def _profile(graph=None, strategy=None):
    workload = _workload()
    if graph is not None:
        workload["graph_json"] = graph
    cases = _cases(12)
    if strategy is None:
        return acct.profile_workload_context(
            "org-1", workload, cases=cases, step_id=STEP_ID
        )
    return acct.profile_strategy_context(
        "org-1", workload, strategy, cases=cases, step_id=STEP_ID
    )


def test_the_baseline_context_is_measured_with_a_real_tokenizer():
    p = _profile()
    assert p is not None
    assert p.tokenizer == "o200k_base"          # gpt-4.1, looked up not assumed
    assert isinstance(p.total_tokens, int) and p.total_tokens > 0
    assert p.coverage == 1.0
    # Something is actually reducible, or there would be nothing to propose.
    assert isinstance(p.reducible_tokens, int) and p.reducible_tokens > 0


def test_a_budget_variant_measures_smaller_through_the_same_path():
    """
    The load-bearing test. A maxChars reduction is not text the caller holds —
    the runtime assembles it — so the only honest measurement is to re-profile
    the candidate strategy. Both numbers must come from the same function and
    the same tokenizer, or subtracting them manufactures a saving.
    """
    baseline = _profile()
    assert baseline is not None and baseline.total_tokens is not None

    base_strategy = strategy_mod.from_graph_json(_base_graph(), workflow_id="wf-1")
    tighter = strategy_mod.with_context_budget(base_strategy, STEP_ID, packaging_max_chars=3000)

    candidate = _profile(strategy=tighter)
    assert candidate is not None, "candidate strategy must be profilable"
    assert candidate.total_tokens is not None, "a budget change must be measurable"

    # Same tokenizer, or the generator refuses the pair outright.
    assert candidate.tokenizer == baseline.tokenizer

    assert candidate.total_tokens < baseline.total_tokens, (
        f"budget 6200 -> 3000 measured {baseline.total_tokens} -> "
        f"{candidate.total_tokens}"
    )


def test_the_measured_reduction_is_not_a_character_ratio():
    """
    Tokens must be counted, not derived. If the saving were chars/4 it would
    track the character cut exactly; a real BPE count does not.
    """
    baseline = _profile()
    base_strategy = strategy_mod.from_graph_json(_base_graph(), workflow_id="wf-1")
    candidate = _profile(strategy=strategy_mod.with_context_budget(base_strategy, STEP_ID, packaging_max_chars=3000))

    token_ratio = candidate.total_tokens / baseline.total_tokens
    # The character budget went 6200 -> 3000. A chars/4 estimate would put the
    # token ratio at the character ratio; the real count does not land there.
    assert token_ratio != pytest.approx(3000 / 6200, abs=0.005)


def test_a_strategy_that_cannot_be_applied_is_unmeasured_not_zero_saving(monkeypatch):
    """
    None means NOT MEASURED. It must never be read as "saves nothing" — a
    candidate that could not be applied has no measurement at all, and the
    generator has to exclude it rather than score it at zero.
    """
    base_strategy = strategy_mod.from_graph_json(_base_graph(), workflow_id="wf-1")
    good = strategy_mod.with_context_budget(
        base_strategy, STEP_ID, packaging_max_chars=3000
    )

    def _boom(graph, strategy):
        raise strategy_mod.StrategyApplyError("cannot apply")

    monkeypatch.setattr(strategy_mod, "apply_to_graph", _boom)
    assert _profile(strategy=good) is None


def test_budgeting_a_step_that_does_not_exist_is_refused_at_the_strategy_layer():
    base_strategy = strategy_mod.from_graph_json(_base_graph(), workflow_id="wf-1")
    with pytest.raises(strategy_mod.StrategyApplyError):
        strategy_mod.with_context_budget(
            base_strategy, "no-such-step", packaging_max_chars=3000
        )


def test_count_tokens_refuses_without_a_named_tokenizer():
    text = "some prompt text that is definitely non-empty"
    assert acct.count_tokens(text, tokenizer=None) is None
    assert acct.count_tokens(text, tokenizer="not_a_real_encoding") is None
    assert acct.count_tokens(None, tokenizer="o200k_base") is None

    n = acct.count_tokens(text, tokenizer="o200k_base")
    assert isinstance(n, int) and n > 0
    # Counted, not estimated.
    assert n != len(text) // 4


def test_count_tokens_agrees_with_the_profile_tokenizer():
    """A rewrite measured by count_tokens is comparable to a profiled budget."""
    p = _profile()
    enc = tiktoken.get_encoding(p.tokenizer)
    sample = "You are a support triage assistant. Respond only with valid JSON."
    assert acct.count_tokens(sample, tokenizer=p.tokenizer) == len(enc.encode(sample))
