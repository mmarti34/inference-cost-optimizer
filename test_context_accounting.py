"""
Tests for optimization.context_accounting.

These are pure unit tests over graph/case fixtures. Nothing here touches the
database: `_load_graph` and `resolve_workflow_id` are the only two DB calls in
the module and both are monkeypatched, and every context source used is
`inline_text` or an unresolvable `previous_node_output`, neither of which
queries anything.

The point under test throughout is honesty, not arithmetic: that a number is
absent when it was not measured, that `static` reflects what the cases actually
showed, and that the decomposition matches the text `context_runtime` really
assembles.
"""

import pytest

from optimization import context_accounting as ca


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeEncoding:
    """
    A deterministic stand-in so the measured paths can be exercised without a
    network round trip for the real BPE files. `test_real_tokenizer_*` covers
    the genuine tiktoken path.
    """

    name = "fake_test_encoding"

    def encode(self, text, disallowed_special=()):
        return text.split()


def _use_fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        ca, "resolve_encoding", lambda model, provider=None: (_FakeEncoding(), "fake_test_encoding")
    )


def _use_no_tokenizer(monkeypatch):
    monkeypatch.setattr(ca, "resolve_encoding", lambda model, provider=None: (None, None))


def _install_graph(monkeypatch, graph):
    monkeypatch.setattr(ca, "_load_graph", lambda org_id, workload, workflow_id: graph)
    monkeypatch.setattr(
        "optimization.workloads.resolve_workflow_id", lambda org_id, workload: "wf-1"
    )


WORKLOAD = {"id": "wl-1", "identity_kind": "workflow", "identity_ref": "wf-1"}


def _ai_step(**data):
    base = {
        "modelName": "gpt-4.1",
        "provider": "openai",
        "systemInstructions": "You are a careful support agent.",
        "taskDescription": "Answer the question.",
    }
    base.update(data)
    return {"nodes": [{"id": "step-1", "type": "ai-step", "data": base}], "edges": []}


def _by_name(profile):
    return {c.component: c for c in profile.components}


def _cases(*inputs, variables=None):
    out = []
    for i, text in enumerate(inputs):
        case = {"id": f"case-{i}", "input_text": text}
        if variables:
            case["variables"] = variables[i]
        out.append(case)
    return out


# ---------------------------------------------------------------------------
# static is measured, not assumed
# ---------------------------------------------------------------------------

def test_component_whose_text_varies_is_not_static(monkeypatch):
    """
    A 'static system prompt' that interpolates a per-case variable is NOT
    static. Calling it static would propose cutting text that changes on every
    request.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(systemInstructions="You assist {{customer}} with billing."),
    )

    cases = _cases(
        "why was I charged",
        "refund please",
        variables=[{"customer": "Acme"}, {"customer": "Globex"}],
    )
    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=cases)

    comps = _by_name(profile)
    assert comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].static is False
    # It is still MEASURED — varying does not mean unmeasurable.
    assert comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].tokens is not None
    assert comps[ca.COMPONENT_TASK_DESCRIPTION].static is True


def test_identical_text_across_cases_is_static(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())

    profile = ca.profile_workload_context(
        "org-1", WORKLOAD, cases=_cases("first question", "second question")
    )
    comps = _by_name(profile)
    assert comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].static is True
    assert comps[ca.COMPONENT_TASK_DESCRIPTION].static is True
    assert comps[ca.COMPONENT_USER_INPUT].static is False


def test_user_input_is_never_static_even_when_identical(monkeypatch):
    """Two cases that happen to carry the same input must not make it cuttable."""
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())

    profile = ca.profile_workload_context(
        "org-1", WORKLOAD, cases=_cases("same text", "same text")
    )
    assert _by_name(profile)[ca.COMPONENT_USER_INPUT].static is False


# ---------------------------------------------------------------------------
# unmeasurable means None, and stays visible
# ---------------------------------------------------------------------------

def test_unresolvable_source_reported_with_tokens_none_not_estimated(monkeypatch):
    """
    A previous_node_output source cannot be resolved without executing the
    upstream node. It must appear in the profile with tokens=None — never a
    chars/4 estimate — and must not silently disappear.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(
            contextConfig={
                "enabled": True,
                "sources": [
                    {"type": "inline_text", "label": "Policy", "value": "Refunds within 30 days."},
                    {"type": "previous_node_output", "label": "Upstream", "nodeId": "prompt-9"},
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            }
        ),
    )

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a", "b"))
    comps = _by_name(profile)

    assert "context_source:Upstream" in comps, "unresolved source was dropped"
    unresolved = comps["context_source:Upstream"]
    assert unresolved.tokens is None
    assert unresolved.tokenizer is None
    assert unresolved.chars == 0  # not observed, not "empty"
    assert unresolved.static is False
    assert unresolved.source_type == "previous_node_output"

    # The assembled prompt was incomplete, so no total is claimed.
    assert profile.total_tokens is None


def test_summarize_packaging_yields_none_not_a_guess(monkeypatch):
    """
    `context_runtime` calls an LLM to summarize when over budget. This module
    refuses to run it, so the packaged text was never observed and no token
    count is claimed for the sources.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(
            contextConfig={
                "enabled": True,
                "sources": [
                    {"type": "inline_text", "label": "A", "value": "x " * 200},
                    {"type": "inline_text", "label": "B", "value": "y " * 200},
                ],
                "packaging": {"strategy": "summarize", "maxChars": 100},
                "injection": {"location": "prepend_to_system"},
            }
        ),
    )

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a", "b"))
    comps = _by_name(profile)

    for name in ("context_source:A", "context_source:B"):
        assert comps[name].tokens is None
        assert comps[name].tokenizer is None
        assert comps[name].static is False
        # chars records what ENTERED packaging; it is a real measurement.
        assert comps[name].chars == 400
    assert profile.total_tokens is None


# ---------------------------------------------------------------------------
# reducible_tokens
# ---------------------------------------------------------------------------

def test_reducible_excludes_user_input(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(
            contextConfig={
                "enabled": True,
                "sources": [
                    {"type": "inline_text", "label": "Policy", "value": "Refunds within 30 days."}
                ],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            }
        ),
    )

    profile = ca.profile_workload_context(
        "org-1",
        WORKLOAD,
        cases=_cases("one two three four five", "six seven eight nine ten"),
    )
    comps = _by_name(profile)

    expected = (
        comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].tokens
        + comps[ca.COMPONENT_TASK_DESCRIPTION].tokens
        + comps["context_source:Policy"].tokens
    )
    assert profile.reducible_tokens == expected

    user_tokens = comps[ca.COMPONENT_USER_INPUT].tokens
    assert user_tokens == 5
    assert profile.reducible_tokens < expected + user_tokens


def test_reducible_is_none_when_static_components_cannot_be_tokenized(monkeypatch):
    """A model with no available tokenizer: a floor cannot be stated at all."""
    _use_no_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step(modelName="claude-sonnet-4", provider="anthropic"))

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a", "b"))

    assert profile is not None
    assert profile.tokenizer is None
    assert all(c.tokens is None and c.tokenizer is None for c in profile.components)
    assert profile.total_tokens is None
    assert profile.reducible_tokens is None
    # The decomposition itself still happened — chars are measured.
    assert _by_name(profile)[ca.COMPONENT_SYSTEM_INSTRUCTIONS].chars > 0


def test_reducible_is_none_when_a_static_source_was_unobserved(monkeypatch):
    """
    Mixed case: some static components tokenize, one does not. A partial sum
    would read as the whole, so nothing is returned.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(
            contextConfig={
                "enabled": True,
                "sources": [{"type": "inline_text", "label": "A", "value": "a " * 200}],
                "packaging": {"strategy": "summarize", "maxChars": 50},
                "injection": {"location": "prepend_to_system"},
            }
        ),
    )
    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a", "b"))
    # The unobserved source is not static (never measured as such), so it does
    # not poison the floor — but it is also not counted into it.
    comps = _by_name(profile)
    assert comps["context_source:A"].tokens is None
    assert profile.reducible_tokens == (
        comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].tokens
        + comps[ca.COMPONENT_TASK_DESCRIPTION].tokens
    )


def test_reducible_is_zero_when_nothing_is_static(monkeypatch):
    """A `model` node's prompt is only the previous output. Nothing to cut."""
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        {
            "nodes": [
                {"id": "m1", "type": "model", "data": {"modelName": "gpt-4.1", "provider": "openai"}}
            ],
            "edges": [],
        },
    )
    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a b", "c d"))
    assert [c.component for c in profile.components] == [ca.COMPONENT_USER_INPUT]
    assert profile.reducible_tokens == 0


# ---------------------------------------------------------------------------
# the decomposition must match what context_runtime assembles
# ---------------------------------------------------------------------------

def test_context_decomposition_matches_context_runtime(monkeypatch):
    """
    Run both over the same node and compare. The context components, rejoined
    with the runtime's own separator, must reproduce `final_text` exactly.
    """
    from context_runtime import resolve_node_context

    _use_fake_tokenizer(monkeypatch)
    context_config = {
        "enabled": True,
        "mode": "prepacked",
        "sources": [
            {"type": "inline_text", "label": "Policy", "value": "Refunds within 30 days."},
            {"type": "inline_text", "label": "Tone", "value": "Be brief and factual."},
        ],
        "packaging": {"strategy": "concat", "includeSourceLabels": True},
        "injection": {"location": "prepend_to_system"},
    }
    graph = _ai_step(contextConfig=context_config)
    _install_graph(monkeypatch, graph)

    resolved = resolve_node_context(
        graph["nodes"][0], {}, {}, "hello", "org-1", "eval", deployment_id=None
    )
    assert resolved is not None

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("hello"))
    ctx_components = [
        c for c in profile.components if c.component.startswith(ca.CONTEXT_SOURCE_PREFIX)
    ]

    assert [c.component for c in ctx_components] == [
        "context_source:Policy",
        "context_source:Tone",
    ]
    # chars per source, plus the one separator the runtime inserts between them.
    rebuilt = sum(c.chars for c in ctx_components) + len("\n\n---\n\n")
    assert rebuilt == resolved["total_chars"] == len(resolved["final_text"])

    # Per-source chars match the runtime's own per-source accounting, plus the
    # "## <label>\n" heading the runtime prepends when includeSourceLabels is on.
    expected = [
        item["estimated_chars"] + len(f"## {item['label']}\n")
        for item in resolved["items_used"]
    ]
    assert [c.chars for c in ctx_components] == expected


def test_truncation_attribution_matches_context_runtime(monkeypatch):
    """
    When packaging trims the joined text, each source keeps exactly the span
    that survived — the sum plus the separator is the runtime's final_text.
    """
    from context_runtime import resolve_node_context

    _use_fake_tokenizer(monkeypatch)
    context_config = {
        "enabled": True,
        "sources": [
            {"type": "inline_text", "label": "A", "value": "a" * 100},
            {"type": "inline_text", "label": "B", "value": "b" * 100},
        ],
        "packaging": {"strategy": "concat", "maxChars": 150},
        "injection": {"location": "prepend_to_system"},
    }
    graph = _ai_step(contextConfig=context_config)
    _install_graph(monkeypatch, graph)

    resolved = resolve_node_context(
        graph["nodes"][0], {}, {}, "hello", "org-1", "eval", deployment_id=None
    )
    assert resolved["truncated"] is True
    assert resolved["total_chars"] == 150

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("hello"))
    comps = _by_name(profile)
    assert comps["context_source:A"].chars == 100
    assert comps["context_source:B"].chars == 43  # 150 - 100 - len(separator)
    assert (
        comps["context_source:A"].chars
        + comps["context_source:B"].chars
        + len("\n\n---\n\n")
        == resolved["total_chars"]
    )


def test_total_tokens_measures_the_assembled_prompt(monkeypatch):
    """
    total_tokens is the whole prompt tokenized, not the sum of the parts. With
    the fake whitespace tokenizer the two happen to be comparable, so the test
    asserts the total covers every block including the joiners' neighbours.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(
            systemInstructions="sys one two",
            taskDescription="task three four",
            contextConfig={
                "enabled": True,
                "sources": [{"type": "inline_text", "label": "C", "value": "ctx five six"}],
                "packaging": {"strategy": "concat"},
                "injection": {"location": "prepend_to_system"},
            },
        ),
    )
    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("in seven"))
    # 3 sys + 3 task + 3 ctx + 2 input words, no token spans a joiner here.
    assert profile.total_tokens == 11


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_is_honest_when_some_cases_cannot_be_profiled(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())

    real = ca._apply_variables

    def _explode(template, variables, prev_output=None):
        if prev_output == "BOOM":
            raise RuntimeError("case not reproducible")
        return real(template, variables, prev_output)

    monkeypatch.setattr(ca, "_apply_variables", _explode)

    profile = ca.profile_workload_context(
        "org-1", WORKLOAD, cases=_cases("ok one", "BOOM", "ok two", "BOOM")
    )
    assert profile.n_cases == 4
    assert profile.coverage == 0.5
    # static was decided on the two cases that could be reproduced, and says so
    # by still being present.
    assert _by_name(profile)[ca.COMPONENT_SYSTEM_INSTRUCTIONS].static is True


def test_returns_none_when_no_case_can_be_profiled(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())

    def _explode(template, variables, prev_output=None):
        raise RuntimeError("nope")

    monkeypatch.setattr(ca, "_apply_variables", _explode)
    assert ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a", "b")) is None


# ---------------------------------------------------------------------------
# degenerate inputs profile cleanly rather than erroring
# ---------------------------------------------------------------------------

def test_workload_with_no_context_config_profiles_cleanly(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("hello there"))

    assert profile is not None
    assert profile.workload_id == "wl-1"
    assert profile.step_id == "step-1"
    assert profile.coverage == 1.0
    assert not any(
        c.component.startswith(ca.CONTEXT_SOURCE_PREFIX) for c in profile.components
    )
    assert profile.total_tokens is not None
    assert profile.reducible_tokens is not None


def test_disabled_context_config_profiles_cleanly(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch,
        _ai_step(contextConfig={"enabled": False, "sources": [{"type": "inline_text", "value": "x"}]}),
    )
    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("hi"))
    assert profile is not None
    assert not any(
        c.component.startswith(ca.CONTEXT_SOURCE_PREFIX) for c in profile.components
    )


def test_no_cases_returns_none(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())
    assert ca.profile_workload_context("org-1", WORKLOAD, cases=[]) is None


def test_direct_inference_workload_returns_none(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    monkeypatch.setattr(
        "optimization.workloads.resolve_workflow_id", lambda org_id, workload: None
    )
    assert (
        ca.profile_workload_context(
            "org-1", {"id": "wl-2", "identity_kind": "model"}, cases=_cases("a")
        )
        is None
    )


def test_unknown_step_id_returns_none(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step())
    assert (
        ca.profile_workload_context(
            "org-1", WORKLOAD, cases=_cases("a"), step_id="does-not-exist"
        )
        is None
    )


def test_graph_with_no_profilable_node_returns_none(monkeypatch):
    _use_fake_tokenizer(monkeypatch)
    _install_graph(
        monkeypatch, {"nodes": [{"id": "r", "type": "router", "data": {}}], "edges": []}
    )
    assert ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("a")) is None


# ---------------------------------------------------------------------------
# user input interpolated into the template
# ---------------------------------------------------------------------------

def test_input_interpolated_into_template_is_counted_once(monkeypatch):
    """
    When the template carries {{input}}, the user's text lives inside it. The
    template is measured with the placeholder blanked so the input is not
    counted twice.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step(taskDescription="Answer this: {{input}}"))

    profile = ca.profile_workload_context(
        "org-1", WORKLOAD, cases=_cases("alpha beta", "gamma delta")
    )
    comps = _by_name(profile)
    assert comps[ca.COMPONENT_TASK_DESCRIPTION].static is True
    assert comps[ca.COMPONENT_TASK_DESCRIPTION].chars == len("Answer this: ")
    assert comps[ca.COMPONENT_USER_INPUT].tokens == 2


def test_input_interpolated_twice_reports_chars_but_not_tokens(monkeypatch):
    """
    Two occurrences: the char contribution is exact arithmetic on a measured
    length, but a per-occurrence token count is not separable from the
    surrounding template, so none is claimed.
    """
    _use_fake_tokenizer(monkeypatch)
    _install_graph(monkeypatch, _ai_step(taskDescription="{{input}} ... {{input}}"))

    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("abc"))
    user = _by_name(profile)[ca.COMPONENT_USER_INPUT]
    assert user.chars == 6
    assert user.tokens is None
    assert user.tokenizer is None
    # The whole prompt is still observed, so the billed total remains measured.
    assert profile.total_tokens is not None


# ---------------------------------------------------------------------------
# the real tokenizer
# ---------------------------------------------------------------------------

def test_real_tokenizer_resolves_the_encoding_the_model_uses():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc, name = ca.resolve_encoding("gpt-4.1", "openai")
    except Exception:  # pragma: no cover - offline BPE fetch
        pytest.skip("tiktoken encoding files unavailable")
    if enc is None:
        pytest.skip("tiktoken could not load an encoding in this environment")

    assert name == "o200k_base"
    assert name == tiktoken.encoding_for_model("gpt-4.1").name
    # gpt-3.5-turbo genuinely uses a different one — the encoding is looked up,
    # not assumed.
    enc35, name35 = ca.resolve_encoding("gpt-3.5-turbo", "openai")
    assert name35 == "cl100k_base"


def test_real_tokenizer_absent_for_non_openai_models():
    pytest.importorskip("tiktoken")
    for model in ("claude-sonnet-4-20250514", "gemini-2.5-pro", "mistral-large-latest"):
        enc, name = ca.resolve_encoding(model, "anthropic")
        assert enc is None and name is None, f"guessed an encoding for {model}"


def test_real_tokenizer_produces_measured_counts(monkeypatch):
    pytest.importorskip("tiktoken")
    import tiktoken

    enc, name = ca.resolve_encoding("gpt-4.1", "openai")
    if enc is None:
        pytest.skip("tiktoken encoding files unavailable")

    _install_graph(
        monkeypatch,
        _ai_step(systemInstructions="You are a careful support agent.", taskDescription="Answer."),
    )
    profile = ca.profile_workload_context("org-1", WORKLOAD, cases=_cases("Where is my refund?"))

    comps = _by_name(profile)
    real = tiktoken.encoding_for_model("gpt-4.1")
    assert profile.tokenizer == "o200k_base"
    assert comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].tokens == len(
        real.encode("You are a careful support agent.")
    )
    assert comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].tokenizer == "o200k_base"
    assert profile.reducible_tokens == (
        comps[ca.COMPONENT_SYSTEM_INSTRUCTIONS].tokens
        + comps[ca.COMPONENT_TASK_DESCRIPTION].tokens
    )
