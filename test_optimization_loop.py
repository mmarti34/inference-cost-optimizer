"""
The optimization loop, end to end.

Every test here drives the REAL loop — workload -> candidates -> replay over the
same inputs -> policy comparison -> persisted evidence -> conclusion -> (only
sometimes) a recommendation. Nothing is stubbed except the two edges of the
system: the database (an in-memory table store) and `execute_workflow` (which
prices calls through the real `utils.pricing`, so a model missing from the price
sheet behaves here exactly as it would in production).

What these tests are really asserting is the product's central claim: that a
recommendation cannot appear without a measurement behind it, and that the
absence of a measurement is reported as ignorance rather than as approval of the
status quo.
"""
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# Stub Crypto so the modules under test can import cleanly under a bare venv
# (same shim the existing router tests use).
if "Crypto" not in sys.modules:
    _crypto = types.ModuleType("Crypto")
    _crypto.__path__ = []
    sys.modules["Crypto"] = _crypto
    for _sub in ("Cipher", "Cipher.AES", "Util", "Util.Padding", "Random"):
        sys.modules["Crypto." + _sub] = types.ModuleType("Crypto." + _sub)
    sys.modules["Crypto.Cipher"].AES = MagicMock()
    sys.modules["Crypto.Util.Padding"].pad = MagicMock()
    sys.modules["Crypto.Util.Padding"].unpad = MagicMock()
    sys.modules["Crypto.Random"].get_random_bytes = MagicMock(return_value=b"0" * 16)

from optimization import benchmark as benchmark_mod  # noqa: E402
from optimization import candidates as candidates_mod  # noqa: E402
from optimization import domain  # noqa: E402
from optimization import executors as executors_mod  # noqa: E402
from optimization import service as service_mod  # noqa: E402
from optimization import strategy as strategy_mod  # noqa: E402
from optimization import workloads as workloads_mod  # noqa: E402
from utils.pricing import get_pricing  # noqa: E402

ORG_ID = "11111111-1111-1111-1111-111111111111"
OTHER_ORG_ID = "99999999-9999-9999-9999-999999999999"
WORKLOAD_ID = "22222222-2222-2222-2222-222222222222"
WORKFLOW_ID = "33333333-3333-3333-3333-333333333333"
POLICY_ID = "44444444-4444-4444-4444-444444444444"
ENDPOINT_SLUG = "support-summary"

BASELINE_MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-4o-mini"
UNPRICED_MODEL = "acme-internal-v3"


# ---------------------------------------------------------------------------
# An in-memory stand-in for the supabase client
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """One PostgREST-ish query. Rebuilt per .table() call so filters never leak."""

    def __init__(self, store, table):
        self._store = store
        self._table = table
        self._op = "select"
        self._payload = None
        self._filters = []
        self._order = None
        self._limit = None

    # -- verbs
    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, row):
        self._op = "insert"
        self._payload = row
        return self

    def update(self, patch):
        self._op = "update"
        self._payload = patch
        return self

    # -- filters
    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def gte(self, col, val):
        self._filters.append(("gte", col, val))
        return self

    def in_(self, col, vals):
        self._filters.append(("in", col, list(vals)))
        return self

    def is_(self, col, val):
        self._filters.append(("is", col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    # -- execution
    def _matches(self, row):
        for op, col, val in self._filters:
            actual = row.get(col)
            if op == "eq":
                if val is None:
                    if actual is not None:
                        return False
                elif str(actual) != str(val):
                    return False
            elif op == "gte":
                if actual is None or str(actual) < str(val):
                    return False
            elif op == "in":
                if str(actual) not in {str(v) for v in val}:
                    return False
            elif op == "is":
                if val is None and actual is not None:
                    return False
                if val is not None and actual is not val:
                    return False
        return True

    def execute(self):
        rows = self._store.setdefault(self._table, [])

        if self._op == "insert":
            payload = self._payload
            batch = payload if isinstance(payload, list) else [payload]
            written = []
            for item in batch:
                row = dict(item)
                row.setdefault("id", str(uuid.uuid4()))
                row.setdefault("created_at", _iso(_now()))
                rows.append(row)
                written.append(dict(row))
            return _Result(written)

        matched = [r for r in rows if self._matches(r)]

        if self._op == "update":
            for r in matched:
                r.update(self._payload)
            return _Result([dict(r) for r in matched])

        if self._order:
            col, desc = self._order
            matched = sorted(
                matched,
                key=lambda r: (r.get(col) is None, r.get(col)),
                reverse=bool(desc),
            )
        if self._limit is not None:
            matched = matched[: self._limit]
        return _Result([dict(r) for r in matched])


class FakeSupabase:
    def __init__(self):
        self.store = {}

    def table(self, name):
        return _Query(self.store, name)

    def rows(self, name):
        return [dict(r) for r in self.store.get(name, [])]

    def seed(self, name, *rows):
        self.store.setdefault(name, []).extend(dict(r) for r in rows)


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _graph(model=BASELINE_MODEL, provider="openai"):
    return {
        "nodes": [
            {
                "id": "n1",
                "type": "ai-step",
                "data": {
                    "provider": provider,
                    "modelName": model,
                    "taskDescription": "Summarise the ticket.",
                },
            }
        ],
        "edges": [],
    }


def _seed(
    db,
    *,
    golden_inputs=20,
    constraints=None,
    materiality=None,
    production_runs=0,
    org_id=ORG_ID,
):
    db.seed("workloads", {
        "id": WORKLOAD_ID,
        "org_id": org_id,
        "project_id": None,
        "name": ENDPOINT_SLUG,
        "surface": "runtime",
        "identity_kind": "endpoint",
        "identity_ref": ENDPOINT_SLUG,
        "identity_level": "structural",
        "grain": "endpoint",
        "default_objective": "cost",
        "metadata": {},
    })
    db.seed("workflow_deployments", {
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "workflow_id": WORKFLOW_ID,
        "version": 3,
        "endpoint_slug": ENDPOINT_SLUG,
        "status": "promoted",
        "graph_json": _graph(),
    })
    db.seed("workflows", {
        "id": WORKFLOW_ID,
        "org_id": org_id,
        "graph_json": _graph(),
    })
    db.seed("eval_suites", {
        "id": str(uuid.uuid4()),
        "org_id": org_id,
        "workflow_id": WORKFLOW_ID,
        "name": "default",
        "enabled": True,
        # A deterministic check: exact-match against expected_output. This is
        # what gives the run a quality number with provenance strong enough to
        # be constrained on.
        "checks": [{"type": "deterministic", "name": "exact_match", "enabled": True}],
    })
    for i in range(golden_inputs):
        db.seed("golden_inputs", {
            "id": f"gi-{i}",
            "org_id": org_id,
            "workflow_id": WORKFLOW_ID,
            "name": f"case-{i}",
            "input_text": f"ticket {i}",
            "variables": None,
            "expected_output": f"summary {i}",
            "source": "production_import",
        })
    db.seed("optimization_policies", {
        "id": POLICY_ID,
        "org_id": org_id,
        "workload_id": WORKLOAD_ID,
        "policy_key": "default",
        "version": 4,
        "is_current": True,
        "enabled": True,
        "priority": 10,
        "name": "default",
        "constraints": constraints if constraints is not None else {"min_quality": 0.95},
        "materiality": materiality,
        "success_signal": {},
        "automation": {},
    })
    for i in range(production_runs):
        db.seed("workflow_runs", {
            "id": f"run-{i}",
            "org_id": org_id,
            "workflow_id": WORKFLOW_ID,
            "endpoint_slug": ENDPOINT_SLUG,
            "execution_mode": "production",
            "total_cost": 0.02,
            "total_latency_ms": 900,
            "node_results": [],
            "created_at": _iso(_now() - timedelta(days=1)),
        })


# ---------------------------------------------------------------------------
# A replay executor that prices calls the way production does
# ---------------------------------------------------------------------------

class FakeRuntime:
    """
    Stands in for `workflow_runtime.execute_workflow`.

    Cost is computed from the REAL price sheet for whatever model the graph
    carries, so a model absent from `shared/providers.json` falls back to the
    same estimated default it would in production — which is what makes the
    pricing-provenance test meaningful rather than staged.

    `quality_for` decides, per model, how many replay cases come back matching
    the expected output. That is the only lever the tests use to make a
    candidate "pass" or "miss" the quality floor.
    """

    def __init__(self, *, quality_for=None, latency_for=None, tokens=(1000, 300),
                 n_cases=20):
        self.quality_for = quality_for or {}
        self.latency_for = latency_for or {}
        self.tokens = tokens
        # How many replay cases the seeded suite holds. The pass/fail split is
        # computed against this rather than a constant, so a test that needs a
        # larger sample (to satisfy the non-inferiority evidence bar) still gets
        # the pass RATE it asked for.
        self.n_cases = n_cases
        self.calls = []

    def model_of(self, graph):
        for node in graph.get("nodes") or []:
            if (node.get("type") or "") in ("ai-step", "model"):
                data = node.get("data") or {}
                return (data.get("provider") or "openai"), data.get("modelName")
        return "openai", None

    def __call__(self, graph, input_text, org_id, _user, **kwargs):
        # The replay must go through the SAME eval path the eval UI uses.
        assert kwargs.get("execution_mode") == "eval"

        provider, model = self.model_of(graph)
        pricing = get_pricing(provider, model or "")
        prompt_tokens, completion_tokens = self.tokens
        cost = (
            prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
        ) / 1000.0

        index = int(str(input_text).split()[-1])
        pass_rate = self.quality_for.get(model, 1.0)
        # Deterministic: the first `pass_rate` share of cases match.
        matches = index < round(pass_rate * self.n_cases)

        self.calls.append({"model": model, "input_text": input_text})
        return {
            "final_output": f"summary {index}" if matches else "garbled",
            "total_cost": round(cost, 10),
            "total_latency_ms": self.latency_for.get(model, 800),
        }


def _candidate(model, *, provider="openai", title=None):
    """A caller-supplied candidate: one model substitution on the only step."""
    baseline = strategy_mod.from_graph_json(_graph(), workflow_id=WORKFLOW_ID)
    cand_strategy = candidates_mod._swap_model(baseline, "n1", provider, model)
    return candidates_mod.Candidate(
        title=title or f"Switch step n1 to {model}",
        strategy=cand_strategy,
        dimensions=strategy_mod.diff_dimensions(baseline, cand_strategy),
        generator="test_supplied",
        rationale="Supplied by the test.",
        evidence_source="none",
    )


@pytest.fixture
def db():
    return FakeSupabase()


def _patched(db, runtime):
    """Patch every module-level supabase handle the loop touches, plus the runtime."""
    targets = [
        "optimization.benchmark",
        "optimization.workloads",
        "optimization.policies",
        "optimization.service",
        "optimization.allocation",
        "optimization.outcomes",
        "optimization.evidence",
    ]
    patches = [patch(f"{t}.supabase", db) for t in targets]
    patches.append(patch("workflow_runtime.execute_workflow", runtime))
    return patches


def _run_loop(db, runtime, **kwargs):
    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        return benchmark_mod.run_benchmark(ORG_ID, workload_id=WORKLOAD_ID, **kwargs)
    finally:
        for p in reversed(patches):
            p.stop()


# ---------------------------------------------------------------------------
# 1. Cheaper and passes policy -> safe_improvement_found + a recommendation
# ---------------------------------------------------------------------------

def test_cheaper_and_passing_candidate_yields_safe_improvement_and_a_recommendation(db):
    # 60 replay cases, not 20. Under the conservative default (no worse than
    # baseline by more than 5 percentage points, 95% one-sided) a candidate that
    # TIES a perfect baseline needs 52 paired cases before non-inferiority is
    # established. 20 identical passes are genuinely compatible with a true 5pp
    # deficit, and the loop now says so instead of calling it verified.
    # production_runs deliberately != golden_inputs: with them equal, the
    # projected monthly figure and the verified sample figure coincide
    # numerically and the assertion below that they are different things would
    # pass for the wrong reason.
    _seed(db, golden_inputs=60, production_runs=90)
    runtime = FakeRuntime(n_cases=60)  # every model returns correct output

    result = _run_loop(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL)],
        create_recommendation=True,
        actor="user-1",
    )

    assert result["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    assert result["is_efficiency_finding"] is False  # only no_material_improvement is
    assert result["is_assessable"] is True

    # Both arms measured over the SAME inputs — that sameness is the whole
    # difference between a counterfactual and an observation.
    baseline_inputs = sorted(c["input_text"] for c in runtime.calls if c["model"] == BASELINE_MODEL)
    candidate_inputs = sorted(c["input_text"] for c in runtime.calls if c["model"] == CHEAP_MODEL)
    assert baseline_inputs == candidate_inputs
    assert len(baseline_inputs) == 60

    arms = db.rows("benchmark_candidate_results")
    assert {a["arm"] for a in arms} == {"baseline", "candidate"}
    assert all(a["evidence_source"] == "replay" for a in arms)

    conclusions = db.rows("benchmark_conclusions")
    assert len(conclusions) == 1
    assert conclusions[0]["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    assert conclusions[0]["is_current"] is True
    # The verdict is bound to the exact policy version that produced it.
    assert conclusions[0]["policy_version"] == 4
    assert conclusions[0]["policy_id"] == POLICY_ID

    # A recommendation exists, cites the benchmark, and requires a human.
    assert result["recommendation_created"] is True
    recs = db.rows("optimization_recommendations")
    assert len(recs) == 1
    rec = recs[0]
    assert rec["status"] == domain.STATUS_VERIFIED
    assert rec["approval_required"] is True
    assert rec["evidence_source"] == "replay"

    citations = db.rows("recommendation_evidence")
    assert [c["benchmark_id"] for c in citations] == [result["benchmark_id"]]

    # Savings live in their own columns and never cross.
    assert rec["verified_savings_usd"] is not None and rec["verified_savings_usd"] > 0
    assert rec["projected_savings_usd"] is not None and rec["projected_savings_usd"] > 0
    assert rec["projected_savings_usd"] != rec["verified_savings_usd"]
    # verified = measured delta over THIS sample; projected = that delta
    # extrapolated over measured monthly volume. Different numbers, by design.
    assert rec["baseline_reference"]["projection"]["result"] == "projected"
    assert rec["baseline_reference"]["measured_over_cases"] == 60
    assert rec.get("realized_savings_usd") is None
    assert rec["baseline_reference"]["benchmark_id"] == result["benchmark_id"]
    # First optimization of this workload: no ancestor to double-count against.
    assert rec["baseline_reference"]["derived_from_recommendation_id"] is None

    # 60 replay cases must not look like a production-scale result.
    assert rec["confidence"] < 0.5

    # The safety claim is EXPLICIT and structured — not folded into `confidence`.
    qs = result["quality_safety"]
    assert qs["established"] is True
    assert qs["method"] == "tango_score_paired_noninferiority"
    assert qs["n_pairs"] == 60
    assert (qs["discordant_b"], qs["discordant_c"]) == (0, 0)
    assert qs["allowed_regression"] == pytest.approx(0.05)
    assert qs["observed_regression"] == pytest.approx(0.0)
    assert qs["lower_confidence_bound"] > -0.05
    assert rec["quality_safety"]["established"] is True
    assert domain.confidence_band(rec["confidence"]) in ("low", "medium")


def test_confidence_from_a_replay_never_reaches_production_strength():
    """14 replay cases and 180k confirmed production outcomes are not the same claim."""
    replay = domain.compute_confidence(
        sample_size=14, evidence_source="replay", quality_provenance="deterministic"
    )
    production = domain.compute_confidence(
        sample_size=180_000, evidence_source="production",
        quality_provenance="business_outcome",
    )
    assert replay < 0.35
    assert production > 0.9


# ---------------------------------------------------------------------------
# 2. Cheaper but below the quality floor -> candidates_failed_policy,
#    and the near-miss arm SURVIVES the adverse verdict
# ---------------------------------------------------------------------------

def test_cheaper_but_quality_below_threshold_fails_policy_and_the_arm_is_retained(db):
    _seed(db, constraints={"min_quality": 0.95}, production_runs=60)
    # The cheap model is right 90% of the time: a real saving, 5pp under the floor.
    runtime = FakeRuntime(quality_for={CHEAP_MODEL: 0.90})

    result = _run_loop(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL)],
        create_recommendation=True,
    )

    assert result["conclusion"] == domain.CONCLUSION_CANDIDATES_FAILED_POLICY
    assert result["is_assessable"] is True          # we KNOW it fails
    assert result["is_efficiency_finding"] is False  # but that is not "you're optimal"

    codes = {r["code"] for r in result["reasons"]}
    assert "quality_below_threshold" in codes
    shortfall = next(r for r in result["reasons"] if r["code"] == "quality_below_threshold")
    assert shortfall["observed"] == pytest.approx(0.90)
    assert shortfall["required"] == pytest.approx(0.95)
    assert shortfall["shortfall"] == pytest.approx(0.05)
    assert shortfall["unit"] == "score"

    # No recommendation. Only safe_improvement_found may produce one.
    assert result["recommendation_created"] is False
    assert db.rows("optimization_recommendations") == []

    # The near miss is a first-class, independently queryable row: it saved real
    # money and missed by 5pp, and relaxing the floor later must be a re-read.
    near_miss = next(
        a for a in db.rows("benchmark_candidate_results") if a["arm"] == "candidate"
    )
    assert near_miss["quality"] == pytest.approx(0.90)
    assert near_miss["cost_delta_pct"] < 0          # genuinely cheaper
    assert near_miss["mean_cost_usd"] is not None   # genuinely measured

    queried = _list_candidate_results(db, benchmark_id=result["benchmark_id"])
    assert any(r["id"] == near_miss["id"] for r in queried)


def test_relaxing_only_the_absolute_floor_does_not_make_a_regression_safe(db):
    """
    THE REGRESSION THIS WHOLE CHANGE EXISTS FOR.

    A candidate 10 percentage points below a perfect baseline, re-read under a
    floor relaxed to exactly its own score. It clears the absolute floor by a
    margin of zero and is the cheapest thing measured. Under the old semantics
    that was `safe_improvement_found` and a VERIFIED recommendation. It must now
    stay a policy failure, because the floor was never the constraint that
    mattered — the baseline was.
    """
    _seed(db, golden_inputs=60, constraints={"min_quality": 0.95}, production_runs=60)
    runtime = FakeRuntime(quality_for={CHEAP_MODEL: 0.90}, n_cases=60)

    first = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])
    assert first["conclusion"] == domain.CONCLUSION_CANDIDATES_FAILED_POLICY

    # The customer relaxes ONLY the absolute floor, to exactly the candidate's
    # own measured score.
    for row in db.store["optimization_policies"]:
        row["constraints"] = {"min_quality": 0.90}
        row["version"] = 5

    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        second = benchmark_mod.reevaluate(ORG_ID, first["benchmark_id"])
    finally:
        for p in reversed(patches):
            p.stop()

    assert second["conclusion"] == domain.CONCLUSION_CANDIDATES_FAILED_POLICY
    codes = {r["code"] for r in second["reasons"]}
    assert "quality_below_threshold" not in codes      # the floor IS satisfied
    assert "quality_regression_above_threshold" in codes

    regression = next(
        r for r in second["reasons"] if r["code"] == "quality_regression_above_threshold"
    )
    # The facts a customer needs to judge it: 0.90 means nothing without 1.00.
    assert regression["baseline_quality"] == pytest.approx(1.0)
    assert regression["candidate_quality"] == pytest.approx(0.90)
    assert regression["observed"] == pytest.approx(0.10)
    assert regression["required"] == pytest.approx(0.05)
    assert regression["threshold_source"] == "default"   # nobody configured it

    assert db.rows("optimization_recommendations") == []


def test_relaxing_the_quality_floor_is_a_re_read_not_a_re_measurement(db):
    """Re-evaluation runs zero model calls and leaves the original verdict intact."""
    _seed(db, golden_inputs=60, constraints={"min_quality": 0.95}, production_runs=60)
    runtime = FakeRuntime(quality_for={CHEAP_MODEL: 0.90}, n_cases=60)

    first = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])
    assert first["conclusion"] == domain.CONCLUSION_CANDIDATES_FAILED_POLICY
    calls_after_benchmark = len(runtime.calls)

    # The customer relaxes the floor to 0.90 AND explicitly accepts up to a 20pp
    # regression against baseline. Both are required now: the absolute floor and
    # the relative ceiling are separate constraints and are ANDed.
    for row in db.store["optimization_policies"]:
        row["constraints"] = {"min_quality": 0.90, "max_quality_regression": 0.20}
        row["version"] = 5

    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        second = benchmark_mod.reevaluate(ORG_ID, first["benchmark_id"])
    finally:
        for p in reversed(patches):
            p.stop()

    assert second is not None
    assert second["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    # Not one additional model call.
    assert len(runtime.calls) == calls_after_benchmark

    # History is immutable: both verdicts coexist, each bound to its policy version.
    conclusions = db.rows("benchmark_conclusions")
    assert len(conclusions) == 2
    current = [c for c in conclusions if c["is_current"]]
    retired = [c for c in conclusions if not c["is_current"]]
    assert len(current) == 1 and len(retired) == 1
    assert current[0]["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    assert current[0]["policy_version"] == 5
    assert retired[0]["conclusion"] == domain.CONCLUSION_CANDIDATES_FAILED_POLICY
    assert retired[0]["policy_version"] == 4


# ---------------------------------------------------------------------------
# 3. Below the sample floor -> insufficient_evidence, nothing proposed,
#    and NOT counted as covered
# ---------------------------------------------------------------------------

def test_below_sample_floor_is_insufficient_evidence_not_an_error(db):
    _seed(db, golden_inputs=6)
    runtime = FakeRuntime()

    result = _run_loop(
        db, runtime,
        candidates=[_candidate(CHEAP_MODEL)],
        create_recommendation=True,
    )

    assert result["conclusion"] == domain.CONCLUSION_INSUFFICIENT_EVIDENCE
    reason = next(
        r for r in result["reasons"] if r["code"] == "sample_size_below_threshold"
    )
    assert reason["observed"] == 6
    assert reason["required"] == benchmark_mod.DEFAULT_MIN_SAMPLE_SIZE
    assert reason["unit"] == "cases"
    assert reason["dataset"] == "golden_inputs"

    # A refusal, not a failure: the benchmark completed and recorded why.
    assert result["status"] == "completed"
    benchmarks = db.rows("optimization_benchmarks")
    assert benchmarks[0]["status"] == "completed"
    assert benchmarks[0]["error"] is None

    # Nothing was run, so nothing was proposed.
    assert runtime.calls == []
    assert result["recommendation_created"] is False
    assert db.rows("optimization_recommendations") == []

    # Ignorance is structurally separated from a finding.
    assert result["is_assessable"] is False
    assert result["is_efficiency_finding"] is False
    assert result["coverage_class"] == domain.COVERAGE_NOT_COVERED
    assert domain.is_efficiency_finding(result["conclusion"]) is False
    assert result["more_data_changes_conclusion"] == domain.MORE_DATA_YES


def test_insufficient_evidence_is_never_no_opportunity():
    """The one rule the summary layer must never get wrong."""
    assert domain.CONCLUSION_INSUFFICIENT_EVIDENCE not in domain.NO_OPPORTUNITY_CONCLUSIONS
    assert domain.CONCLUSION_BENCHMARK_FAILED not in domain.NO_OPPORTUNITY_CONCLUSIONS
    assert domain.NO_OPPORTUNITY_CONCLUSIONS == (domain.CONCLUSION_NO_MATERIAL_IMPROVEMENT,)
    for conclusion in domain.IGNORANCE_CONCLUSIONS:
        assert domain.coverage_class(conclusion) == domain.COVERAGE_NOT_COVERED


# ---------------------------------------------------------------------------
# 4. A real but immaterial improvement -> no_material_improvement
# ---------------------------------------------------------------------------

def test_improvement_below_materiality_is_no_material_improvement(db):
    # 2% cheaper: measurable, eligible, and not worth touching production for.
    _seed(
        db,
        materiality={
            "thresholds": [
                {"metric": "cost", "comparator": "relative_decrease_at_least",
                 "value": 0.05, "unit": "ratio"},
            ],
            "combine": "any",
        },
        production_runs=0,
    )
    runtime = FakeRuntime()
    # Same model on both arms would dedupe; use a second real model and force a
    # 2% delta by adjusting the token mix the candidate arm consumes.
    cheap = _candidate(CHEAP_MODEL)

    baseline_price = get_pricing("openai", BASELINE_MODEL)
    target_cost = (
        (1000 * baseline_price["input"] + 300 * baseline_price["output"]) / 1000.0
    ) * 0.98

    class _TwoPercent(FakeRuntime):
        def __call__(self, graph, input_text, org_id, _user, **kwargs):
            out = super().__call__(graph, input_text, org_id, _user, **kwargs)
            _, model = self.model_of(graph)
            if model == CHEAP_MODEL:
                out["total_cost"] = round(target_cost, 10)
            return out

    runtime = _TwoPercent()
    result = _run_loop(db, runtime, candidates=[cheap], create_recommendation=True)

    assert result["conclusion"] == domain.CONCLUSION_NO_MATERIAL_IMPROVEMENT
    # This IS knowledge — the only conclusion that may be rendered as
    # "your configuration looks efficient".
    assert result["is_efficiency_finding"] is True
    assert result["is_assessable"] is True
    assert result["coverage_class"] == domain.COVERAGE_COVERED

    below = next(
        r for r in result["reasons"] if r["code"] == "improvement_below_materiality"
    )
    assert below["metric"] == "cost"
    assert below["required"] == pytest.approx(0.05)
    assert below["observed"] == pytest.approx(0.02, abs=0.005)

    assert result["recommendation_created"] is False
    assert db.rows("optimization_recommendations") == []


def test_nothing_measurable_is_ignorance_not_an_efficiency_finding(db):
    """
    A cheaper-looking candidate whose cost could not be judged must not come
    back as "not worth changing". That would render an absent comparison as
    approval of the status quo.
    """
    _seed(db, constraints={}, production_runs=0)

    class _NoCost(FakeRuntime):
        def __call__(self, graph, input_text, org_id, _user, **kwargs):
            out = super().__call__(graph, input_text, org_id, _user, **kwargs)
            out["total_cost"] = None
            return out

    result = _run_loop(db, _NoCost(), candidates=[_candidate(CHEAP_MODEL)])

    assert result["conclusion"] == domain.CONCLUSION_INSUFFICIENT_EVIDENCE
    assert result["is_efficiency_finding"] is False
    assert result["coverage_class"] == domain.COVERAGE_NOT_COVERED


# ---------------------------------------------------------------------------
# 5. Estimated pricing is never reported as measured
# ---------------------------------------------------------------------------

def test_pricing_provenance_separates_a_real_price_from_a_guess():
    known = executors_mod.pricing_provenance(
        [{"executor_type": "model", "vendor": "openai", "external_id": CHEAP_MODEL}]
    )
    assert known["basis"] == executors_mod.COST_BASIS_MEASURED
    assert known["estimated_models"] == []

    guessed = executors_mod.pricing_provenance(
        [{"executor_type": "model", "vendor": "openai", "external_id": UNPRICED_MODEL}]
    )
    assert guessed["basis"] == executors_mod.COST_BASIS_ESTIMATED
    assert guessed["estimated_models"][0]["model"] == UNPRICED_MODEL

    # A software step carries no token cost and cannot be mispriced.
    assert executors_mod.pricing_provenance(
        [{"executor_type": "software", "vendor": "optiml", "external_id": "workflow_router"}]
    )["basis"] == executors_mod.COST_BASIS_MEASURED


def test_a_cost_from_an_estimated_price_is_never_reported_as_measured(db):
    """
    An unknown model resolves to the fallback default ($0.001/$0.002), which
    happens to look far cheaper than gpt-4o. If that guess were treated as a
    measurement the loop would 'discover' a large saving that was never
    observed. It must instead refuse to conclude a cost improvement.
    """
    _seed(db, production_runs=60)
    runtime = FakeRuntime()

    result = _run_loop(
        db, runtime,
        candidates=[_candidate(UNPRICED_MODEL)],
        create_recommendation=True,
    )

    arm = next(a for a in db.rows("benchmark_candidate_results") if a["arm"] == "candidate")
    # The measured columns are NULL...
    assert arm["mean_cost_usd"] is None
    assert arm["total_cost_usd"] is None
    # ...and the guess is retained, clearly labelled, somewhere else.
    metrics = arm["outcome_metrics"]
    assert metrics["cost_basis"] == executors_mod.COST_BASIS_ESTIMATED
    assert metrics["mean_cost_estimated_usd"] is not None
    assert metrics["mean_cost_estimated_usd"] > 0
    assert metrics["pricing_provenance"]["estimated_models"][0]["model"] == UNPRICED_MODEL

    # Per-case rows obey the same rule.
    for case in arm["per_case_results"]:
        assert case["cost_usd"] is None
        assert case["cost_basis"] == executors_mod.COST_BASIS_ESTIMATED

    # And the verdict says why, rather than banking the fictitious saving.
    assert result["conclusion"] == domain.CONCLUSION_INSUFFICIENT_EVIDENCE
    codes = {r["code"] for r in result["reasons"]}
    assert "cost_pricing_estimated" in codes
    estimated = next(r for r in result["reasons"] if r["code"] == "cost_pricing_estimated")
    assert estimated["arm"] == "candidate"
    assert f"openai/{UNPRICED_MODEL}" in estimated["models"]

    assert result["recommendation_created"] is False
    assert db.rows("optimization_recommendations") == []


# ---------------------------------------------------------------------------
# Candidate generation: vendor price is a reason to measure, not evidence
# ---------------------------------------------------------------------------

def test_generated_candidates_carry_honest_evidence_labels(db):
    """
    A vendor price sheet proposes candidates at evidence_source='none'; measured
    history proposes at 'observational'. Neither may be recommended unmeasured.
    """
    baseline = strategy_mod.from_graph_json(_graph(), workflow_id=WORKFLOW_ID)
    workload = {"id": WORKLOAD_ID, "identity_kind": "endpoint", "identity_ref": ENDPOINT_SLUG}
    history = {
        "org_id": ORG_ID,
        "model_stats": {},
        "traffic": {"run_count": 300, "coverage": {}},
        "lookback_days": 30,
    }

    proposed = candidates_mod.AlternateModelGenerator().generate(workload, baseline, history)
    assert proposed, "expected the price sheet to offer something cheaper than gpt-4o"
    for cand in proposed:
        assert cand.evidence_source == "none"
        assert domain.evidence_strength(cand.evidence_source) == 0
        assert any(n["code"] == "vendor_price_only" for n in cand.notes)
        assert cand.dimensions  # it actually changes something applicable

    measured_history = {
        **history,
        "model_stats": {
            BASELINE_MODEL: {
                "model": BASELINE_MODEL, "provider": "openai", "runs": 40,
                "avg_cost": 0.01, "avg_latency": 900, "error_rate": 0.01,
            },
            CHEAP_MODEL: {
                "model": CHEAP_MODEL, "provider": "openai", "runs": 30,
                "avg_cost": 0.002, "avg_latency": 700, "error_rate": 0.01,
            },
        },
    }
    observed = candidates_mod.CheaperMeasuredModelGenerator().generate(
        workload, baseline, measured_history
    )
    assert observed
    assert all(c.evidence_source == "observational" for c in observed)
    assert all(
        domain.evidence_strength(c.evidence_source)
        < domain.evidence_strength("replay")
        for c in observed
    )
    assert all(
        any(n["code"] == "observational_not_counterfactual" for n in c.notes)
        for c in observed
    )


def test_the_loop_runs_end_to_end_from_generated_candidates(db):
    """No caller-supplied candidates: discovery, generation, replay, verdict."""
    _seed(db, golden_inputs=60, production_runs=120)
    runtime = FakeRuntime(n_cases=60)

    result = _run_loop(db, runtime, create_recommendation=True, actor="user-1")

    # Candidates really were generated from the price sheet, and every one of
    # them was replayed over the same cases as the baseline.
    arms = db.rows("benchmark_candidate_results")
    candidate_arms = [a for a in arms if a["arm"] == "candidate"]
    assert len(candidate_arms) >= 2
    assert all(a["generator"] == "alternate_model" for a in candidate_arms)
    assert {c["model"] for c in runtime.calls} != {BASELINE_MODEL}
    per_model_cases = {}
    for call in runtime.calls:
        per_model_cases.setdefault(call["model"], set()).add(call["input_text"])
    assert len(set(map(frozenset, per_model_cases.values()))) == 1, "arms saw different inputs"

    assert result["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    recs = db.rows("optimization_recommendations")
    assert len(recs) == 1
    assert recs[0]["approval_required"] is True
    assert recs[0]["status"] == domain.STATUS_VERIFIED
    # The winner is the cheapest ELIGIBLE arm, and it is the arm the conclusion
    # points at — not a separately re-derived choice.
    conclusion_row = db.rows("benchmark_conclusions")[0]
    selected = next(
        a for a in arms if a["id"] == conclusion_row["selected_candidate_result_id"]
    )
    assert selected["mean_cost_usd"] == min(
        a["mean_cost_usd"] for a in candidate_arms if a["mean_cost_usd"] is not None
    )
    assert recs[0]["title"] == selected["label"]


# ---------------------------------------------------------------------------
# Workload selection
# ---------------------------------------------------------------------------

def test_selection_ranks_on_measured_spend_and_states_why_it_skipped(db):
    _seed(db, golden_inputs=20)
    # A big, worthwhile workload...
    for i in range(40):
        db.seed("workflow_runs", {
            "id": f"big-{i}", "org_id": ORG_ID, "workflow_id": WORKFLOW_ID,
            "endpoint_slug": ENDPOINT_SLUG, "execution_mode": "production",
            "total_cost": 0.5, "created_at": _iso(_now() - timedelta(days=1)),
        })
    # ...and a registered but barely-used one.
    db.seed("workloads", {
        "id": "55555555-5555-5555-5555-555555555555", "org_id": ORG_ID,
        "name": "rare", "surface": "runtime", "identity_kind": "endpoint",
        "identity_ref": "rare-endpoint", "default_objective": "cost",
    })
    for i in range(3):
        db.seed("workflow_runs", {
            "id": f"rare-{i}", "org_id": ORG_ID, "workflow_id": "wf-rare",
            "endpoint_slug": "rare-endpoint", "execution_mode": "production",
            "total_cost": 0.001, "created_at": _iso(_now() - timedelta(days=1)),
        })
    # ...and traffic for an endpoint nobody has registered.
    for i in range(50):
        db.seed("workflow_runs", {
            "id": f"ghost-{i}", "org_id": ORG_ID, "workflow_id": "wf-ghost",
            "endpoint_slug": "unregistered", "execution_mode": "production",
            "total_cost": 0.02, "created_at": _iso(_now() - timedelta(days=1)),
        })

    with patch("optimization.workloads.supabase", db):
        out = workloads_mod.select_optimization_targets(ORG_ID)

    assert [t["endpoint_slug"] for t in out["targets"]] == [ENDPOINT_SLUG]
    target = out["targets"][0]
    assert target["observed_run_count"] == 40
    assert target["observed_cost_usd"] == pytest.approx(20.0)
    assert target["replay_cases"] == 20
    assert target["workload_id"] == WORKLOAD_ID

    skipped = {s["endpoint_slug"]: s for s in out["skipped"]}
    rare_codes = {r["code"] for r in skipped["rare-endpoint"]["reasons"]}
    assert "workload_volume_below_threshold" in rare_codes
    volume = next(
        r for r in skipped["rare-endpoint"]["reasons"]
        if r["code"] == "workload_volume_below_threshold" and r["unit"] == "runs"
    )
    assert volume["observed"] == 3
    assert volume["required"] == workloads_mod.MIN_RUNS_TO_OPTIMIZE

    ghost_codes = {r["code"] for r in skipped["unregistered"]["reasons"]}
    assert ghost_codes == {"workload_not_registered"}
    assert skipped["unregistered"]["workload_id"] is None

    # The floors it applied are part of the answer, not a hidden constant.
    assert out["floors"]["min_runs"] == workloads_mod.MIN_RUNS_TO_OPTIMIZE
    assert out["floors"]["window_days"] == 30


def test_selection_declines_a_workload_with_nothing_to_replay(db):
    _seed(db, golden_inputs=0)
    for i in range(40):
        db.seed("workflow_runs", {
            "id": f"r-{i}", "org_id": ORG_ID, "workflow_id": WORKFLOW_ID,
            "endpoint_slug": ENDPOINT_SLUG, "execution_mode": "production",
            "total_cost": 0.5, "created_at": _iso(_now() - timedelta(days=1)),
        })

    with patch("optimization.workloads.supabase", db):
        out = workloads_mod.select_optimization_targets(ORG_ID)

    assert out["targets"] == []
    reasons = {r["code"] for r in out["skipped"][0]["reasons"]}
    assert "no_replay_cases" in reasons


# ---------------------------------------------------------------------------
# Contract: codes and facts, never prose; org isolation
# ---------------------------------------------------------------------------

def test_every_emitted_reason_code_is_a_documented_one(db):
    _seed(db, constraints={"min_quality": 0.95}, production_runs=10)
    runtime = FakeRuntime(quality_for={CHEAP_MODEL: 0.5})
    result = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])

    assert result["reasons"]
    for r in result["reasons"]:
        assert r["code"] in domain.REASON_CODES
    # An undocumented code cannot slip into the contract.
    with pytest.raises(ValueError):
        domain.reason("looks_a_bit_expensive")


def test_a_benchmark_refuses_a_workload_from_another_org(db):
    _seed(db)
    runtime = FakeRuntime()
    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        with pytest.raises(benchmark_mod.BenchmarkError):
            benchmark_mod.run_benchmark(OTHER_ORG_ID, workload_id=WORKLOAD_ID)
    finally:
        for p in reversed(patches):
            p.stop()
    assert db.rows("optimization_benchmarks") == []


def test_a_recommendation_cannot_be_created_without_a_measured_strategy(db):
    """
    The reevaluate path holds stored rows, not executable strategies. It must
    decline to create a recommendation rather than reconstruct one from a hash.
    """
    _seed(db, golden_inputs=60, production_runs=60)
    runtime = FakeRuntime(n_cases=60)

    first = _run_loop(db, runtime, candidates=[_candidate(CHEAP_MODEL)])
    assert first["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    assert db.rows("optimization_recommendations") == []  # not requested

    created = benchmark_mod._create_recommendation_from_evidence(
        ORG_ID,
        benchmark_id=first["benchmark_id"],
        workload={"id": WORKLOAD_ID},
        objective="cost",
        winner={"candidate": benchmark_mod._StoredCandidate({"label": "x"}), "metrics": {}},
        baseline_strategy=None,
        confidence=0.2,
        sample_size=20,
        success_signal=domain.SuccessSignal(),
        conclusion=domain.CONCLUSION_SAFE_IMPROVEMENT,
        actor=None,
        service_mod=service_mod,
    )
    assert created is None


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _list_candidate_results(db, **kwargs):
    with patch("optimization.benchmark.supabase", db):
        return benchmark_mod.list_candidate_results(ORG_ID, **kwargs)


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------

def test_every_optimization_route_requires_org_membership():
    """No unauthenticated route may exist on this router."""
    import inspect

    from auth_dependency import require_org_member
    from routers.optimization_router import router as opt_router

    unguarded = []
    for route in opt_router.routes:
        signature = inspect.signature(route.endpoint)
        guarded = any(
            getattr(param.default, "dependency", None) is require_org_member
            for param in signature.parameters.values()
        )
        if not guarded:
            unguarded.append((sorted(route.methods), route.path))
    assert unguarded == []
    # And every one of them carries org_id in the PATH, so the handler can
    # re-assert it against the org the dependency actually verified.
    assert all("{org_id}" in route.path for route in opt_router.routes)


def _client(db, runtime):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth_dependency import AuthenticatedUser, require_org_member
    from routers.optimization_router import router as opt_router

    app = FastAPI()
    app.include_router(opt_router, prefix="/api")

    user = AuthenticatedUser(user_id="user-1", email="a@b.c", email_verified=True)
    user._verified_org_id = ORG_ID
    app.dependency_overrides[require_org_member] = lambda: user
    return TestClient(app)


def test_optimize_endpoint_returns_the_verdict_and_the_recommendation(db):
    _seed(db, golden_inputs=60, production_runs=120)
    runtime = FakeRuntime(n_cases=60)
    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        client = _client(db, runtime)
        resp = client.post(
            f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/optimize", json={}
        )
    finally:
        for p in reversed(patches):
            p.stop()

    assert resp.status_code == 200
    body = resp.json()
    assert body["conclusion"] == domain.CONCLUSION_SAFE_IMPROVEMENT
    # Codes and facts, never prose the frontend would have to parse.
    assert all(r["code"] in domain.REASON_CODES for r in body["reasons"])
    assert body["recommendation"]["governance"]["approval_required"] is True
    assert body["recommendation"]["status"] == domain.STATUS_VERIFIED
    assert body["recommendation"]["evidence"]["source"] == "replay"
    assert body["recommendation"]["evidence"]["benchmarks"][0]["benchmark_id"] == (
        body["benchmark_id"]
    )
    savings = body["recommendation"]["savings"]
    assert savings["verified_usd"] is not None
    assert savings["realized_usd"] is None  # never written before promotion


def test_optimize_endpoint_404s_on_a_workload_belonging_to_another_org(db):
    _seed(db, org_id=OTHER_ORG_ID)
    runtime = FakeRuntime()
    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        client = _client(db, runtime)
        resp = client.post(
            f"/api/optimization/{ORG_ID}/workloads/{WORKLOAD_ID}/optimize", json={}
        )
    finally:
        for p in reversed(patches):
            p.stop()

    assert resp.status_code == 404
    assert db.rows("optimization_benchmarks") == []


def test_optimization_targets_endpoint_reports_what_it_skipped(db):
    _seed(db, golden_inputs=0)
    for i in range(40):
        db.seed("workflow_runs", {
            "id": f"r-{i}", "org_id": ORG_ID, "workflow_id": WORKFLOW_ID,
            "endpoint_slug": ENDPOINT_SLUG, "execution_mode": "production",
            "total_cost": 0.5, "created_at": _iso(_now() - timedelta(days=1)),
        })
    runtime = FakeRuntime()
    patches = _patched(db, runtime)
    for p in patches:
        p.start()
    try:
        client = _client(db, runtime)
        resp = client.get(f"/api/optimization/{ORG_ID}/optimization-targets")
    finally:
        for p in reversed(patches):
            p.stop()

    assert resp.status_code == 200
    body = resp.json()
    assert body["targets"] == []
    assert body["skipped"]
    assert all(
        r["code"] in domain.REASON_CODES
        for s in body["skipped"] for r in s["reasons"]
    )
    assert body["floors"]["min_runs"] == workloads_mod.MIN_RUNS_TO_OPTIMIZE
