# The Optimization Layer

> **Benchmarks discover facts. Recommendations propose actions.**

That sentence settles more design questions than any other statement in this
package. Its consequences, each of which is enforced somewhere in the code:

- A benchmark can exist **forever** with no recommendation. `optimization_benchmarks`
  is owned by `workload_id`; `recommendation_id` is deprecated and never written.
- A recommendation **requires** evidence. `accept` refuses a recommendation that
  cites no benchmark (`routers/optimization_router.py`).
- Evidence **does not disappear** when a recommendation is rejected. Every
  measured arm lands in `benchmark_candidate_results`, independent of any verdict.
- A policy change **reinterprets** evidence; it does not invalidate it.
  `benchmark.reevaluate` re-derives a conclusion from stored results with zero
  model calls, and writes a new immutable `benchmark_conclusions` row.
- **Multiple recommendations** can arise from one piece of evidence, and
  **multiple pieces of evidence** can justify one recommendation. Hence the
  `recommendation_evidence` join table.

And the rule the whole product rests on:

> A candidate is never promoted because an LLM said it seemed good.
> `rationale` is prose. `optimization_benchmarks` is evidence.

---

## Internal north star (never customer-facing language)

OptiML is an economic scheduler for enterprise work. Given a unit of work it
should eventually choose among a model, an AI agent, an external tool,
deterministic software, a SaaS API, or a human — on cost, outcome, quality,
speed, reliability, risk, policy and business value.

**We are not building that now.** We are shipping the Runtime wedge. The schema
is arranged so that getting there is an `INSERT`, not a migration.

---

## Module layout, and the one boundary it enforces

The layout exists to keep **vendor metadata** and **empirical evidence** apart.
A price sheet says what a call *would* cost. Only a measurement says what it
*did* cost. Mixing them is how an optimization product starts lying.

| Module | Role |
|---|---|
| `domain.py` | Vocabulary and pure functions: objectives, lifecycle machine, evidence strength, provenance defaults, confidence, materiality, coverage, reason codes. No I/O. |
| `executors.py` | **VENDOR side.** Published prices, advertised capabilities, declared regions. Contains no query against any measurement table. |
| `evidence.py` | **EMPIRICAL side.** What was measured on this org's own workloads. Contains no vendor lookup. |
| `strategy.py` | Execution strategies: ordered executor steps; the runtime `graph_json` adapter and the direct-inference adapter. Dimension applicability is scoped **per execution surface**. |
| `attempts.py` | **The only `node_results` parser in the backend.** Thin domain layer over the existing execution records, on both surfaces. |
| `candidates.py` | Candidate generators. Stages 1–2 built; 4–6 are stubs that refuse. |
| `capabilities.py` | **DECLARED** executor capabilities and the adapter layer: what a model family accepts, and the named transformations that can make a request executable. A declaration table plus one generic matcher — no name checks at call sites. |
| `eligibility.py` | The **preflight gate** between hypothesis and spend. A generated candidate is not automatically a benchmark arm: provider, catalog, surface, policy, request-shape, capability, context-window, pricing and objective checks run first, and **no external provider request is made for an ineligible candidate**. Also holds the cost-objective screen — the one place a price sheet may remove an arm, and never a place it may claim a saving. |
| `benchmark.py` | The evidence engine, and the pure `evaluate_conclusion` verdict function. |
| `outcomes.py` | Outcome recording: delayed, plural, named, correctable, idempotent. |
| `policies.py` | Constraints that make a strategy invalid; versioned by insert. |
| `workloads.py` | Identity resolution (explicit / structural), discovery, and read-only selection of which workload is worth benchmarking. |
| `allocation.py` | Records which strategy was chosen and why. |
| `service.py` | Recommendation CRUD, lifecycle transitions, audit trail. |

---

## The one loop that runs today

```
observed Runtime workload            workloads.select_optimization_targets
  -> alternate model candidates      candidates.generate_candidates
  -> replayed against the SAME       benchmark._execute_arm, via
     golden inputs                   workflow_runtime.execute_workflow(execution_mode='eval')
  -> compared under the policy       policies.evaluate + domain.evaluate_materiality
  -> evidence persisted per ARM      benchmark_candidate_results
  -> ONE immutable conclusion        benchmark_conclusions
  -> recommendation, only on         service.create_recommendation, cited through
     safe_improvement_found          recommendation_evidence, approval_required=TRUE
```

`POST /{org_id}/workloads/{workload_id}/optimize` runs it end to end and returns
the verdict. `POST /{org_id}/workloads/{workload_id}/benchmark` runs the same
engine exploratorily and creates nothing, because discovering a fact and
proposing an action are different events.

**Model substitution is the only dimension this loop varies.** Everything else
in `SURFACE_APPLICABLE_DIMENSIONS` is applicable but has no generator behind it,
and everything in `UNSUPPORTED_DIMENSIONS` refuses at apply time.

### Selection is allowed to say no, and to say why

`select_optimization_targets` is READ-ONLY (discovery is what writes) and ranks
on **measured spend, then measured volume** — never on which workload looks
expensive. Two documented floors, both overridable per call:

| Floor | Value | Why |
|---|---|---|
| `MIN_RUNS_TO_OPTIMIZE` | 20 runs in the window | Below this a workload's own averages are dominated by individual-case noise. |
| `MIN_OBSERVED_SPEND_USD` | $0.10 in the window | Excludes only workloads whose entire measured spend rounds to nothing. |

Every workload passed over is returned under `skipped` with structured reason
codes (`workload_volume_below_threshold`, `no_replay_cases`,
`workload_not_registered`) and the facts behind them. **A workload missing from
`targets` was not assessed** — which is an absence of evidence, never a finding
that it is already efficient.

### Estimated pricing can never become a measured saving

`utils.pricing.get_pricing` returns a loud fallback guess ($0.001/$0.002 per 1k)
for a model id it cannot resolve. That guess is *cheaper than almost every real
model*, so treating it as a measurement would make the loop "discover" enormous
savings that were never observed — the single most dangerous failure mode in a
cost-optimization product.

`executors.pricing_provenance` resolves an arm's prices from its strategy
**before** the arm executes. When any model in it is unpriced:

- `mean_cost_usd` / `total_cost_usd` on the arm and `cost_usd` on every per-case
  row stay **NULL**; the figures land in `*_estimated_usd` with
  `cost_basis='estimated'` and the responsible models named;
- the arm is therefore unrankable on a cost objective, so it cannot win;
- the conclusion carries `cost_pricing_estimated`, whose `more_data` reason says
  the fix is a price-sheet entry, **not** more replay cases;
- `verified_savings_usd` is never computed, because its baseline is NULL.

This mirrors `AttemptFacts.cost_measured` on the Direct Inference surface: one
rule, applied identically wherever a dollar figure is produced.

### Nothing judgeable is ignorance, not efficiency

If not one materiality threshold could be evaluated — because a metric was
missing on an arm — the conclusion is `insufficient_evidence`, never
`no_material_improvement`. "Not worth changing" is a knowledge claim, and the
absence of a comparison is not knowledge.

## The Work Graph

```
Work → Strategy → Executor → Config → Context → Cost → Outcome → Quality →
Human Intervention → Business Value
```

That chain **is** `workloads` + `execution_strategies` + `executors` +
`attempts` + `cost_events` + `outcomes`. It is a **data architecture**: plain
relational tables designed so that *"for workloads like this, which strategy
produced the better economic outcome"* is a natural join. It is not a graph
database, not a visualization, and not the retired knowledge-graph/GraphRAG
concept.

The smoke query is at the bottom of `migration_optimization_work_graph.sql`.

---

## The `node_results` rule

`workflow_runs.node_results` is JSONB with an informal, evolving per-node shape
(`cost` vs `cost_usd`, `tokens` vs `tokens_output`, `status` vs `error` vs
`output_warning`). **All parsing lives in `optimization/attempts.py`.** If it
spreads, the informality becomes debt in a dozen files and the shape can never
change.

Call sites routed through the domain layer:

| Call site | Status |
|---|---|
| `routers/openai_compat.py::_sum_usage_tokens` | **Routed** → `attempts.sum_usage_tokens` |
| `workflow_management.py::_nr_has_error` (8 call sites depend on it) | **Routed** → `attempts.node_result_has_error` |
| `optimization/evidence.py` (all three helpers) | **Routed** → `attempts.parse_step_results` |

Call sites deliberately **not** touched, and why:

| Call site | Why not |
|---|---|
| `workflow_runtime.py` (~43 refs) | These **construct** `node_results` during execution; it is the writer, not a parser. Rewriting the writer during a refactor is how tracing breaks. |
| `workflow_streaming.py` (~18 refs) | Also a constructor, on the SSE path. |
| `synthetic_mind/observer.py` (~15 refs) | Extracts `context_trace` KB-asset ids and agent tool calls — a different projection with its own semantics, owned by another subsystem. Folding it in would couple SM's evolution to ours for no benefit today. |
| `routers/public_execution.py` (~6 refs) | Reads a first ai-step for response shaping and constructs synthetic entries for error paths. Actively being modified by the direct-inference work; routing it now would collide. |
| `sdk/python/optiml/types.py` (1 ref) | A client-side type declaration, not a parser. |

`attempts.parse_step_results` returns `unparseable=True` rather than zeros when
`node_results` is missing or malformed, so a caller can always tell *"measured
zero"* from *"could not read"*.

---

## Interfaces for the Direct Inference surface

`POST /v1/chat/completions` has **no workflow and no deployment**. These are the
functions that path should call rather than reimplementing attribution:

```python
# 1. Identify the work. Never requires the customer to have named anything.
workloads.resolve_workload(
    org_id,
    external_key=...,        # customer-supplied name, e.g. "support-refund" — wins if present
    surface="direct_inference",
    model_target=...,        # structural fallback when unnamed
    create=True,
) -> workload row | None

# 2. Describe the customer's own configuration as a baseline strategy.
strategy.from_direct_inference_request(
    model=..., provider=..., system_prompt=..., workload_id=...,
) -> Strategy            # surface='direct_inference', deployment_id=None

# 3. Parse an execution. THE ONLY node_results parser.
attempts.parse_step_results(node_results) -> AttemptFacts
attempts.sum_usage_tokens(node_results)   -> (prompt, completion, total)
attempts.executors_used(node_results)     -> [executor_ref, ...]
attempts.node_result_has_error(nr)        -> bool

# 4. Fetch an attempt, org-scoped. Handles attempt_source='api_request'
#    (api_request_log.id) and follows workflow_run_id when present.
attempts.get_attempt(org_id, attempt_ref, attempt_source="api_request") -> Attempt | None

# 5. Cost that is not token cost (never write inference cost here — it is
#    already in workflow_runs and would be double-counted).
attempts.record_cost_event(org_id, cost_type=..., amount=..., unit=..., idempotency_key=...)

# 6. Outcomes, arriving whenever they arrive.
outcomes.record_outcome(org_id, outcome_type=..., idempotency_key=..., attempt_ref=..., ...)
outcomes.correct_outcome(org_id, outcome_id, idempotency_key=..., correction_reason=...)

# 7. Persist a strategy, deduped on fingerprint.
service.upsert_strategy(org_id, strategy, workload_id=..., kind="baseline")
```

`optimization/__init__.py` is deliberately import-light: `workflow_runtime`
imports `optimization.evidence`, and `optimization.benchmark` imports
`workflow_runtime`. Eager imports in `__init__` would create a cycle.

---

## Two execution surfaces, one `attempts` abstraction

`public.attempts` UNIONs the two execution records:

| Surface | Execution record | Attempt id | Cost |
|---|---|---|---|
| `runtime` | `workflow_runs` | `workflow_runs.id` | `total_cost`, inside `node_results` per step |
| `direct_inference` | `api_request_log` | `api_request_log.id`, plus `external_attempt_id` = the customer-visible `chatcmpl-…` | `api_request_log.total_cost` (measured only) |

Neither record is duplicated and there is **no second execution table**.
`api_request_log` already carries org, status, latency, measured cost and — in
`custom_metrics` — provider, model, token counts, cost provenance and the
workload identity ref.

**Why a narrow companion table exists anyway.** Exactly two facts cannot be
recovered from the log row, and `direct_inference_attempt_links` carries those
two and nothing else — no cost, latency, token or status column:

1. **The resolved `workloads.id`.** `custom_metrics` holds only a string identity
   *ref*; a join on it is a string match that breaks silently on rename.
2. **The baseline strategy fingerprint.** Direct-inference identity is derived
   partly from the **system prompt**, which `api_request_log` deliberately does
   not store — so the fingerprint is permanently unrecoverable from it.

**The two writers race, and the view tolerates it.** The router writes
`api_request_log`; `optimization_bridge` writes the link. On the streaming path
the bridge runs *first*; on the non-streaming path both are scheduled
concurrently. Neither can wait for the other without adding latency to a
customer's production request. So the view drives from `api_request_log` and
LEFT JOINs the link: if attribution is missing, the attempt still appears with
`workload_id` resolved by the identity-ref fallback and `strategy_id` NULL. **An
execution is never invisible merely because attribution failed** — the bridge
logs `direct_inference.attempt.unattributed` when that happens.

**`step_results` is NULL on the direct branch, by design.** Synthesizing a
`node_results` array in SQL would put knowledge of that contract in a second
place. The branch exposes raw measured columns instead, and
`attempts.facts_from_direct_row` assembles the single step. All shape knowledge
stays in the one module that owns it.

**Estimated cost is never laundered into measured spend.**
`api_request_log.total_cost` is NULL when pricing had to be estimated;
`AttemptFacts.cost_measured` goes False and the estimate lands in
`cost_estimated_usd`. `Attempt.to_dict()["cost_usd"]` returns `None`, and
`total_economic_cost` reports it under `estimated_not_counted`, never in the
total.

### The inference-cost rule, restated

It was previously written as *"inference cost always lives on `workflow_runs`"*.
That assumption died with Direct Inference. The rule was never about
`workflow_runs` — it is:

> Inference cost that the `attempts` view **already surfaces** must not also be
> written as a cost event, on any surface.

This is now **enforced**, not merely documented: `record_cost_event` raises
`DoubleCountedCost` for `cost_type='inference'` with `basis='measured'` on any
attempt source whose execution record carries the cost. The one legitimate
inference cost event is an **estimated** one — where `total_cost` is genuinely
NULL and the cost has no home — recorded with `basis='estimated'`, which forces
`amount_usd` to `None` so a guess can never reach a measured-dollars aggregate.

### Outcome attachment across surfaces

The customer-visible id is `chatcmpl-<24 hex>` — **not a UUID**, so it cannot be
`api_request_log.id`. `outcomes.attempt_ref` stays UUID; a direct-inference id
goes to the new TEXT column `external_attempt_ref`. `attempts.get_attempt`
routes on the *shape* of the id, so a caller who mislabels `attempt_source`
still resolves rather than silently failing.

---

## The 14 architectural tests

**1. Can a Workload exist without an LLM?**
Yes. `workloads` has no model, provider, prompt or token column. Its fields are
`surface`, `identity_kind`, `identity_ref`, `intended_outcome`, `grain`,
`default_objective`. Which executor performs the work is a property of the
*strategy*, not of the workload.

**2. Can a Strategy use multiple executors?**
Yes. `execution_strategies.steps` is an ordered array, each step bound to its
own executor with its own `role` (`primary`, `fallback`, `verifier`, `approver`,
`preprocessor`, `router`). A deterministic classifier → model → human-approval
chain is representable today. `Strategy.unexecutable_steps` reports steps this
runtime cannot execute rather than silently dropping them, so a benchmark can
never claim to have measured a strategy it only partly ran.

**3. Can an Executor be something other than a model?**
Yes. `executors.executor_type ∈ {model, agent, software, human}`. A model is one
kind, not the centre. `strategy.from_graph_json` already emits a
`software` executor for the runtime's `router` node — non-model executors are in
the data today, not just in the CHECK constraint.

**4. Can an Outcome arrive days after execution?**
Yes, and this is the design centre of `outcomes`. `occurred_at` (when it
happened in the world) is separate from `recorded_at` (when we learned), all
analyses window on `occurred_at`, and `POST /v1/outcomes` attaches to an
arbitrarily old attempt. `idempotency_key` is required with a unique index on
`(org_id, idempotency_key)` because outcome feeds are retrying webhooks.

**5. Can multiple outcomes belong to one attempt?**
Yes — first-class, not an edge case. `outcome_type` is an **open vocabulary**,
not an enum. One support response accumulates `thumbs_up`, `ticket_resolved`,
`escalation`, `reopened_7d`. `outcome_value` is `NUMERIC`, not boolean, so a
0.93 resolution rate, a 7.0 CSAT and a `$1,240` figure all fit. Outcomes are
also **correctable**: a revision is a new row linked by `supersedes_outcome_id`,
with the original retained (`is_current=false`), because savings math may have
already consumed the old value.

**6. Can optimization target quality instead of cost?**
Yes. `objective ∈ {cost, quality, latency, balanced, custom}` on the
recommendation, the benchmark and the workload's default.
`benchmark._best_by_objective` ranks on the objective's own metric — and for
`custom` it **refuses** rather than silently minimising cost. Materiality is
objective-aware: `{metric, comparator, value, unit}` OR/AND-combined, so
"≥5% cost reduction OR ≥$1,000/month", "≥150ms p95", and "≥1.5 percentage
points" are all expressible. Savings-shaped keys survive only as sugar.

**7. Can policies constrain strategy selection?**
Yes, and a violation makes a strategy **invalid, not merely worse**.
`policies.evaluate` returns `violated` separately from any ranking, and
`benchmark.evaluate_conclusion` gates on eligibility before it ranks anything.
Two honesty properties: an unmeasured metric can **never** satisfy a constraint
(it becomes `unmeasured`, which fails eligibility), and a constraint OptiML
cannot verify — EU-only residency, zero data retention — is reported as
`unenforced`, never as satisfied.

**8. Can measured production outcomes outweigh LLM-judge signals?**
Yes — and, per the product owner's correction, **not via a hardcoded global
hierarchy.** The deciding signal is declared per workload in
`optimization_policies.success_signal`; for one workload JSON-schema validity
genuinely *is* the hard requirement, for another it is conversion rate. The
provenance ranking (`business_outcome` 80 … `unknown` 10) is the **default** used
only when no policy names a signal. `domain.resolve_success_signal` records
`resolved_from ∈ {policy, default}`, and that resolution is snapshotted onto the
benchmark and the recommendation so a later policy edit cannot rewrite what an
old verdict meant. Signals of different provenance are never averaged:
`group_outcomes_by_provenance` exists to push callers away from doing it, and
`MIN_QUALITY_PROVENANCE_RANK_FOR_CONSTRAINT` puts `llm_judge` below the bar for
satisfying a hard quality floor.

**9. Can an optimization progress replay → canary → production?**
Yes. `evidence_source` encodes **counterfactual strength**, not mere presence:
`none 0 < observational 10 < replay 30 < shadow 40 < ab_test 60 < canary 70 <
production 80`. That number feeds `compute_confidence` alongside sample size,
signal strength and variance — so 14 replay cases score **0.136** and 180,000
confirmed production outcomes score **0.99**. The lifecycle is
`discovered → benchmarking → verified → awaiting_approval → shadowing → canary →
promoted`, with `rejected / inconclusive / failed / superseded / rolled_back`.
There is deliberately **no edge** from `verified` straight to `canary`: human
approval is the default and production is never changed autonomously unless an
org opts into an `automation` flag.

**10. Can realized impact be attributed without double-counting?**
Yes. Three separate columns with three separate meanings, enforced by
`domain.savings_column()` which raises rather than let a projection be written
into a verified column:
- `projected_savings_usd` — extrapolated (measured delta × measured volume)
- `verified_savings_usd` — measured in a benchmark or canary, over the sample
- `realized_savings_usd` — observed in production after promotion

`domain.attributable_savings` walks `parent_recommendation_id` /
`baseline_reference.derived_from_recommendation_id` and counts only the deepest
surviving descendant of a chain, because an ancestor's savings are already
embedded in the current baseline. Within a `bundle_id` only the widest candidate
counts, so a bundle and its parts are never both claimed. It returns `None`, not
`0.0`, when nothing is measurable, plus a `coverage` block naming every
exclusion.

**11. Can a future Claude Code / Devin connector use the same engine?**
Yes. It registers as `executors(executor_type='agent', integration_source='connector')`
with its own `cost_model` (`usd_per_acu`, `usd_per_credit`). Its work becomes a
`workload` (`surface='workforce'`), its configuration a `strategy` step, its
runs `attempts`, its spend `cost_events`, its results `outcomes` (`pr_merged`,
`pr_reverted`). `evaluate_conclusion` is a pure function over measured arms and
a policy — nothing in it is model-shaped. What is missing is an *execution*
adapter for that surface, not a schema change.

**12. Could a future allocation engine select among agents without replacing the schema?**
Yes. `allocation_decisions` already stores `considered_strategies` (including
rejects and *why* each was rejected), `selected_strategy_id`, the objective and
`objective_config`, the policy, the expected metrics, the confidence, and an
`actual_result` backfilled later for expected-vs-actual calibration. It is
written today for benchmark and recommendation decisions and is not surfaced
prominently. An allocation engine reads and writes exactly this table.

**13. Could deterministic software compete against an AI strategy?**
Yes at the modelling layer, not yet at the execution layer — stated plainly
rather than implied. `executor_type='software'` is registrable, is already
emitted for router nodes, and a strategy may mix software and model steps.
`apply_to_graph` **refuses** a software step it cannot execute rather than
dropping it. The gap is an executable software-step type in `workflow_runtime`;
that is the whole remaining work, and `CallCountReductionGenerator` documents it.

**14. Could human execution eventually be represented without a schema rewrite?**
Yes. `executor_type='human'` is permitted; `cost_events.cost_type='human_time'`
with `unit='minutes'` captures the cost; `outcomes` with `provenance='human'` and
`outcome_type='human_intervention'` captures the result; a strategy step with
`role='approver'` captures the position in the chain. **Nothing is built for it** —
no routing, queueing, assignment or payout — and `domain.UNBUILT_EXECUTOR_TYPES`
says so in code so nobody mistakes the schema's permission for a feature.

---

## Benchmark conclusions: knowledge vs ignorance

A benchmark records an **explicit** conclusion. It is never inferred from
absence.

| Conclusion | Meaning | Epistemic class | → recommendation status |
|---|---|---|---|
| `safe_improvement_found` | Satisfies policy **and** materially improves the objective | knowledge | `verified` |
| `no_material_improvement` | Satisfies policy, but not worth changing | knowledge | `rejected` |
| `candidates_failed_policy` | Tested with sufficient evidence, violated a constraint | knowledge | `rejected` |
| `insufficient_evidence` | Cannot conclude either way | **ignorance** | `inconclusive` |
| `benchmark_failed` | Technical failure prevented a valid comparison | **ignorance** | `failed` |

`no_material_improvement → rejected` rather than `inconclusive` because reaching
it requires that the sample cleared the floor, the candidates satisfied policy,
and the improvement was genuinely below threshold. We *know* the answer.
`inconclusive` is reserved strictly for not being able to tell.

**The hard rule.** `insufficient_evidence` is never evidence that the current
configuration is optimal. `domain.is_efficiency_finding()` returns true for
exactly one conclusion, and the summary endpoint reports ignorance under
`not_yet_assessable` — never under `no_opportunity`.

The API returns **codes and facts, never prose**:

```json
{ "conclusion": "candidates_failed_policy",
  "reasons": [{ "code": "quality_below_threshold", "constraint": "min_quality",
                "observed": 0.933, "required": 0.94, "unit": "score",
                "shortfall": 0.007, "candidate": "Switch to claude-3-haiku" }],
  "confidence": 0.41, "confidence_band": "medium" }
```

The vocabulary is `domain.REASON_CODES`; `domain.reason()` raises on an
undocumented code so one cannot slip into the contract unannounced. All wording
is derived by the frontend, so rephrasing a sentence is never an API change.

### Conclusions are recomputable, and history is immutable

`evidence + policy version + objective → conclusion` is a **pure function**:
`benchmark.evaluate_conclusion`. Every measured arm is persisted in
`benchmark_candidate_results` independently of the verdict, so a candidate that
saved 51% but landed 0.7pp under the quality floor is retained as a first-class
result. When the customer relaxes `min_quality` to 0.94, `benchmark.reevaluate`
re-derives the verdict over stored rows with **zero model calls** and inserts a
new `benchmark_conclusions` row bound to the new policy version. The original is
retained. It was not wrong — it was correct under the policy in force at the
time. Both coexist.

Policies are therefore versioned: `policies.update_policy` **inserts a new
version** and flips `is_current` on the old one. It never edits constraint values
in place.

---

## Optimization Coverage

> The percentage of eligible workload spend/volume for which OptiML has
> sufficient evidence to make an optimization determination.

One classifier — `domain.coverage_class()` — not scattered predicates.

- **Covered:** `safe_improvement_found`, `no_material_improvement`, `candidates_failed_policy`
- **Not covered:** `insufficient_evidence`, `benchmark_failed`, in-progress, never run

Three figures are computed: **workload**, **spend** and **volume**.
`compute_coverage` marks **spend coverage as primary** when the objective is
cost, because *"8 of 10 workloads assessed"* reads as excellent right up until
the two unassessed ones are 78% of spend.

```
coverage 64% — $81k/month assessable, $46k/month awaiting sufficient evidence
```

---

## Recommendation engine staging

| Stage | Status |
|---|---|
| 1 · rules / obvious opportunities | **Built** — `AlternateModelGenerator` (vendor price sheet) |
| 2 · replay evidence | **Built** — `CheaperMeasuredModelGenerator` + `benchmark.py` |
| 3 · production experiment | **Wired** to the existing canary/experiment infrastructure; no new machinery |
| 4 · historical workload learning | Extension point |
| 5 · prediction | Extension point |
| 6 · allocation optimization | Extension point — `allocation_decisions` is the substrate |

---

## What is real vs. what is a documented extension point

### Real
- Four idempotent migrations; RLS + `is_org_member` SELECT policy on every new table.
- Workload identity: **explicit** (`external_key`) and **structural** (endpoint /
  workflow / template / model target); discovery from production `workflow_runs`.
- Executor registry synced from `shared/providers.json`, labelled `vendor_metadata`.
- Strategy ↔ `graph_json` adapter for the dimensions this runtime can genuinely
  apply: **model, provider, prompt, context_length, fallback_chain**.
- Two candidate generators, each backed by real data — one over the vendor
  price sheet (`evidence_source='none'`), one over measured history
  (`evidence_source='observational'`). Neither may be recommended unmeasured.
- Read-only workload selection with documented floors and structured skip codes.
- Cost provenance: an estimated price never reaches a measured-cost field, on any
  row, at any layer.
- Recommendation creation gated on `safe_improvement_found` alone, cited through
  `recommendation_evidence`, `approval_required` per policy (default TRUE), with
  `projected_savings_usd` (measured delta x measured volume) and
  `verified_savings_usd` (measured over the sample) written to their own columns
  and `baseline_reference.derived_from_recommendation_id` set so a chain is not
  counted twice.
- Golden-input replay benchmark reusing `execute_workflow(execution_mode='eval')`
  and the workflow's own `eval_suites.checks`.
- Explicit conclusions, structured reason codes, immutable policy-versioned
  conclusion history, independently queryable candidate results, `reevaluate`.
- Policies: enforceable constraint evaluation, versioning by insert,
  objective-aware materiality, per-workload success signal.
- Outcomes: delayed, plural, named, correctable, idempotent — both routes.
- Lifecycle state machine with audit trail; `accept` creates a real candidate
  deployment reusing `workflow_deployments`.
- Summary with real numbers, three coverage figures, and explicit nulls.

### Documented extension points (each refuses rather than fabricates)
- **`temperature` / `max_tokens` / `top_p`** — applicability is **per surface**,
  not global. On **runtime** they are *not* applicable: `_execute_model_node`
  sends only `(org_id, provider, model, prompt, prompt_id)` and
  `anthropic_router` hardcodes `max_tokens=1024`, so `apply_to_graph` raises
  `UnsupportedDimension`. On **direct_inference** they are **real and
  benchmarkable** — the OpenAI dialect forwards the body and the Anthropic
  translation maps them explicitly — so `from_direct_inference_request` carries
  them as genuine strategy config and a generator may propose changing them.
- **`caching`, `reasoning_effort`, `retrieval`, `reranking`** — no mechanism
  exists in `workflow_runtime` / `context_runtime`. Listed in
  `UNSUPPORTED_DIMENSIONS` with the reason.
- **`PromptCompressionGenerator`** — needs a per-component token breakdown and a
  quality signal able to detect tail degradation. Raises.
- **`CallCountReductionGenerator`** — needs per-step outcome attribution and an
  executable software executor. Raises.
- **`discover_learned_workloads`** — needs input embeddings, a stability
  criterion, a merge/split protocol and human confirmation. Raises.
- **Realized post-promotion monitoring** — `realized_savings_usd`,
  `realized_metrics` and `monitoring_status='degraded'` exist and are never
  written. The summary returns `realized_usd: null` with the coverage note
  `realized_savings_uninstrumented` — null, never zero.
- **Shadow / A/B / canary evidence collection** — `evidence_source` values and
  the `shadowing` lifecycle state exist; only `replay` is collected today.
- **Human, agent and software execution** — registrable, not executable.
- **Direct Inference execution** — the surface's request path is owned by
  `direct_inference.py` / `routers/openai_compat.py`. The optimization layer's
  side is complete: attempts, workload attribution, baseline strategies and
  outcome attachment all work (see below).
