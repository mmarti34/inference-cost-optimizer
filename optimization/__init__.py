"""
OptiML optimization layer.

The loop: Observe -> Benchmark -> Optimize -> Verify -> Canary -> Monitor ->
Promote/Rollback.  The deliverable is an evidence-backed Optimization
Recommendation.  A candidate is NEVER promoted because an LLM said it seemed
good: evidence must come from measurable replay/benchmark against the
customer's own data.

Module layout enforces one hard boundary — VENDOR METADATA is never mixed with
EMPIRICAL EVIDENCE:

    domain.py       Shared vocabulary: objectives, lifecycle state machine,
                    evidence strength, outcome provenance ranking, confidence.
                    Pure functions, no I/O.
    executors.py    VENDOR side. Published prices, advertised capabilities,
                    declared regions. Vendor claims are NEVER evidence.
    evidence.py     EMPIRICAL side. What was actually measured on THIS org's
                    workloads (workflow_runs, outcomes). Never vendor data.
    strategy.py     Execution strategies: ordered executor steps, and the
                    Runtime adapter to/from workflow graph_json.
    candidates.py   Candidate generators (recommendation engine stages 1-2).
    benchmark.py    The replay evidence engine.
    noninferiority.py
                    Paired non-inferiority statistics over the per-case
                    pass/fail data both arms produce. Answers "can we RULE OUT
                    a material quality regression against the customer's
                    measured baseline", which an absolute quality floor and a
                    point estimate both fail to answer. Pure, no I/O.
    outcomes.py     Outcome recording: delayed arrival, idempotent, ranked.
    attempts.py     Thin domain adapter over the EXISTING tracing tables.
    policies.py     Constraints that make a strategy invalid, not merely worse.
    allocation.py   Records which strategy was chosen and why.
    service.py      Recommendation CRUD + lifecycle transitions + audit trail.
    workloads.py    Workload discovery.

This package's __init__ is deliberately import-light: workflow_runtime imports
optimization.evidence, and optimization.benchmark imports workflow_runtime.
Eager imports here would create a cycle.
"""

__all__ = [
    "domain",
    "executors",
    "evidence",
    "strategy",
    "candidates",
    "benchmark",
    "noninferiority",
    "outcomes",
    "attempts",
    "policies",
    "allocation",
    "service",
    "workloads",
]
