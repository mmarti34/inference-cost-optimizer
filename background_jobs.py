"""
Background control loop: the timer that makes safety automation actually fire.

The rollback monitor and the experiment auto-conclude check were both complete
and correct, but nothing ever called them on a schedule:

  * run_rollback_monitor_cycle() had exactly one caller, an HTTP endpoint gated
    by require_org_member — a *user* JWT, so no machine could reach it. The UI
    literally told operators to wire up their own cron.
  * _maybe_auto_conclude() only ran from GET /experiments/{id}, i.e. only when
    a human opened the page.

This module registers an in-process asyncio task that runs both on an interval,
plus a service-key/cron-secret HTTP path for external schedulers.

Environment:
  OPTIML_CONTROL_LOOP_ENABLED   "true" (default) | "false" to opt out
  OPTIML_CONTROL_LOOP_INTERVAL_SECONDS   seconds between cycles, default 60
  OPTIML_CONTROL_LOOP_START_DELAY_SECONDS  delay before the first cycle, default 15
  OPTIML_CRON_SECRET            shared secret for the org-wide external cron path

Guarantees:
  * Only one cycle runs at a time (an asyncio.Lock shared with the HTTP path);
    a slow cycle skips the next tick instead of overlapping.
  * Every failure is caught. A crash in one org's rule, or in the whole cycle,
    logs and the loop keeps ticking.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from auth_dependency import require_auth, AuthenticatedUser

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_START_DELAY_SECONDS = 15
MIN_INTERVAL_SECONDS = 5

# asyncio primitives are created inside the running loop, never at import time.
# Under uvicorn this module is imported before the event loop exists; an
# asyncio.Lock/Event built then can bind to the wrong loop, and the symptom is
# the loop task dying silently after one tick — precisely the class of failure
# this module exists to prevent.
_cycle_lock: Optional[asyncio.Lock] = None
_stopping: Optional[asyncio.Event] = None
_task: Optional[asyncio.Task] = None


def _get_cycle_lock() -> asyncio.Lock:
    global _cycle_lock
    if _cycle_lock is None:
        _cycle_lock = asyncio.Lock()
    return _cycle_lock


def _get_stop_event() -> asyncio.Event:
    global _stopping
    if _stopping is None:
        _stopping = asyncio.Event()
    return _stopping


async def _sleep_or_stop(seconds: float) -> bool:
    """Sleep, returning True if a shutdown was requested before the time elapsed."""
    if seconds <= 0:
        return _get_stop_event().is_set()
    try:
        await asyncio.wait_for(_get_stop_event().wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False

_state = {
    "enabled": False,
    "interval_seconds": DEFAULT_INTERVAL_SECONDS,
    "cycles": 0,
    "skipped_overlaps": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "last_result": None,
    "last_error": None,
}

router = APIRouter()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def run_benchmark_job_reaper(org_id: Optional[str] = None) -> dict:
    """
    Declare orphaned benchmark jobs failed. Never raises.

    A benchmark job renews `heartbeat_at` while it is alive. A process that dies
    renews nothing, so a job still in an ACTIVE status whose lease has expired
    is one whose worker is gone — and the only thing that must never happen is
    for it to sit at `running` forever, because that is a state nothing in the
    system can ever resolve. This is the thing that resolves it.

    Runs blocking Supabase calls, so it goes to a thread rather than stalling
    the loop that also drives rollback monitoring.
    """
    from optimization import jobs as jobs_mod

    return await asyncio.to_thread(jobs_mod.reap_stale_jobs, org_id)


async def run_control_loop_cycle(org_id: Optional[str] = None) -> dict:
    """
    Run one control-loop cycle: rollback rules, experiment auto-conclude, then
    orphaned benchmark jobs.

    org_id scopes the cycle to a single organization. Never raises — a failure
    in one stage is reported in the result and the other stages still run.
    """
    # Imported lazily so this module can be imported before the app's routers.
    from workflow_management import (
        run_experiment_auto_conclude_cycle,
        run_rollback_monitor_cycle,
    )

    result: dict = {"org_id": org_id, "started_at": _now()}
    try:
        result["rollback"] = await run_rollback_monitor_cycle(org_id=org_id)
    except Exception as e:
        logger.exception("Rollback monitor cycle failed")
        result["rollback"] = {"error": str(e)[:300]}
    try:
        result["experiments"] = await run_experiment_auto_conclude_cycle(org_id=org_id)
    except Exception as e:
        logger.exception("Experiment auto-conclude cycle failed")
        result["experiments"] = {"error": str(e)[:300]}
    try:
        result["benchmark_jobs"] = await run_benchmark_job_reaper(org_id=org_id)
    except Exception as e:
        logger.exception("Benchmark job reaper failed")
        result["benchmark_jobs"] = {"error": str(e)[:300]}
    result["finished_at"] = _now()
    return result


async def run_control_loop_cycle_guarded(org_id: Optional[str] = None) -> dict:
    """run_control_loop_cycle() with the overlap guard applied."""
    lock = _get_cycle_lock()
    if lock.locked():
        _state["skipped_overlaps"] += 1
        logger.warning(
            "Control loop cycle already running; skipping this tick (skipped=%d).",
            _state["skipped_overlaps"],
        )
        return {"skipped": True, "reason": "cycle_already_running"}
    async with lock:
        _state["last_started_at"] = _now()
        try:
            result = await run_control_loop_cycle(org_id=org_id)
            _state["last_result"] = result
            _state["last_error"] = None
            return result
        except Exception as e:  # defensive: run_control_loop_cycle already catches
            logger.exception("Control loop cycle raised")
            _state["last_error"] = str(e)[:300]
            return {"error": str(e)[:300]}
        finally:
            _state["cycles"] += 1
            _state["last_finished_at"] = _now()


async def _loop(interval_seconds: int, start_delay: int) -> None:
    logger.info(
        "Control loop starting: interval=%ds, first cycle in %ds "
        "(rollback rules + experiment auto-conclude).",
        interval_seconds, start_delay,
    )
    stop = _get_stop_event()
    if await _sleep_or_stop(start_delay):
        return  # shutdown before the first cycle
    while not stop.is_set():
        try:
            await run_control_loop_cycle_guarded()
        except asyncio.CancelledError:
            raise
        except BaseException:
            # Nothing below run_control_loop_cycle_guarded should raise, but if
            # anything does the loop must survive it. A dead loop means rollbacks
            # silently stop firing, which is the bug this module fixes.
            logger.exception("Control loop tick failed; continuing.")
        if await _sleep_or_stop(interval_seconds):
            break
    logger.info("Control loop stopped.")


def _on_loop_done(task: "asyncio.Task") -> None:
    """Make a dead scheduler loud instead of silent."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.critical(
            "Control loop task exited with an exception; automatic rollback and "
            "experiment auto-conclude are NO LONGER running in this process.",
            exc_info=exc,
        )
    elif _stopping is not None and not _stopping.is_set():
        logger.critical(
            "Control loop task exited unexpectedly; automatic rollback and "
            "experiment auto-conclude are NO LONGER running in this process."
        )


def _authorize_machine_cycle(cron_secret: Optional[str], authorization: Optional[str]) -> Optional[str]:
    """
    Authorize a machine-triggered cycle without a user JWT.

    Returns the org_id the cycle is scoped to, or None for an org-wide cycle.
    Two accepted credentials, in order:
      1. X-OptiML-Cron-Secret matching OPTIML_CRON_SECRET -> org-wide.
      2. Authorization: Bearer <service api key> -> scoped to that key's org.
    Raises 401 otherwise. This adds a machine path; it does not widen what a
    user token can do, and a per-org service key can only act on its own org.
    """
    configured = (os.getenv("OPTIML_CRON_SECRET") or "").strip()
    if configured and cron_secret and hmac.compare_digest(cron_secret.strip(), configured):
        return None

    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        if token:
            from api_key_validation import validate_api_key
            ctx = validate_api_key(token)  # raises 401 when invalid
            return ctx.org_id

    raise HTTPException(
        status_code=401,
        detail=(
            "Control loop requires X-OptiML-Cron-Secret (org-wide) or "
            "Authorization: Bearer <service_api_key> (scoped to that key's org)."
        ),
    )


def _is_operator(cron_secret: Optional[str]) -> bool:
    """
    True only for a caller presenting the configured OPTIML_CRON_SECRET.

    Deliberately no default: an unset secret means "no operator callers", not
    "everyone is an operator". Constant-time compare.
    """
    configured = (os.getenv("OPTIML_CRON_SECRET") or "").strip()
    if not configured or not cron_secret:
        return False
    return hmac.compare_digest(cron_secret.strip(), configured)


@router.post("/control-loop/run")
async def trigger_control_loop(
    x_optiml_cron_secret: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Run one control-loop cycle. Machine-callable: no user JWT required.

    Use this only if the in-process scheduler is disabled
    (OPTIML_CONTROL_LOOP_ENABLED=false) or you want a second, external trigger.
    """
    org_id = _authorize_machine_cycle(x_optiml_cron_secret, authorization)
    return await run_control_loop_cycle_guarded(org_id=org_id)


@router.get("/control-loop/status")
async def control_loop_status(
    x_optiml_cron_secret: Optional[str] = Header(None),
    _user: AuthenticatedUser = Depends(require_auth),
):
    """
    Whether the scheduler is running, and what the last cycle did.

    Requires authentication (it was anonymous). Signed-in callers get liveness
    only. `last_error` is a raw `str(e)[:300]` from the cycle and `last_result`
    is a cross-tenant summary of what the loop did across every org, so both are
    returned only to an operator presenting OPTIML_CRON_SECRET. Full detail is
    always in the server logs.
    """
    payload = {
        "enabled": _state["enabled"],
        "running": bool(_task and not _task.done()),
        "interval_seconds": _state["interval_seconds"],
        "cycles": _state["cycles"],
        "skipped_overlaps": _state["skipped_overlaps"],
        "last_started_at": _state["last_started_at"],
        "last_finished_at": _state["last_finished_at"],
        "last_cycle_failed": _state["last_error"] is not None,
    }
    from optimization import jobs as jobs_mod
    # Liveness of the async-benchmark machinery. Never org data: a count of
    # tasks in THIS process and whether the v9 columns exist. `degraded_reason`
    # is a raw exception string and stays operator-only, like `last_error`.
    schema = jobs_mod.schema_status()
    payload["benchmark_jobs"] = {
        "in_process_tasks": jobs_mod.active_task_count(),
        "job_columns_available": schema["job_columns_available"],
        "heartbeat_seconds": schema["heartbeat_seconds"],
        "lease_seconds": schema["lease_seconds"],
    }
    if _is_operator(x_optiml_cron_secret):
        payload["last_result"] = _state["last_result"]
        payload["last_error"] = _state["last_error"]
        payload["benchmark_jobs"]["degraded_reason"] = schema["degraded_reason"]
        payload["benchmark_jobs"]["worker_id"] = schema["worker_id"]
    return payload


@router.get("/observability/pricing-misses")
async def pricing_misses(
    x_optiml_cron_secret: Optional[str] = Header(None),
):
    """
    Models seen at runtime that have no price in shared/providers.json.

    Every one of these is being costed with an estimated default, so any savings
    number that includes them is a guess. Non-empty means: add the model.

    Operator-only (OPTIML_CRON_SECRET). This counter is process-global and has
    no org dimension at all — it reports which models are being run across every
    tenant — so no org-scoped guard can be correct here: require_org_admin would
    prove the caller administers org A while handing back data about orgs B..Z.
    It is an operator diagnostic, not a customer-facing endpoint, and the same
    information is emitted to the logs as `PRICING MISS:` warnings.
    """
    if not _is_operator(x_optiml_cron_secret):
        raise HTTPException(
            status_code=403,
            detail="Operator credential required (X-OptiML-Cron-Secret).",
        )
    from utils.pricing import get_pricing_miss_stats
    return get_pricing_miss_stats()


def register_background_jobs(app, prefix: str = "/api") -> None:
    """Attach the control-loop routes and the scheduler task to a FastAPI app."""
    app.include_router(router, prefix=prefix)

    enabled = _env_flag("OPTIML_CONTROL_LOOP_ENABLED", True)
    interval = _env_int("OPTIML_CONTROL_LOOP_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS)
    start_delay = _env_int("OPTIML_CONTROL_LOOP_START_DELAY_SECONDS", DEFAULT_START_DELAY_SECONDS, 0)
    _state["enabled"] = enabled
    _state["interval_seconds"] = interval

    @app.on_event("startup")
    async def _reap_orphaned_benchmarks_on_boot() -> None:
        """
        Resolve jobs orphaned by the process that just died, immediately.

        On a single-worker deployment the process that lost a benchmark IS the
        one coming back, and nothing else is watching. Waiting for the first
        control-loop tick would leave the dashboard showing a run that has been
        dead since before the restart. This runs once, costs one query, and
        cannot mark a live job failed: a job started by THIS process has not
        been created yet, and any job whose lease has not expired is left alone.
        """
        try:
            result = await run_benchmark_job_reaper()
            if result.get("failed"):
                logger.warning(
                    "Startup reap: %d benchmark job(s) orphaned by a previous "
                    "process were marked failed (worker_lost): %s",
                    result["failed"], result.get("benchmark_ids"),
                )
        except Exception:
            logger.exception("Startup benchmark-job reap failed; the loop will retry.")

    @app.on_event("startup")
    async def _start_control_loop() -> None:
        global _task
        if not enabled:
            logger.warning(
                "Control loop disabled (OPTIML_CONTROL_LOOP_ENABLED=false). Automatic "
                "rollback and experiment auto-conclude will NOT fire unless an external "
                "cron calls POST /control-loop/run."
            )
            return
        if _task and not _task.done():
            return
        _get_stop_event().clear()
        _task = asyncio.create_task(_loop(interval, start_delay))
        _task.add_done_callback(_on_loop_done)

    @app.on_event("shutdown")
    async def _note_abandoned_benchmark_jobs() -> None:
        """
        Say what this process is walking away from.

        In-flight benchmark jobs are NOT marked failed here. This handler runs
        while the run may still be finishing in a worker thread, and a process
        that is going away must not guess how much of a 28-minute run it
        completed — killing a job that then writes a real conclusion a second
        later would destroy evidence that was already paid for. The heartbeat
        stops with the loop, so the job falls out of its lease and is reaped by
        whichever process is alive next, including this one on restart.
        """
        from optimization import jobs as jobs_mod

        outstanding = jobs_mod.active_task_count()
        if outstanding:
            logger.warning(
                "Shutting down with %d benchmark job(s) still in flight. They are "
                "left ACTIVE deliberately; their lease will expire and the "
                "startup/interval reaper will mark them worker_lost. Measured "
                "arms are already durable in benchmark_candidate_results.",
                outstanding,
            )

    @app.on_event("shutdown")
    async def _stop_control_loop() -> None:
        global _task
        _get_stop_event().set()
        if _task and not _task.done():
            _task.cancel()
            try:
                await _task
            except (asyncio.CancelledError, Exception):
                pass
        _task = None
