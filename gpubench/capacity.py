"""Max concurrent users: SLO-bound (measured) and KV-cache-bound (from vLLM's log)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from gpubench.config import SLO, CapacitySearch, Workload
from gpubench.schema import CapacityResult, PerfPoint, RunResult

log = logging.getLogger("gpubench")


def kv_cache_max_users(kv_cache_tokens: int | None, workload: Workload) -> int | None:
    """Requests of ISL+OSL tokens that fit in the KV cache at once."""
    per_request = workload.tokens_per_request
    if not kv_cache_tokens or not per_request:
        return None
    return kv_cache_tokens // per_request


def find_max_users(
    sweep: dict[int, PerfPoint],
    measure: Callable[[int], PerfPoint],
    search: CapacitySearch,
    ceiling: int | None = None,
) -> tuple[int, dict[int, PerfPoint]]:
    """Highest concurrency that passes the SLO.

    Starts from the coarse sweep. If even the top level passes, keeps doubling (up to
    `ceiling`, e.g. 2x the KV-cache limit, beyond which requests only queue) until a level
    fails. Then bisects between the highest passing level and the first failing one.
    Assumes latency grows with concurrency, which holds for one deployment under
    closed-loop load. Returns (max_users, all measured points including new probes).
    """
    points = dict(sweep)
    levels = sorted(points)
    first_fail = next((c for c in levels if not points[c].slo_pass), None)
    passing = [c for c in levels if points[c].slo_pass and (first_fail is None or c < first_fail)]
    lo = passing[-1] if passing else 0  # 0 = not even the smallest level passed

    if not search.enabled:
        return lo, points

    def probe(c: int) -> PerfPoint | None:
        # A failed probe (e.g. the load generator hitting an OS limit) ends the search; the
        # levels already measured still give a valid (lower-bound) answer.
        try:
            return measure(c)
        except Exception:
            log.exception("capacity probe at %d users failed; stopping search", c)
            return None

    ceiling = ceiling or search.max_users
    for _ in range(search.max_iterations):
        if first_fail is not None or lo <= 0 or lo >= ceiling:
            break
        nxt = min(lo * 2, ceiling)
        point = probe(nxt)
        if point is None:
            return lo, points
        points[nxt] = point
        if point.slo_pass:
            lo = nxt
        else:
            first_fail = nxt

    if first_fail is None:
        return lo, points  # passed up to the ceiling; the site shows this as ">= lo"

    hi = first_fail
    for _ in range(search.max_iterations):
        if hi - lo <= search.resolution:
            break
        mid = (lo + hi) // 2
        if mid <= 0 or mid in points:
            break
        point = probe(mid)
        if point is None:
            break
        points[mid] = point
        if points[mid].slo_pass:
            lo = mid
        else:
            hi = mid
    return lo, points


def capacity_result(
    workload: Workload,
    slo: SLO,
    max_users: int,
    points: dict[int, PerfPoint],
    kv_cache_tokens: int | None,
) -> CapacityResult:
    at_max = points.get(max_users)
    return CapacityResult(
        workload=workload.name,
        slo=slo,
        max_users_slo=max_users,
        max_users_kv_cache=kv_cache_max_users(kv_cache_tokens, workload),
        output_throughput_at_max_users=at_max.output_throughput if at_max else None,
        probed={c: p.slo_pass for c, p in sorted(points.items())},
    )


def fill_missing_capacity(run: RunResult) -> RunResult:
    """Derive capacity from the sweep for workloads that have perf points but no capacity.

    Happens when the capacity step crashed after the sweep finished (e.g. a probe past the
    sweep hit the load generator's fd limit before probe failures were caught). The sweep's
    per-point SLO verdicts still give a valid lower bound; no new probes are possible here.
    """
    have = {c.workload for c in run.capacity}
    slo = run.capacity[0].slo if run.capacity else SLO()
    added = []
    for name in dict.fromkeys(p.workload for p in run.perf):
        if name in have:
            continue
        pts = {p.concurrency: p for p in run.perf if p.workload == name}
        first = next(iter(pts.values()))
        workload = Workload(name=name, input_len=first.input_len, output_len=first.output_len)
        # Search disabled: the sweep's verdicts are all there is, so `measure` is never called.
        max_users, _ = find_max_users(pts, pts.__getitem__, CapacitySearch(enabled=False))
        added.append(capacity_result(workload, slo, max_users, pts, run.kv_cache_tokens))
        log.warning("%s: capacity for %s derived from the sweep (%d users)",
                    run.run_id, name, max_users)
    # Appended: existing entries keep config order (perf is stored sorted by name, so it
    # can't tell us where the lost workload belonged).
    return run.model_copy(update={"capacity": [*run.capacity, *added]}) if added else run
