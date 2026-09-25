"""Self-hosting vs API cost, from measured capacity and dated price tables.

Performance depends on the hardware, not on who rents it, so one benchmark on an H100 prices
every provider renting that H100. For each (run, workload) we take the throughput at the most
users the deployment serves within the latency SLO, and turn each GPU offer's hourly price into
$ per 1M tokens and per 1k requests. Each API provider's price for the same model gives the
per-request cost of the alternative; the break-even is the daily volume where an always-on GPU
costs the same as paying per token.
"""

from __future__ import annotations

import statistics
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from gpubench.schema import PerfPoint, RunResult

HOURS_PER_MONTH = 730


class GpuOffer(BaseModel):
    provider: str
    gpu_type: str  # normalised, same names as results (gpubench.hardware)
    provisioning: str = "on-demand"
    usd_per_gpu_hour: float
    min_gpus: int = 1
    product: str | None = None  # the provider's name for it
    source: str
    checked_at: date
    notes: str | None = None


class ApiOffer(BaseModel):
    model: str  # Hugging Face id
    provider: str
    usd_per_1m_input: float
    usd_per_1m_output: float
    quantization: str | None = None
    context_length: int | None = None
    source: str
    fetched_at: date
    notes: str | None = None


def load_gpu_offers(path: Path) -> list[GpuOffer]:
    data = yaml.safe_load(path.read_text()) or {}
    return [GpuOffer.model_validate(o) for o in data.get("offers", [])]


def load_api_offers(path: Path) -> list[ApiOffer]:
    data = yaml.safe_load(path.read_text()) or {}
    return [ApiOffer.model_validate(o) for o in data.get("offers", [])]


def api_usd_per_request(offer: ApiOffer, input_len: int, output_len: int) -> float:
    return (input_len * offer.usd_per_1m_input + output_len * offer.usd_per_1m_output) / 1e6


def point_at_capacity(run: RunResult, workload: str, max_users: int) -> PerfPoint | None:
    """The measured point at the SLO capacity (else the fastest point that met the SLO)."""
    pts = [p for p in run.perf if p.workload == workload]
    exact = [p for p in pts if p.concurrency == max_users]
    if exact:
        return exact[0]
    passing = [p for p in pts if p.slo_pass]
    return max(passing, key=lambda p: p.output_throughput, default=None)


def hosted_prices(usd_per_hour: float, point: PerfPoint) -> dict:
    per_s = usd_per_hour / 3600
    total_tps = point.total_token_throughput
    return {
        "usd_per_hour": round(usd_per_hour, 4),
        "usd_per_month": round(usd_per_hour * HOURS_PER_MONTH, 2),
        "usd_per_1m_output_tokens": per_s / point.output_throughput * 1e6,
        "usd_per_1m_total_tokens": per_s / total_tps * 1e6 if total_tps else None,
        "usd_per_1k_requests": per_s / point.request_throughput * 1e3,
    }


def deployments(runs: list[RunResult], gpu_offers: list[GpuOffer]) -> list[dict]:
    """One row per (run, workload) with its SLO capacity and every matching GPU offer priced."""
    rows = []
    for run in runs:
        hw = run.hardware
        gpus = hw.gpu_count * hw.node_count
        accuracy = {a.task: a.value for a in run.accuracy}
        for cap in run.capacity:
            point = point_at_capacity(run, cap.workload, cap.max_users_slo)
            if point is None or cap.max_users_slo <= 0:
                continue
            offers = []
            for o in gpu_offers:
                if o.gpu_type == hw.gpu_type and gpus >= o.min_gpus:
                    offers.append({
                        "provider": o.provider, "provisioning": o.provisioning,
                        "product": o.product, "source": o.source,
                        "checked_at": o.checked_at.isoformat(),
                        **hosted_prices(o.usd_per_gpu_hour * gpus, point),
                    })
            listed = {(o["provider"], o["provisioning"], o["usd_per_hour"]) for o in offers}
            if hw.price_per_hour_usd and (hw.provider, hw.provisioning,
                                          round(hw.price_per_hour_usd, 4)) not in listed:
                # The price the run was actually measured at, whole machine.
                offers.append({
                    "provider": hw.provider, "provisioning": hw.provisioning,
                    "product": hw.machine_type, "source": f"run {run.run_id}",
                    "checked_at": run.created_at.date().isoformat(), "measured": True,
                    **hosted_prices(hw.price_per_hour_usd, point),
                })
            offers.sort(key=lambda o: o["usd_per_hour"])
            rows.append({
                "run_id": run.run_id,
                "model": run.model.hf_id,
                "precision": run.model.precision,
                "gpu_type": hw.gpu_type,
                "gpu_count": gpus,
                "engine_version": run.software.engine_version,
                "workload": cap.workload,
                "input_len": point.input_len,
                "output_len": point.output_len,
                "slo": cap.slo.model_dump(),
                "max_users_slo": cap.max_users_slo,
                "max_users_kv_cache": cap.max_users_kv_cache,
                "capacity_is_lower_bound": _is_lower_bound(cap.probed, cap.max_users_slo),
                "requests_per_s": point.request_throughput,
                "output_tokens_per_s": point.output_throughput,
                "total_tokens_per_s": point.total_token_throughput,
                "ttft_p99_ms": point.ttft_ms.p99,
                "itl_median_ms": point.itl_ms.median,
                "accuracy": accuracy,
                "offers": offers,
            })
    return rows


def _is_lower_bound(probed: dict[int, bool], max_users: int) -> bool:
    """True when nothing above max_users was measured failing (the sweep hit its ceiling)."""
    return not any(c > max_users and not ok for c, ok in probed.items())


def api_rows(api_offers: list[ApiOffer], deployment_rows: list[dict]) -> list[dict]:
    """Each API offer priced for each workload shape that was benchmarked for its model."""
    shapes = {(d["model"], d["workload"], d["input_len"], d["output_len"])
              for d in deployment_rows}
    rows = []
    for model, workload, in_len, out_len in sorted(shapes):
        for o in api_offers:
            if o.model != model:
                continue
            rows.append({
                "model": model, "workload": workload, "provider": o.provider,
                "quantization": o.quantization,
                "usd_per_1m_input": o.usd_per_1m_input,
                "usd_per_1m_output": o.usd_per_1m_output,
                "usd_per_1k_requests": api_usd_per_request(o, in_len, out_len) * 1e3,
            })
    return rows


def comparisons(deployment_rows: list[dict], api_price_rows: list[dict]) -> list[dict]:
    """Per (model, workload): cheapest self-hosted setup vs cheapest and median API."""
    out = []
    keys = sorted({(d["model"], d["workload"]) for d in deployment_rows})
    for model, workload in keys:
        candidates = [
            (o["usd_per_1k_requests"], d, o)
            for d in deployment_rows if (d["model"], d["workload"]) == (model, workload)
            for o in d["offers"]
        ]
        if not candidates:
            continue
        _, best_d, best_o = min(candidates, key=lambda c: c[0])
        apis = [a for a in api_price_rows if (a["model"], a["workload"]) == (model, workload)]
        row = {
            "model": model, "workload": workload,
            "input_len": best_d["input_len"], "output_len": best_d["output_len"],
            "hosted": {
                "run_id": best_d["run_id"], "precision": best_d["precision"],
                "gpu_type": best_d["gpu_type"], "gpu_count": best_d["gpu_count"],
                "provider": best_o["provider"], "provisioning": best_o["provisioning"],
                "usd_per_hour": best_o["usd_per_hour"],
                "usd_per_1k_requests": best_o["usd_per_1k_requests"],
                "usd_per_1m_output_tokens": best_o["usd_per_1m_output_tokens"],
                "max_users_slo": best_d["max_users_slo"],
                "requests_per_day_at_capacity": best_d["requests_per_s"] * 86400,
            },
            "api_cheapest": None, "api_median_usd_per_1k_requests": None,
        }
        if apis:
            cheapest = min(apis, key=lambda a: a["usd_per_1k_requests"])
            api_per_req = cheapest["usd_per_1k_requests"] / 1e3
            breakeven_rpd = best_o["usd_per_hour"] * 24 / api_per_req
            row.update({
                "api_cheapest": cheapest,
                "api_median_usd_per_1k_requests": statistics.median(
                    a["usd_per_1k_requests"] for a in apis),
                # Requests/day at which an always-on GPU costs the same as the cheapest API.
                "breakeven_requests_per_day": breakeven_rpd,
                "breakeven_utilization": breakeven_rpd / row["hosted"][
                    "requests_per_day_at_capacity"],
                # Saving vs the cheapest API when the GPU is kept busy at its SLO capacity.
                "savings_at_capacity": 1 - best_o["usd_per_1k_requests"] / 1e3 / api_per_req,
            })
        out.append(row)
    return out


def build_cost(runs: list[RunResult], gpu_offers: list[GpuOffer],
               api_offers: list[ApiOffer]) -> dict:
    deps = deployments(runs, gpu_offers)
    apis = api_rows(api_offers, deps)
    return {
        "assumptions": {
            "hours_per_month": HOURS_PER_MONTH,
            "hosted": "always-on rented GPU; throughput at the most concurrent users that "
                      "meet the latency SLO (p99 TTFT and median inter-token latency)",
            "api": "list price per token for the same model on each provider "
                   "(input and output priced separately)",
        },
        "gpu_offers": [o.model_dump(mode="json") for o in gpu_offers],
        "api_offers": [o.model_dump(mode="json") for o in api_offers],
        "deployments": deps,
        "api": apis,
        "comparisons": comparisons(deps, apis),
    }
