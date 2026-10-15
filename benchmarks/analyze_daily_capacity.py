"""Reproduce an observed-trace CPU capacity scenario from measured benchmark evidence."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np


def number(value, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'} and finite")
    return float(value)


def daily_capacity(workers: int, rate: float, *, reserved: float = 0.25, hours: float = 4) -> int:
    number(rate, "seconds per photo", positive=True)
    if workers <= 0 or not 0 <= reserved < 1:
        raise ValueError("workers must be positive and reserved fraction must lie in [0,1)")
    number(hours, "worker hours", positive=True)
    if hours * 3600 <= 600:
        raise ValueError("worker hours must exceed the ten minute overhead")
    return math.floor(workers * (hours * 3600 - 600) * (1 - reserved) / rate)


def simulate(trace, normal: int, recovery: int, missed: int, phase: int, horizon: int = 90):
    """Daily arrivals precede service; recovery mode depends on the prior end-day queue."""
    if not trace or min(normal, recovery) < 0 or horizon <= 0 or missed < 0 or phase < 0:
        raise ValueError(
            "simulation requires a trace, positive capacities/horizon and valid outage"
        )
    for value in trace:
        number(value, "trace arrival count")
    backlog = 0
    maximum = 0
    maximum_pending = 0
    cleared = None
    days = []
    for day in range(horizon):
        previous = backlog
        arrivals = trace[(phase + day) % len(trace)]
        pending = previous + arrivals
        maximum_pending = max(maximum_pending, pending)
        mode = "outage" if day < missed else "recovery" if previous > 0 else "normal"
        capacity = 0 if mode == "outage" else recovery if mode == "recovery" else normal
        backlog = max(0, pending - capacity)
        maximum = max(maximum, backlog)
        if day >= missed and backlog == 0 and cleared is None:
            cleared = day - missed + 1
        days.append({"day": day + 1, "arrivals": arrivals, "mode": mode, "backlog": backlog})
    return {
        "missed_days": missed,
        "starting_phase": phase,
        "horizon_days": horizon,
        "recovery_processing_days": cleared,
        "max_end_day_backlog": maximum,
        "max_pending_before_service": maximum_pending,
        "final_backlog": backlog,
        "days": days,
    }


def simulate_with_time_budget(
    trace,
    rate: float,
    target: int | None = None,
    *,
    workers: int = 4,
    hours: float = 3,
    horizon: int = 90,
):
    """Use the worker time budget, optionally capped, and permit queue spillover."""
    if target is not None and (
        isinstance(target, bool) or not isinstance(target, int) or target <= 0
    ):
        raise ValueError("daily photo cap must be a positive integer")
    raw_capacity = daily_capacity(workers, rate, hours=hours)
    effective = raw_capacity if target is None else min(target, raw_capacity)
    scenarios = [
        simulate(trace, effective, effective, missed, phase, horizon=horizon)
        for missed in (0, 1, 3, 7)
        for phase in range(len(trace))
    ]
    for scenario in scenarios:
        previous = 0
        for day in scenario["days"]:
            day["mode"] = "outage" if day["mode"] == "outage" else "budgeted_processing"
            day["processed"] = previous + day["arrivals"] - day["backlog"]
            previous = day["backlog"]
    summaries = {}
    for missed in (0, 1, 3, 7):
        subset = [item for item in scenarios if item["missed_days"] == missed]
        cleared = [item["recovery_processing_days"] for item in subset]
        summaries[str(missed)] = {
            "starting_phases": len(trace),
            "uncleared_phases_within_horizon": sum(value is None for value in cleared),
            "worst_recovery_processing_days": max(
                (value for value in cleared if value is not None), default=None
            ),
            "max_end_day_backlog": max(item["max_end_day_backlog"] for item in subset),
            "max_final_backlog": max(item["final_backlog"] for item in subset),
        }
    mean = sum(trace) / len(trace)
    clears = all(item["recovery_processing_days"] is not None for item in scenarios)
    return {
        "target_daily_photo_cap": target,
        "workers": workers,
        "worker_hours": hours,
        "overhead_seconds_per_worker": 600,
        "reserved_compute_fraction": 0.25,
        "raw_measured_budget_capacity_per_day": raw_capacity,
        "effective_daily_capacity": effective,
        "feasible_with_measured_budget": target is None or raw_capacity >= target,
        "capacity_without_compute_reserve_per_day": daily_capacity(
            workers, rate, hours=hours, reserved=0
        ),
        "overhead_warning": (
            "600 seconds is an assumed budget for model, metadata, transfer "
            "and index work, not a measured production envelope"
        ),
        "user_allows_spillover": True,
        "new_observation_next_day_SLA": False,
        "observed_trace_gate": "passes" if mean < effective and clears else "fails",
        "gate_rule": (
            "Mean observed upper-bound arrivals below effective capacity, and "
            "all modeled outage backlogs clear within the stated horizon; "
            "no seven-processing-day deadline"
        ),
        "conservative_mean_arrivals": mean,
        "mean_arrivals_minus_daily_capacity": mean - effective,
        "horizon_days": horizon,
        "summaries": summaries,
        "scenarios": scenarios,
        "interpretation": (
            "Modeled repeated observed trace, not measured outage recovery, "
            "unique photo demand, or an annual capacity guarantee"
        ),
    }


def _host_rate(report, label):
    chunks = report.get("chunks", [])
    if not chunks or sum(chunk.get("count", 0) for chunk in chunks) != report["encoded_count"]:
        raise ValueError(f"{label}: chunk coverage differs from encoded count")
    rates = [
        number(chunk["elapsed_seconds"], f"{label} chunk time", positive=True)
        / number(chunk["count"], f"{label} chunk count", positive=True)
        for chunk in chunks
    ]
    batches = report.get("batches", [])
    if sum(batch.get("count", 0) for batch in batches) != report["encoded_count"]:
        raise ValueError(f"{label}: batch coverage differs from encoded count")
    return {
        "chunk_count": len(chunks),
        "per_chunk_seconds_per_photo": rates,
        "p95_chunk_seconds_per_photo": float(np.percentile(rates, 95)),
        "model_and_setup_seconds": {
            key: value
            for key, value in report.get("seconds", {}).items()
            if key
            in {
                "framework_import",
                "model_download_or_cache_lookup",
                "model_verification",
                "model_load",
            }
        },
    }


def analyze(
    arrivals,
    first,
    resumed,
    cached,
    index,
    *,
    seasonality=None,
    normal_worker_hours=4,
    recovery_worker_hours=4,
):
    if first.get("status") != "intentionally_stopped" or resumed.get("status") != "complete":
        raise ValueError("encoding must prove intentional checkpoint stop and completed resume")
    if cached.get("status") != "cached" or cached.get("model_loaded") is not False:
        raise ValueError("cached rerun must explicitly prove no model loading")
    if any(report.get("failures") for report in (first, resumed, cached)):
        raise ValueError("encoding evidence contains failures")
    expected = [
        (
            first,
            {"sample_count": 512, "encoded_count": 64, "reused_count": 0, "completed_count": 64},
        ),
        (
            resumed,
            {"sample_count": 512, "encoded_count": 448, "reused_count": 64, "completed_count": 512},
        ),
        (
            cached,
            {"sample_count": 512, "encoded_count": 0, "reused_count": 512, "completed_count": 512},
        ),
    ]
    for report, counts in expected:
        if any(report.get(key) != value for key, value in counts.items()):
            raise ValueError("benchmark coverage does not prove 64 + 448 = 512 with cached reuse")
    if cached.get("batches") != []:
        raise ValueError("cached rerun must prove zero inference batches")
    if not first.get("source_sha256") or any(
        report.get("source_sha256") != first["source_sha256"] for report in (resumed, cached)
    ):
        raise ValueError("benchmark source identity differs across runs")
    for key in ("backbone", "revision", "device"):
        value = first.get("model", {}).get(key)
        if not value or any(
            report.get("model", {}).get(key) != value for report in (resumed, cached)
        ):
            raise ValueError("benchmark model identity differs across runs")
    if (
        not first["model"].get("sha256")
        or resumed["model"].get("sha256") != first["model"]["sha256"]
    ):
        raise ValueError("encoding runs lack matching verified model hashes")
    if index.get("status") != "complete":
        raise ValueError("index benchmark did not complete")
    first_rate, resume_rate = _host_rate(first, "first host"), _host_rate(resumed, "resume host")
    chunk_rate = max(
        first_rate["p95_chunk_seconds_per_photo"], resume_rate["p95_chunk_seconds_per_photo"]
    )
    photo_seconds = sum(
        number(report.get("seconds", {}).get("photo_download"), "photo download duration")
        for report in (first, resumed)
    )
    photo_seconds_per_sample = photo_seconds / 512
    rate = chunk_rate + photo_seconds_per_sample
    normal = daily_capacity(2, rate, hours=normal_worker_hours)
    recovery = daily_capacity(4, rate, hours=recovery_worker_hours)
    daily = arrivals.get("daily", {})
    dates = sorted(daily)
    if len(dates) != 14:
        raise ValueError("arrival evidence must contain the actual 14-day trace")
    parsed = [date.fromisoformat(value) for value in dates]
    if any((right - left).days != 1 for left, right in zip(parsed, parsed[1:], strict=False)):
        raise ValueError("arrival dates must be consecutive")
    trace = []
    for day in dates:
        value = daily[day].get("changed_or_recovered_eligible_upper_bound")
        number(value, "daily eligible upper bound")
        if int(value) != value:
            raise ValueError("arrival counts must be integers")
        trace.append(int(value))
    scenarios = [
        simulate(trace, normal, recovery, missed, phase)
        for missed in (0, 1, 3, 7)
        for phase in range(14)
    ]
    summaries = {}
    for missed in (0, 1, 3, 7):
        subset = [item for item in scenarios if item["missed_days"] == missed]
        cleared = [item["recovery_processing_days"] for item in subset]
        summaries[str(missed)] = {
            "starting_phases": 14,
            "uncleared_phases": sum(value is None for value in cleared),
            "worst_recovery_processing_days": max(
                (value for value in cleared if value is not None), default=None
            ),
            "max_end_day_backlog": max(item["max_end_day_backlog"] for item in subset),
        }
    seven = summaries["7"]
    mean = sum(trace) / len(trace)
    gate = (
        mean < normal
        and seven["uncleared_phases"] == 0
        and seven["worst_recovery_processing_days"] <= 7
    )
    timings = index.get("timings_seconds", {})
    for key, value in timings.items():
        number(value, f"index {key} timing")
    scaling = {}
    query_count = index.get("query_count")
    append_count = index.get("append_count")
    for name, count, key in (("search", query_count, "search"), ("append", append_count, "append")):
        if count is not None and key in timings:
            number(count, f"index {name} count", positive=True)
            scaling[name] = {
                "measured_count": count,
                "measured_seconds": timings[key],
                "normalized_seconds_per_512": timings[key] * 512 / count,
                "linear_extrapolation_mean_daily_seconds": timings[key] * mean / count,
                "linear_extrapolation_peak_daily_seconds": timings[key] * max(trace) / count,
            }
    budget_policy = simulate_with_time_budget(trace, rate)
    lower_cap_stress = simulate_with_time_budget(trace, rate, target=5000)
    return {
        "time_budget_spillover_policy": budget_policy,
        "five_thousand_daily_cap_stress": lower_cap_stress,
        "scope": (
            "CPU encoding capacity under observed BC needs-ID changed-record delivery upper bounds"
        ),
        "evidence_gates": {
            "checkpoint_resume_and_cached_reuse": "verified by supplied benchmark reports",
            "observed_trace_encoding_capacity": "passes" if gate else "fails",
            "observed_trace_encoding_capacity_scope": "Legacy two-worker/recovery comparison",
            "time_budget_spillover_observed_trace": budget_policy["observed_trace_gate"],
            "annual_or_seasonal_capacity": "UNVERIFIED",
            "release_transfer_and_deployment_io": "UNVERIFIED, not measured in these inputs",
            "global_existing_observation_neighbor_refresh": "UNVERIFIED",
            "full_population_retrieval_quality": "UNVERIFIED",
            "production_updater_end_to_end": "UNVERIFIED",
        },
        "encoding": {
            "first_host": first_rate,
            "resume_host": resume_rate,
            "planning_seconds_per_photo": rate,
            "planning_chunk_seconds_per_photo": chunk_rate,
            "measured_total_photo_download_seconds": photo_seconds,
            "photo_download_seconds_per_sample": photo_seconds_per_sample,
            "photo_rate_denominator": 512,
            "rate_rule": (
                "Maximum per-host p95 chunk elapsed/photo, plus combined photo staging "
                "seconds divided by all 512 samples. Setup is budgeted separately."
            ),
        },
        "policy": {
            "role": "Legacy comparison, separate from time budget spillover policy",
            "normal_workers": 2,
            "recovery_workers": 4,
            "normal_worker_hours": normal_worker_hours,
            "recovery_worker_hours": recovery_worker_hours,
            "overhead_seconds_per_worker": 600,
            "reserved_compute_fraction": 0.25,
            "normal_capacity_per_day": normal,
            "recovery_capacity_per_day": recovery,
            "formula": "floor(workers * (worker_hours * 3600 - 600) * 0.75 / seconds_per_photo)",
            "overhead_warning": (
                "600 seconds is an assumed budget for model, metadata, transfer "
                "and index work, not a measured production envelope"
            ),
        },
        "arrival_trace": {
            "dates": dates,
            "daily_upper_bounds": trace,
            "mean": mean,
            "maximum": max(trace),
            "caveats": arrivals.get("caveats", []),
            "scope": arrivals.get("scope"),
        },
        "queue_model": {
            "kind": (
                "Modeled repeated observed 14-day trace, "
                "not an actual outage experiment or annual forecast"
            ),
            "recovery_rule": (
                "Recovery capacity when prior end-day backlog is positive; "
                "otherwise normal capacity"
            ),
            "horizon_days": 90,
            "summaries": summaries,
            "scenarios": scenarios,
        },
        "index_evidence": {
            "synthetic": index.get("synthetic"),
            "synthetic_warning": index.get("quality_warning"),
            "measured_timings_seconds": timings,
            "incremental_cost_scaling": scaling,
            "scaling_warning": (
                "Linear extrapolations on this benchmark's index, not measured "
                "daily runs; local I/O excludes release transfer and deployment"
            ),
            "initial_index_bytes": index.get("initial_index_bytes"),
            "final_index_bytes": index.get("final_index_bytes"),
        },
        "seasonality_input_supplied": seasonality is not None,
        "seasonality_evidence": seasonality,
        "seasonality_note": (
            "Retrospective planning envelope only. "
            "No annual capacity gate inferred by this analyzer"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arrivals", "first", "resumed", "cached", "index", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seasonality", type=Path)
    parser.add_argument("--normal-worker-hours", type=float, default=4)
    parser.add_argument("--recovery-worker-hours", type=float, default=4)
    args = parser.parse_args(argv)
    inputs = {
        name: json.loads(getattr(args, name).read_text())
        for name in ("arrivals", "first", "resumed", "cached", "index")
    }
    if args.seasonality:
        inputs["seasonality"] = json.loads(args.seasonality.read_text())
    result = analyze(
        **inputs,
        normal_worker_hours=args.normal_worker_hours,
        recovery_worker_hours=args.recovery_worker_hours,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
