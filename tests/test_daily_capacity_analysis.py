"""Measured rates and explicitly modeled queue recovery stay separate."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "benchmarks/analyze_daily_capacity.py"
spec = importlib.util.spec_from_file_location("daily_capacity", SCRIPT)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def evidence():
    arrivals = {
        "daily": {
            f"2026-10-{day:02d}": {"changed_or_recovered_eligible_upper_bound": 6000}
            for day in range(1, 15)
        }
    }
    base = {
        "sample_count": 512,
        "source_sha256": "sample-hash",
        "failures": [],
        "seconds": {"photo_download": 0.0},
        "model": {
            "backbone": "bioclip25",
            "revision": "frozen",
            "device": "cpu",
            "sha256": {"weights": "verified-hash"},
        },
    }
    first = {
        **copy.deepcopy(base),
        "status": "intentionally_stopped",
        "encoded_count": 64,
        "reused_count": 0,
        "completed_count": 64,
        "chunks": [{"count": 32, "elapsed_seconds": 80.0}] * 2,
        "batches": [{"count": 1}] * 64,
    }
    resumed = {
        **copy.deepcopy(base),
        "status": "complete",
        "encoded_count": 448,
        "reused_count": 64,
        "completed_count": 512,
        "chunks": [{"count": 32, "elapsed_seconds": 64.0}] * 14,
        "batches": [{"count": 1}] * 448,
    }
    cached = {
        **copy.deepcopy(base),
        "status": "cached",
        "encoded_count": 0,
        "reused_count": 512,
        "completed_count": 512,
        "batches": [],
        "model_loaded": False,
    }
    cached["model"].pop("sha256")
    index = {
        "status": "complete",
        "synthetic": True,
        "query_count": 512,
        "append_count": 512,
        "timings_seconds": {"search": 2.0, "append": 4.0},
    }
    return {
        "arrivals": arrivals,
        "first": first,
        "resumed": resumed,
        "cached": cached,
        "index": index,
    }


def test_budget_preserves_headroom_and_overhead():
    assert capacity.daily_capacity(2, 2.5) == 8280
    assert capacity.daily_capacity(2, 2.5, reserved=0) == 11040
    assert capacity.daily_capacity(4, 2.5, hours=5) == 20880


def test_analysis_uses_slower_host_and_scopes_gate():
    result = capacity.analyze(**evidence())
    assert result["encoding"]["planning_seconds_per_photo"] == 2.5
    assert result["policy"]["normal_capacity_per_day"] == 8280
    assert result["evidence_gates"]["observed_trace_encoding_capacity"] == "passes"
    assert result["evidence_gates"]["production_updater_end_to_end"] == "UNVERIFIED"
    assert len(result["queue_model"]["scenarios"]) == 56
    assert result["index_evidence"]["synthetic"] is True
    assert result["index_evidence"]["incremental_cost_scaling"]["search"]["measured_count"] == 512


def test_outage_keeps_arrivals_and_recovery_is_modeled():
    result = capacity.simulate([10], normal=12, recovery=20, missed=3, phase=0, horizon=10)
    assert result["max_end_day_backlog"] == 30
    assert result["recovery_processing_days"] == 3
    assert [day["backlog"] for day in result["days"][:6]] == [10, 20, 30, 20, 10, 0]
    assert result["days"][6]["mode"] == "normal"


def test_zero_capacity_never_claims_recovery():
    result = capacity.simulate([10], normal=0, recovery=0, missed=0, phase=0, horizon=5)
    assert result["recovery_processing_days"] is None
    assert result["final_backlog"] == 50


@pytest.mark.parametrize(
    "mutation",
    [
        "failed",
        "coverage",
        "cache",
        "batch",
        "identity",
        "hash",
        "arrival",
        "date",
        "time",
        "index",
    ],
)
def test_incomplete_or_invalid_evidence_rejected(mutation):
    data = evidence()
    if mutation == "failed":
        data["first"]["failures"] = [{"id": 7}]
    elif mutation == "coverage":
        data["resumed"]["encoded_count"] = 447
    elif mutation == "cache":
        del data["cached"]["model_loaded"]
    elif mutation == "batch":
        data["cached"]["batches"] = [{"count": 1}]
    elif mutation == "identity":
        data["cached"]["source_sha256"] = "wrong"
    elif mutation == "hash":
        del data["first"]["model"]["sha256"]
    elif mutation == "arrival":
        data["arrivals"]["daily"]["2026-10-01"]["changed_or_recovered_eligible_upper_bound"] = -1
    elif mutation == "date":
        data["arrivals"]["daily"]["2026-11-01"] = data["arrivals"]["daily"].pop("2026-10-01")
    elif mutation == "time":
        data["first"]["chunks"][0]["elapsed_seconds"] = float("nan")
    else:
        data["index"]["status"] = "failed"
    with pytest.raises(ValueError):
        capacity.analyze(**data)


def test_low_capacity_fails_observed_gate_without_production_claim():
    data = evidence()
    for report in (data["first"], data["resumed"]):
        for chunk in report["chunks"]:
            chunk["elapsed_seconds"] = 3200
    result = capacity.analyze(**data)
    assert result["evidence_gates"]["observed_trace_encoding_capacity"] == "fails"
    assert result["queue_model"]["summaries"]["7"]["uncleared_phases"] == 14


def test_cold_download_is_amortized_over_all_512_not_first_64():
    data = evidence()
    data["first"]["seconds"]["photo_download"] = 64.0
    data["resumed"]["seconds"]["photo_download"] = 0.0
    result = capacity.analyze(**data)
    assert result["encoding"]["photo_download_seconds_per_sample"] == 0.125
    assert result["encoding"]["planning_seconds_per_photo"] == 2.625


def test_user_capped_policy_allows_recovery_longer_than_seven_days():
    result = capacity.simulate_with_time_budget([6235] * 14, rate=2.7, target=10000)
    assert result["target_daily_photo_cap"] == 10000
    assert result["effective_daily_capacity"] == 10000
    assert result["feasible_with_measured_budget"] is True
    assert result["user_allows_spillover"] is True
    assert result["observed_trace_gate"] == "passes"
    assert result["summaries"]["7"]["worst_recovery_processing_days"] > 7
    assert all(
        day["processed"] <= 10000 for scenario in result["scenarios"] for day in scenario["days"]
    )


def test_five_thousand_cap_cannot_keep_up_with_conservative_trace():
    result = capacity.simulate_with_time_budget([6235] * 14, rate=2.7, target=5000)
    assert result["observed_trace_gate"] == "fails"
    assert result["mean_arrivals_minus_daily_capacity"] == 1235
    assert result["summaries"]["7"]["uncleared_phases_within_horizon"] == 14


def test_capped_policy_respects_actual_compute_not_just_requested_target():
    result = capacity.simulate_with_time_budget([5000] * 14, rate=5, target=10000)
    assert result["raw_measured_budget_capacity_per_day"] == 6120
    assert result["effective_daily_capacity"] == 6120
    assert result["feasible_with_measured_budget"] is False
    assert result["observed_trace_gate"] == "passes"


def test_analyzer_exposes_user_policy_separately_from_legacy_comparison():
    result = capacity.analyze(**evidence())
    assert result["time_budget_spillover_policy"]["workers"] == 4
    assert result["time_budget_spillover_policy"]["worker_hours"] == 3
    assert result["time_budget_spillover_policy"]["target_daily_photo_cap"] is None
    assert result["policy"]["normal_workers"] == 2
    assert result["evidence_gates"]["annual_or_seasonal_capacity"] == "UNVERIFIED"


def test_time_budget_uses_reserved_capacity_without_arbitrary_photo_cap():
    result = capacity.simulate_with_time_budget([6235] * 14, rate=2.5)
    assert result["target_daily_photo_cap"] is None
    assert result["effective_daily_capacity"] == 12240
    assert result["capacity_without_compute_reserve_per_day"] == 16320
    assert result["observed_trace_gate"] == "passes"
    assert result["summaries"]["7"]["uncleared_phases_within_horizon"] == 0
    assert (
        max(day["processed"] for scenario in result["scenarios"] for day in scenario["days"])
        > 10000
    )
