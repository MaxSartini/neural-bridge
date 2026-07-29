"""Independent verification of the staged sparse-rescue executor backtest."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json
from neural_bridge.veatic21.phase02_stage_a_rescue import rescue_solver_code_identity
from neural_bridge.veatic21.phase02_stage_a_rescue_backtest import (
    RESCUE_EXECUTOR_BACKTEST_REGISTRATION,
    RESCUE_EXECUTOR_BACKTEST_ROOT,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_executor import rescue_executor_code_identity

EXPECTED_STAGE_COUNTS = {
    "stage-1-cell-batch": 6,
    "stage-2-topology": 19,
    "stage-3-metric-workers": 16,
    "stage-4-compilation": 30,
    "stage-5-final-repetitions": 8,
}
REQUIRED_SAFETY_GATES = (
    "determinism_gate",
    "resume_gate",
    "ledger_gate",
    "access_gate",
    "memory_gate",
    "thermal_gate",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be int")
    return cast(int, value)


def _number(value: object, name: str) -> float:
    _require(isinstance(value, int | float) and not isinstance(value, bool), f"{name} changed")
    return float(cast(int | float, value))


def _configuration(summary: dict[str, object]) -> dict[str, object]:
    value = summary.get("configuration")
    _require(isinstance(value, dict), "candidate configuration changed")
    return cast(dict[str, object], value)


def _independent_selection(finalists: list[dict[str, object]]) -> dict[str, object]:
    eligible = sorted(
        (item for item in finalists if item.get("eligible_without_saturation_gate") is True),
        key=lambda item: -_number(
            item.get("median_rescue_cells_per_second"), "finalist median throughput"
        ),
    )
    _require(bool(eligible), "no eligible final rescue-executor configuration")
    fastest = _number(
        eligible[0].get("median_rescue_cells_per_second"), "fastest finalist throughput"
    )
    tied = [
        item
        for item in eligible
        if _number(item.get("median_rescue_cells_per_second"), "finalist throughput")
        >= fastest * 0.97
    ]

    def tie_key(item: dict[str, object]) -> tuple[int, int, int, int]:
        config = _configuration(item)
        peaks = cast(list[object], item.get("peak_summed_mlx_active_memory_bytes_by_repetition"))
        _require(bool(peaks), "finalist MLX memory evidence is empty")
        return (
            _integer(config.get("mlx_lanes"), "mlx_lanes")
            * _integer(config.get("gpu_streams_per_lane"), "gpu_streams_per_lane"),
            _integer(config.get("mlx_lanes"), "mlx_lanes"),
            max(_integer(value, "peak MLX memory") for value in peaks),
            -_integer(config.get("cell_batch_size"), "cell_batch_size"),
        )

    selected = min(tied, key=tie_key)
    config = _configuration(selected)
    gpu = _number(selected.get("gpu_utilization_mean_percent"), "selected GPU utilization")
    _require(gpu >= 90.0, "selected executor did not saturate the GPU")
    return {
        "status": "PASS",
        "selected_configuration": config,
        "selected_configuration_id": config["id"],
        "selected_executor_sha256": rescue_executor_code_identity(),
        "selected_solver_sha256": rescue_solver_code_identity(),
        "selected_median_rescue_cells_per_second": selected[
            "median_rescue_cells_per_second"
        ],
        "fastest_observed_rescue_cells_per_second": fastest,
        "tie_within_three_percent_applied": selected is not eligible[0],
        "saturation_gate": {
            "status": "PASS",
            "reason": "mean_gpu_at_least_ninety_percent",
        },
    }


def verify_rescue_executor_backtest(
    *,
    output_root: Path = RESCUE_EXECUTOR_BACKTEST_ROOT,
    write_verification: bool = True,
) -> dict[str, object]:
    """Verify coverage, gates, selection, identities, and the access firewall."""

    output_root = reject_forbidden_runtime_path(output_root)
    _require(
        output_root == RESCUE_EXECUTOR_BACKTEST_ROOT,
        "rescue executor backtest verification root changed",
    )
    registration = load_json(RESCUE_EXECUTOR_BACKTEST_REGISTRATION)
    request_path = output_root / "request.json"
    summaries_path = output_root / "stage-summaries.json"
    result_path = output_root / "result.json"
    request = load_json(request_path)
    summaries = load_json(summaries_path)
    result = load_json(result_path)

    _require(result.get("status") == "PASS", "rescue executor backtest did not pass")
    _require(result.get("stage_candidate_counts") == EXPECTED_STAGE_COUNTS, "stage counts changed")
    _require(set(summaries) == set(EXPECTED_STAGE_COUNTS), "stage summary coverage changed")
    _require(result.get("stages") == summaries, "result and stage summaries diverged")
    _require(
        result.get("registration_sha256") == sha256_file(RESCUE_EXECUTOR_BACKTEST_REGISTRATION),
        "backtest registration identity changed",
    )
    _require(result.get("request_sha256") == sha256_file(request_path), "request hash changed")
    _require(result.get("solver_sha256") == rescue_solver_code_identity(), "solver changed")
    _require(result.get("executor_sha256") == rescue_executor_code_identity(), "executor changed")
    _require(
        request.get("registration_sha256") == sha256_file(RESCUE_EXECUTOR_BACKTEST_REGISTRATION),
        "request registration identity changed",
    )

    input_identity = cast(dict[str, object], registration.get("input_identity"))
    _require(
        input_identity.get("rescue_solver_code_sha256") == rescue_solver_code_identity(),
        "registered rescue solver identity changed",
    )
    _require(
        input_identity.get("rescue_executor_code_sha256") == rescue_executor_code_identity(),
        "registered rescue executor identity changed",
    )

    candidates: list[dict[str, object]] = []
    candidate_ids: set[str] = set()
    for stage_name, expected_count in EXPECTED_STAGE_COUNTS.items():
        stage = cast(list[dict[str, object]], summaries[stage_name])
        _require(len(stage) == expected_count, f"{stage_name} candidate count changed")
        for candidate in stage:
            configuration = _configuration(candidate)
            identifier = cast(str, configuration.get("id"))
            _require(isinstance(identifier, str) and bool(identifier), "candidate id changed")
            _require(identifier not in candidate_ids, "duplicate staged candidate id")
            candidate_ids.add(identifier)
            candidates.append(candidate)
            for gate in REQUIRED_SAFETY_GATES:
                _require(candidate.get(gate) == "PASS", f"{identifier} failed {gate}")
            for firewall in (
                "outer_test_scores_opened",
                "cortical_values_opened",
                "aggregation_or_pruning_performed",
            ):
                _require(candidate.get(firewall) is False, f"{identifier} violated {firewall}")
            equivalence = cast(dict[str, object], candidate.get("equivalence"))
            eligible = candidate.get("eligible_without_saturation_gate") is True
            _require(
                (equivalence.get("status") == "PASS") is eligible,
                f"{identifier} eligibility/equivalence mismatch",
            )
            _require(
                (candidate.get("status") == "PASS") is eligible,
                f"{identifier} status/eligibility mismatch",
            )

    stage1 = cast(list[dict[str, object]], summaries["stage-1-cell-batch"])
    _require(
        {_integer(_configuration(item).get("cell_batch_size"), "batch") for item in stage1}
        == {1, 4, 8, 16, 32, 64},
        "cell-batch search coverage changed",
    )
    stage2 = cast(list[dict[str, object]], summaries["stage-2-topology"])
    search = cast(dict[str, object], registration["search"])
    registered_topologies = {
        tuple(cast(list[int], pair))
        for pair in cast(list[object], search["stage_2_safe_topologies"])
    }
    observed_topologies = {
        (
            _integer(_configuration(item).get("mlx_lanes"), "mlx_lanes"),
            _integer(_configuration(item).get("gpu_streams_per_lane"), "streams"),
        )
        for item in stage2
    }
    _require(observed_topologies == registered_topologies, "safe topology coverage changed")

    stage4 = cast(list[dict[str, object]], summaries["stage-4-compilation"])
    compilation_pairs: dict[tuple[int, int, int, int], set[bool]] = {}
    for item in stage4:
        config = _configuration(item)
        key = (
            _integer(config.get("mlx_lanes"), "mlx_lanes"),
            _integer(config.get("gpu_streams_per_lane"), "streams"),
            _integer(config.get("metric_workers_per_lane"), "metric workers"),
            _integer(config.get("cell_batch_size"), "cell batch"),
        )
        compilation_pairs.setdefault(key, set()).add(
            cast(bool, config["compiled_update_blocks"])
        )
    _require(
        len(compilation_pairs) == 15
        and all(values == {False, True} for values in compilation_pairs.values()),
        "compiled/uncompiled paired coverage changed",
    )

    finalists = cast(list[dict[str, object]], summaries["stage-5-final-repetitions"])
    for finalist in finalists:
        identifier = cast(str, _configuration(finalist)["id"])
        digests = cast(list[str], finalist.get("normalized_unit_digests"))
        rates = cast(list[object], finalist.get("rescue_cells_per_second"))
        elapsed = cast(list[object], finalist.get("elapsed_seconds"))
        _require(
            len(digests) == len(rates) == len(elapsed) == 3,
            f"{identifier} repetitions changed",
        )
        _require(len(set(digests)) == 1, f"{identifier} is not bitwise repeatable")
        equivalence = cast(dict[str, object], finalist["equivalence"])
        _require(equivalence.get("mismatch_count") == 0, f"{identifier} has mismatches")
        _require(
            _number(equivalence.get("max_float_absolute_difference"), "equivalence drift") == 0.0,
            f"{identifier} is not exactly equivalent",
        )

    independent_selection = _independent_selection(finalists)
    _require(result.get("selection") == independent_selection, "recorded selection changed")
    for firewall in (
        "scientific_scores_used_for_executor_selection",
        "outer_test_scores_opened",
        "cortical_values_opened",
        "aggregation_or_pruning_performed",
    ):
        _require(result.get(firewall) is False, f"backtest violated {firewall}")

    verification = {
        "schema_version": "veatic21_phase02_stage_a_rescue_executor_backtest_verification_v1",
        "status": "PASS",
        "registration_sha256": sha256_file(RESCUE_EXECUTOR_BACKTEST_REGISTRATION),
        "request_sha256": sha256_file(request_path),
        "stage_summaries_sha256": sha256_file(summaries_path),
        "result_sha256": sha256_file(result_path),
        "candidate_count": len(candidates),
        "stage_candidate_counts": EXPECTED_STAGE_COUNTS,
        "selection": independent_selection,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    if write_verification:
        _write_json(output_root / "verification.json", verification)
    return verification
