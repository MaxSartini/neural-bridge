"""Registered numerical-equivalence and throughput backtest for Stage A executors."""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import cast

from neural_bridge.veatic21.contracts import (
    PHASE02_EXECUTOR_BACKTEST_REGISTRATION,
    PHASE02_EXECUTOR_BACKTEST_ROOT,
    PHASE02_REGISTRATION_SHA256,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json
from neural_bridge.veatic21.phase02_stage_a_executor import (
    ExecutorConfiguration,
    ExecutorRunSelection,
    _configuration_from_dict,
    executor_code_identity,
    run_hardware_saturated_executor,
)

METRIC_FIELDS = frozenset({"raw_pr_auc", "roc_auc", "brier"})
EXACT_UNIT_FIELDS = (
    "unit",
    "split_sha256",
    "feature_names",
    "feature_count",
    "feature_matrix_sha256",
    "scaler_sha256",
    "target_thresholds_sha256",
    "train_row_counts",
    "validation_row_counts",
    "configuration_count",
    "outer_test_scores_opened",
    "cortical_values_opened",
)
SIX_GIBIBYTES = 6 * 1024**3


def _as_int(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected int, received {type(value).__name__}")
    return value


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected numeric value, received {type(value).__name__}")
    return float(value)


def _selection_from_ranges(value: object) -> ExecutorRunSelection:
    if not isinstance(value, list) or not value:
        raise ValueError("executor sequence ranges must be a non-empty list")
    ranges: list[tuple[int, int]] = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("invalid executor sequence range")
        start = _as_int(pair[0])
        end = _as_int(pair[1])
        if start < 0 or end < start:
            raise ValueError("invalid executor sequence range")
        ranges.append((start, end))
    return ExecutorRunSelection(tuple(ranges))


def _single_range_selection(value: object) -> ExecutorRunSelection:
    if not isinstance(value, list):
        raise ValueError("executor single range must be a list")
    return _selection_from_ranges([value])


def _unit_paths_for_selection(output_root: Path, selection: ExecutorRunSelection) -> list[Path]:
    registry = load_json(output_root / "work-unit-registry.json")
    units = cast(list[dict[str, object]], registry["units"])
    selected = [unit for unit in units if selection.contains(cast(int, unit["sequence"]))]
    selected.sort(key=lambda unit: cast(int, unit["sequence"]))
    return [output_root / "units" / f"{unit['unit_id']}.json" for unit in selected]


def _record_mismatch(
    stats: dict[str, object], path: str, reference: object, candidate: object
) -> None:
    stats["mismatch_count"] = cast(int, stats["mismatch_count"]) + 1
    examples = cast(list[dict[str, object]], stats["mismatch_examples"])
    if len(examples) < 100:
        examples.append({"path": path, "reference": reference, "candidate": candidate})


def _compare_solver_value(
    reference: object,
    candidate: object,
    *,
    path: str,
    tolerance: float,
    stats: dict[str, object],
) -> None:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        reference = {
            key: value for key, value in reference.items() if key != "compiled_update_blocks"
        }
        candidate = {
            key: value for key, value in candidate.items() if key != "compiled_update_blocks"
        }
        if set(reference) != set(candidate):
            _record_mismatch(stats, f"{path}.keys", sorted(reference), sorted(candidate))
            return
        for key in sorted(reference):
            _compare_solver_value(
                reference[key],
                candidate[key],
                path=f"{path}.{key}",
                tolerance=tolerance,
                stats=stats,
            )
        return
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            _record_mismatch(stats, f"{path}.length", len(reference), len(candidate))
            return
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            _compare_solver_value(
                left,
                right,
                path=f"{path}[{index}]",
                tolerance=tolerance,
                stats=stats,
            )
        return
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
    ):
        if isinstance(reference, int) and isinstance(candidate, int):
            if reference != candidate:
                _record_mismatch(stats, path, reference, candidate)
            return
        difference = abs(float(reference) - float(candidate))
        stats["max_solver_absolute_difference"] = max(
            cast(float, stats["max_solver_absolute_difference"]), difference
        )
        if not math.isfinite(difference) or difference > tolerance:
            _record_mismatch(stats, path, reference, candidate)
        return
    if reference != candidate:
        _record_mismatch(stats, path, reference, candidate)


def _compare_stage_a_unit(
    reference: dict[str, object],
    candidate: dict[str, object],
    *,
    metric_tolerance: float,
    solver_tolerance: float,
    stats: dict[str, object],
) -> None:
    unit_id = cast(dict[str, object], reference["unit"])["unit_id"]
    for field in EXACT_UNIT_FIELDS:
        if reference.get(field) != candidate.get(field):
            _record_mismatch(
                stats, f"{unit_id}.{field}", reference.get(field), candidate.get(field)
            )

    reference_records = cast(list[dict[str, object]], reference["records"])
    candidate_records = cast(list[dict[str, object]], candidate["records"])
    if len(reference_records) != len(candidate_records):
        _record_mismatch(
            stats,
            f"{unit_id}.records.length",
            len(reference_records),
            len(candidate_records),
        )
    else:
        for index, (left, right) in enumerate(
            zip(reference_records, candidate_records, strict=True)
        ):
            left_nonmetric = {key: value for key, value in left.items() if key not in METRIC_FIELDS}
            right_nonmetric = {
                key: value for key, value in right.items() if key not in METRIC_FIELDS
            }
            if left_nonmetric != right_nonmetric:
                _record_mismatch(
                    stats,
                    f"{unit_id}.records[{index}].nonmetric",
                    left_nonmetric,
                    right_nonmetric,
                )
            for field in METRIC_FIELDS:
                left_value = left.get(field)
                right_value = right.get(field)
                if left_value is None or right_value is None:
                    if left_value != right_value:
                        _record_mismatch(
                            stats,
                            f"{unit_id}.records[{index}].{field}",
                            left_value,
                            right_value,
                        )
                    continue
                difference = abs(float(cast(float, left_value)) - float(cast(float, right_value)))
                stats["max_metric_absolute_difference"] = max(
                    cast(float, stats["max_metric_absolute_difference"]), difference
                )
                if not math.isfinite(difference) or difference > metric_tolerance:
                    _record_mismatch(
                        stats,
                        f"{unit_id}.records[{index}].{field}",
                        left_value,
                        right_value,
                    )

    _compare_solver_value(
        reference["solver"],
        candidate["solver"],
        path=f"{unit_id}.solver",
        tolerance=solver_tolerance,
        stats=stats,
    )


def compare_executor_output_to_reference(
    *,
    reference_root: Path,
    candidate_root: Path,
    selection: ExecutorRunSelection,
    metric_tolerance: float,
    solver_tolerance: float,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "status": "RUNNING",
        "units_compared": 0,
        "mismatch_count": 0,
        "max_metric_absolute_difference": 0.0,
        "max_solver_absolute_difference": 0.0,
        "mismatch_examples": [],
    }
    reference_paths = {
        load_json(path)["unit"]["unit_id"]: path
        for path in _unit_paths_for_selection(reference_root, selection)
    }
    candidate_paths = _unit_paths_for_selection(candidate_root, selection)
    for candidate_path in candidate_paths:
        candidate = load_json(candidate_path)
        unit_id = cast(dict[str, object], candidate["unit"])["unit_id"]
        reference_path = reference_paths.get(unit_id)
        if reference_path is None:
            _record_mismatch(stats, f"{unit_id}.reference", "present", "missing")
            continue
        _compare_stage_a_unit(
            load_json(reference_path),
            candidate,
            metric_tolerance=metric_tolerance,
            solver_tolerance=solver_tolerance,
            stats=stats,
        )
        stats["units_compared"] = _as_int(stats["units_compared"]) + 1
    if len(candidate_paths) != len(reference_paths):
        _record_mismatch(
            stats,
            "unit_count",
            len(reference_paths),
            len(candidate_paths),
        )
    stats["status"] = "PASS" if stats["mismatch_count"] == 0 else "FAIL"
    stats["outer_test_scores_opened"] = False
    stats["cortical_values_opened"] = False
    return stats


def _normalized_unit_digest(output_root: Path, selection: ExecutorRunSelection) -> str:
    digest = hashlib.sha256()
    for path in _unit_paths_for_selection(output_root, selection):
        value = load_json(path)
        value.pop("runtime_seconds", None)
        value.pop("execution_provenance", None)
        digest.update(path.stem.encode())
        digest.update(b"\0")
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _resume_snapshot(output_root: Path) -> dict[str, object]:
    unit_paths = sorted((output_root / "units").glob("*.json"))
    shard_ledgers = sorted((output_root / "shards").glob("shard-*/append-only-ledger.jsonl"))
    return {
        "unit_count": len(unit_paths),
        "unit_sha256": {path.name: sha256_file(path) for path in unit_paths},
        "shard_ledger_sha256": {str(path): sha256_file(path) for path in shard_ledgers},
        "canonical_ledger_sha256": sha256_file(output_root / "append-only-experiment-ledger.jsonl"),
    }


def _access_gate(output_root: Path) -> dict[str, object]:
    failures: list[str] = []
    for path in sorted((output_root / "units").glob("*.json")):
        value = load_json(path)
        provenance = cast(dict[str, object], value.get("execution_provenance", {}))
        if value.get("outer_test_scores_opened") is not False:
            failures.append(f"{path.name}:outer")
        if value.get("cortical_values_opened") is not False:
            failures.append(f"{path.name}:cortical")
        if provenance.get("outer_test_scores_opened") is not False:
            failures.append(f"{path.name}:provenance_outer")
        if provenance.get("cortical_values_opened") is not False:
            failures.append(f"{path.name}:provenance_cortical")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failure_count": len(failures),
        "failure_examples": failures[:100],
    }


def _candidate_summary(
    *,
    configuration: ExecutorConfiguration,
    equivalence: dict[str, object],
    resume_pass: bool,
    timed_roots: list[Path],
    timed_selection: ExecutorRunSelection,
) -> dict[str, object]:
    timed_states = [load_json(path / "run-state.json") for path in timed_roots]
    throughput = [float(state["work_units_per_second"]) for state in timed_states]
    normalized = [_normalized_unit_digest(path, timed_selection) for path in timed_roots]
    headroom = [
        cast(dict[str, object], state["resource_summary"]).get(
            "estimated_minimum_memory_headroom_bytes"
        )
        for state in timed_states
    ]
    gpu = [
        cast(dict[str, object], state["resource_summary"]).get(
            "gpu_device_utilization_mean_percent"
        )
        for state in timed_states
    ]
    sample_counts = [
        _as_int(cast(dict[str, object], state["resource_summary"])["sample_count"])
        for state in timed_states
    ]
    access = [_access_gate(path) for path in timed_roots]
    thermal_snapshots = [
        cast(dict[str, object], state["resource_summary"]).get("pressure_after", {})
        for state in timed_states
    ]
    thermal_pass = all(
        "No thermal warning level has been recorded"
        in str(cast(dict[str, object], item).get("thermal"))
        and "No performance warning level has been recorded"
        in str(cast(dict[str, object], item).get("thermal"))
        for item in thermal_snapshots
    )
    memory_pass = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= SIX_GIBIBYTES
        for value in headroom
    ) and all(count > 0 for count in sample_counts)
    deterministic_pass = len(set(normalized)) == 1
    access_pass = all(item["status"] == "PASS" for item in access)
    ledger_pass = all(
        int(state["ledger_lines"]) == int(state["work_units_total"])
        and int(state["unique_unit_ids"]) == int(state["work_units_total"])
        for state in timed_states
    )
    eligibility_without_saturation = all(
        (
            equivalence["status"] == "PASS",
            resume_pass,
            deterministic_pass,
            memory_pass,
            thermal_pass,
            access_pass,
            ledger_pass,
        )
    )
    peak_mlx = [
        sum(
            _as_int(shard["peak_mlx_active_memory_bytes"])
            for shard in cast(list[dict[str, object]], state["shards"])
        )
        for state in timed_states
    ]
    return {
        "configuration": asdict(configuration),
        "status": "PASS" if eligibility_without_saturation else "FAIL",
        "eligible_without_saturation_gate": eligibility_without_saturation,
        "equivalence": equivalence,
        "resume_gate": "PASS" if resume_pass else "FAIL",
        "determinism_gate": "PASS" if deterministic_pass else "FAIL",
        "normalized_timed_unit_digests": normalized,
        "ledger_gate": "PASS" if ledger_pass else "FAIL",
        "access_gate": "PASS" if access_pass else "FAIL",
        "memory_gate": "PASS" if memory_pass else "FAIL",
        "thermal_gate": "PASS" if thermal_pass else "FAIL",
        "timed_work_units_per_second": throughput,
        "median_work_units_per_second": median(throughput),
        "timed_elapsed_seconds": [float(state["elapsed_seconds"]) for state in timed_states],
        "gpu_utilization_mean_percent_by_repetition": gpu,
        "gpu_utilization_mean_percent": (
            fmean(_as_float(value) for value in gpu if value is not None)
            if any(value is not None for value in gpu)
            else None
        ),
        "estimated_minimum_memory_headroom_bytes_by_repetition": headroom,
        "peak_summed_mlx_active_memory_bytes_by_repetition": peak_mlx,
        "sample_counts": sample_counts,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }


def _select_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("eligible_without_saturation_gate") is True
    ]
    if not eligible:
        return {"status": "FAIL", "reason": "no_candidate_passed_prerequisite_gates"}
    fastest = max(_as_float(candidate["median_work_units_per_second"]) for candidate in eligible)
    tied = [
        candidate
        for candidate in eligible
        if _as_float(candidate["median_work_units_per_second"]) >= fastest * 0.97
    ]

    def concurrency(candidate: dict[str, object]) -> int:
        config = cast(dict[str, object], candidate["configuration"])
        return _as_int(config["mlx_lanes"]) * _as_int(config["gpu_streams_per_lane"])

    selected = min(
        tied,
        key=lambda candidate: (
            concurrency(candidate),
            _as_int(cast(dict[str, object], candidate["configuration"])["mlx_lanes"]),
            max(cast(list[int], candidate["peak_summed_mlx_active_memory_bytes_by_repetition"])),
        ),
    )
    selected_concurrency = concurrency(selected)
    selected_speed = _as_float(selected["median_work_units_per_second"])
    gpu = selected.get("gpu_utilization_mean_percent")
    next_candidates = [
        candidate for candidate in eligible if concurrency(candidate) > selected_concurrency
    ]
    saturation: dict[str, object]
    if gpu is not None and _as_float(gpu) >= 90.0:
        saturation = {
            "status": "PASS",
            "reason": "mean_gpu_device_utilization_at_least_ninety_percent",
        }
    elif next_candidates:
        next_concurrency = min(concurrency(candidate) for candidate in next_candidates)
        next_speed = max(
            _as_float(candidate["median_work_units_per_second"])
            for candidate in next_candidates
            if concurrency(candidate) == next_concurrency
        )
        gain = (next_speed - selected_speed) / selected_speed
        saturation = {
            "status": "PASS" if gain < 0.05 else "FAIL",
            "reason": "next_safe_concurrency_gain_below_five_percent"
            if gain < 0.05
            else "next_safe_concurrency_not_saturated",
            "selected_total_gpu_concurrency": selected_concurrency,
            "next_total_gpu_concurrency": next_concurrency,
            "next_throughput_gain_fraction": gain,
        }
    elif selected_concurrency == 12:
        saturation = {
            "status": "PASS",
            "reason": "safe_registered_host_concurrency_ceiling_reached",
        }
    else:
        saturation = {
            "status": "FAIL",
            "reason": "no_higher_safe_configuration_and_gpu_below_ninety_percent",
        }
    return {
        "status": saturation["status"],
        "selected_configuration_id": cast(dict[str, object], selected["configuration"])["id"],
        "selected_configuration": selected["configuration"],
        "selected_executor_sha256": executor_code_identity(),
        "selected_median_work_units_per_second": selected_speed,
        "fastest_observed_work_units_per_second": fastest,
        "tie_within_three_percent_applied": selected_speed < fastest,
        "saturation_gate": saturation,
    }


def run_phase02_executor_backtest(
    *, output_root: Path = PHASE02_EXECUTOR_BACKTEST_ROOT
) -> dict[str, object]:
    """Run every frozen executor candidate and select only from systems evidence."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE02_EXECUTOR_BACKTEST_ROOT:
        raise ValueError(f"executor backtest must use the canonical root: {output_root}")
    registration = load_json(PHASE02_EXECUTOR_BACKTEST_REGISTRATION)
    if registration.get("scientific_registration_sha256") != PHASE02_REGISTRATION_SHA256:
        raise ValueError("executor backtest scientific registration identity changed")
    if registration.get("outer_test_scores_allowed") is not False:
        raise ValueError("executor backtest cannot authorize outer-test access")
    if registration.get("cortical_values_allowed") is not False:
        raise ValueError("executor backtest cannot authorize cortical access")
    representative = cast(dict[str, object], registration["representative_units"])
    equivalence_selection = _selection_from_ranges(
        representative["numerical_equivalence_sequence_ranges_inclusive"]
    )
    timed_selection = _single_range_selection(representative["timed_sequence_range_inclusive"])
    warmup_selection = _single_range_selection(representative["warmup_sequence_range_inclusive"])
    repetitions = _as_int(representative["timed_repetitions"])
    numerical = cast(dict[str, object], registration["numerical_equivalence"])
    reference = cast(dict[str, object], registration["reference_attempt"])
    reference_root = Path(cast(str, reference["root"]))
    configurations = [
        _configuration_from_dict(value)
        for value in cast(list[dict[str, object]], registration["candidate_configurations"])
    ]
    request = {
        "schema_version": "veatic21_phase02_executor_backtest_request_v2",
        "registration_sha256": sha256_file(PHASE02_EXECUTOR_BACKTEST_REGISTRATION),
        "scientific_registration_sha256": PHASE02_REGISTRATION_SHA256,
        "executor_sha256": executor_code_identity(),
        "configurations": [asdict(configuration) for configuration in configurations],
        "equivalence_selection": asdict(equivalence_selection),
        "timed_selection": asdict(timed_selection),
        "warmup_selection": asdict(warmup_selection),
        "timed_repetitions": repetitions,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    request_path = output_root / "request.json"
    if request_path.exists() and load_json(request_path) != request:
        raise ValueError("executor backtest request identity changed")
    _write_json(request_path, request)
    summaries: list[dict[str, object]] = []
    for configuration in configurations:
        candidate_root = output_root / "candidates" / configuration.id
        try:
            _write_json(
                candidate_root / "candidate-state.json",
                {"status": "RUNNING_EQUIVALENCE", "configuration": asdict(configuration)},
            )
            equivalence_root = candidate_root / "equivalence"
            run_hardware_saturated_executor(
                output_root=equivalence_root,
                configuration=configuration,
                selection=equivalence_selection,
            )
            equivalence = compare_executor_output_to_reference(
                reference_root=reference_root,
                candidate_root=equivalence_root,
                selection=equivalence_selection,
                metric_tolerance=_as_float(numerical["metric_absolute_tolerance"]),
                solver_tolerance=_as_float(numerical["solver_diagnostic_absolute_tolerance"]),
            )
            _write_json(candidate_root / "equivalence-result.json", equivalence)
            before_resume = _resume_snapshot(equivalence_root)
            run_hardware_saturated_executor(
                output_root=equivalence_root,
                configuration=configuration,
                selection=equivalence_selection,
            )
            after_resume = _resume_snapshot(equivalence_root)
            resume_pass = before_resume == after_resume
            _write_json(
                candidate_root / "resume-result.json",
                {
                    "status": "PASS" if resume_pass else "FAIL",
                    "before": before_resume,
                    "after": after_resume,
                },
            )
            timed_roots: list[Path] = []
            for repetition in range(repetitions):
                _write_json(
                    candidate_root / "candidate-state.json",
                    {
                        "status": "RUNNING_TIMED_REPETITION",
                        "configuration": asdict(configuration),
                        "repetition": repetition,
                    },
                )
                repetition_root = candidate_root / "timed" / f"repetition-{repetition:02d}"
                run_hardware_saturated_executor(
                    output_root=repetition_root,
                    configuration=configuration,
                    selection=timed_selection,
                    warmup_selection=warmup_selection,
                )
                timed_roots.append(repetition_root)
            summary = _candidate_summary(
                configuration=configuration,
                equivalence=equivalence,
                resume_pass=resume_pass,
                timed_roots=timed_roots,
                timed_selection=timed_selection,
            )
        except BaseException as error:
            summary = cast(
                dict[str, object],
                {
                    "configuration": asdict(configuration),
                    "status": "ERROR",
                    "eligible_without_saturation_gate": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "outer_test_scores_opened": False,
                    "cortical_values_opened": False,
                },
            )
        summaries.append(summary)
        _write_json(candidate_root / "candidate-state.json", summary)
        _write_json(output_root / "candidate-summaries.json", summaries)
    selection = _select_candidate(summaries)
    result = {
        "schema_version": "veatic21_phase02_executor_backtest_result_v2",
        "status": "PASS" if selection["status"] == "PASS" else "FAIL",
        "registration_sha256": sha256_file(PHASE02_EXECUTOR_BACKTEST_REGISTRATION),
        "scientific_registration_sha256": PHASE02_REGISTRATION_SHA256,
        "executor_sha256": executor_code_identity(),
        "candidate_count": len(configurations),
        "candidates": summaries,
        "selection": selection,
        "request_sha256": sha256_file(request_path),
        "scientific_scores_used_for_executor_selection": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(output_root / "result.json", result)
    return result
