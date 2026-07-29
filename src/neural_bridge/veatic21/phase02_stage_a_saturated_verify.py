"""Exhaustive integrity verification for the saturated VEATIC Phase 02 Stage A run."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import cast

from neural_bridge.veatic21.contracts import (
    PHASE02_EXECUTOR_BACKTEST_REGISTRATION,
    PHASE02_REGISTRATION_ROOT,
    PHASE02_REGISTRATION_SHA256,
    PHASE02_STAGE_A_SATURATED_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json
from neural_bridge.veatic21.phase02_stage_a import (
    _stage_a_code_identity,
    enumerate_stage_a_work_units,
)
from neural_bridge.veatic21.phase02_stage_a_executor import executor_code_identity
from neural_bridge.veatic21.phase02_stage_a_saturated import (
    SELECTED_EXECUTOR,
    verify_phase02_selected_executor,
)

EXPECTED_WORK_UNITS = 40_824
EXPECTED_CONFIGURATIONS_PER_UNIT = 210
EXPECTED_CONFIGURATION_EVALUATIONS = 8_573_040
VERIFICATION_WORKERS = 8
EXPECTED_UNIT_KEYS = {
    "configuration_count",
    "cortical_values_opened",
    "execution_provenance",
    "feature_count",
    "feature_matrix_sha256",
    "feature_names",
    "outer_test_scores_opened",
    "records",
    "registration_sha256",
    "runtime_seconds",
    "scaler_sha256",
    "schema_version",
    "solver",
    "split_sha256",
    "stage_a_code_sha256",
    "target_thresholds_sha256",
    "train_row_counts",
    "unit",
    "validation_row_counts",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(payload: bytes) -> dict[str, object]:
    value = json.loads(payload, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast(dict[str, object], value)


def _strict_json_line(line: str) -> dict[str, object]:
    value = json.loads(line, parse_constant=_reject_nonfinite)
    if not isinstance(value, dict):
        raise TypeError("expected a JSONL object")
    return cast(dict[str, object], value)


def _ids_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _matrix_shape(value: object, rows: int, columns: int, name: str) -> list[list[object]]:
    matrix = cast(list[list[object]], value)
    _require(len(matrix) == rows, f"{name} row count changed")
    _require(all(len(row) == columns for row in matrix), f"{name} column count changed")
    return matrix


def _verify_unit_job(job: tuple[object, ...]) -> dict[str, object]:
    (
        path_text,
        expected_sha256,
        expected_configuration_ids_sha256,
        registry_unit_value,
        expected_executor_sha256,
        expected_configuration_value,
        expected_stage_a_code_sha256,
    ) = job
    path = Path(cast(str, path_text))
    payload = path.read_bytes()
    _require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"unit hash mismatch: {path}",
    )
    result = _strict_json_bytes(payload)
    registry_unit = cast(dict[str, object], registry_unit_value)
    expected_unit = {key: value for key, value in registry_unit.items() if key != "shard_index"}
    _require(set(result) == EXPECTED_UNIT_KEYS, f"unit schema changed: {path}")
    _require(result["unit"] == expected_unit, f"unit identity changed: {path}")
    _require(
        result["registration_sha256"] == PHASE02_REGISTRATION_SHA256,
        f"unit registration changed: {path}",
    )
    _require(
        result["stage_a_code_sha256"] == expected_stage_a_code_sha256,
        f"unit solver identity changed: {path}",
    )
    _require(result["outer_test_scores_opened"] is False, f"outer access in unit: {path}")
    _require(result["cortical_values_opened"] is False, f"cortical access in unit: {path}")
    _require(
        result["configuration_count"] == EXPECTED_CONFIGURATIONS_PER_UNIT,
        f"unit configuration count changed: {path}",
    )
    feature_names = cast(list[str], result["feature_names"])
    _require(result["feature_count"] == len(feature_names), f"feature schema changed: {path}")
    _require(
        len(cast(list[object], result["train_row_counts"])) == 21,
        f"train target coverage changed: {path}",
    )
    _require(
        len(cast(list[object], result["validation_row_counts"])) == 21,
        f"validation target coverage changed: {path}",
    )
    runtime = float(cast(float, result["runtime_seconds"]))
    _require(math.isfinite(runtime) and runtime >= 0.0, f"invalid runtime: {path}")

    configuration = cast(dict[str, object], expected_configuration_value)
    expected_provenance = {
        "schema_version": "veatic21_phase02_stage_a_execution_v3",
        "executor_sha256": expected_executor_sha256,
        "configuration_id": configuration["id"],
        "mlx_lanes": configuration["mlx_lanes"],
        "gpu_streams_per_lane": configuration["gpu_streams_per_lane"],
        "metric_workers_per_lane": configuration["metric_workers_per_lane"],
        "pair_cache": configuration["pair_cache"],
        "compiled_ridge_update_blocks": configuration["compiled_ridge_update_blocks"],
        "compiled_logistic_update_blocks": configuration["compiled_logistic_update_blocks"],
        "fast_metrics": configuration["fast_metrics"],
        "pipeline_depth": configuration["pipeline_depth"],
        "shard_index": registry_unit["shard_index"],
        "warmup": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _require(
        result["execution_provenance"] == expected_provenance,
        f"unit execution provenance changed: {path}",
    )

    records = cast(list[dict[str, object]], result["records"])
    _require(
        len(records) == EXPECTED_CONFIGURATIONS_PER_UNIT,
        f"record count changed: {path}",
    )
    configuration_ids = [cast(str, record["configuration_id"]) for record in records]
    _require(len(set(configuration_ids)) == len(configuration_ids), f"duplicate records: {path}")
    _require(
        _ids_digest(configuration_ids) == expected_configuration_ids_sha256,
        f"ledger/result configuration identity changed: {path}",
    )
    candidate_counts = Counter(cast(str, record["candidate_id"]) for record in records)
    _require(
        len(candidate_counts) == 21 and set(candidate_counts.values()) == {10},
        f"target/regularizer coverage changed: {path}",
    )
    model_family = cast(str, expected_unit["model_family"])
    feature_form = cast(str, expected_unit["feature_form"])
    history_depth = cast(int, expected_unit["history_depth"])
    for record in records:
        _require(record["model_family"] == model_family, f"record family changed: {path}")
        _require(record["feature_form"] == feature_form, f"record feature form changed: {path}")
        _require(
            record["history_depth_rows"] == history_depth,
            f"record history depth changed: {path}",
        )
        rows = int(cast(int, record["rows"]))
        _require(rows >= 0 and record["validation_rows"] == rows, f"row count changed: {path}")
        _require(
            int(cast(int, record["positives"])) + int(cast(int, record["negatives"])) == rows,
            f"event support changed: {path}",
        )

    solver = cast(dict[str, object], result["solver"])
    _require(solver["total_cells"] == 210, f"solver cell count changed: {path}")
    _require(solver["compiled_update_blocks"] is False, f"compiled solver in main run: {path}")
    converged_mask = _matrix_shape(solver["converged_mask"], 10, 21, "convergence mask")
    diagnostic_key = (
        "relative_residual_by_cell"
        if model_family == "continuous_ridge"
        else "relative_gradient_by_cell"
    )
    diagnostics = _matrix_shape(solver[diagnostic_key], 10, 21, diagnostic_key)
    converged_cells = sum(bool(value) for row in converged_mask for value in row)
    _require(solver["converged_cells"] == converged_cells, f"convergence count changed: {path}")
    for value in (value for row in diagnostics for value in row):
        number = float(cast(float, value))
        _require(math.isfinite(number) and number >= 0.0, f"invalid solver diagnostic: {path}")

    return {
        "bytes": len(payload),
        "records": len(records),
        "converged_records": sum(bool(record["converged"]) for record in records),
        "solver_converged_cells": converged_cells,
        "statuses": dict(Counter(cast(str, record["status"]) for record in records)),
        "dispositions": dict(Counter(cast(str, record["disposition"]) for record in records)),
        "model_family": model_family,
    }


def _merge_counts(target: Counter[str], value: object) -> None:
    target.update(cast(dict[str, int], value))


def _assert_float_equal(left: object, right: float, name: str) -> None:
    _require(
        math.isclose(float(cast(float, left)), right, rel_tol=0.0, abs_tol=1e-12),
        f"resource summary changed: {name}",
    )


def verify_phase02_stage_a_saturated_output(
    *, output_root: Path = PHASE02_STAGE_A_SATURATED_ROOT
) -> dict[str, object]:
    """Read, hash, and structurally verify every Stage A output and ledger entry."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE02_STAGE_A_SATURATED_ROOT:
        raise ValueError(f"Stage A verification must use the canonical root: {output_root}")
    selected = verify_phase02_selected_executor()
    selected_record = load_json(SELECTED_EXECUTOR)
    configuration = cast(dict[str, object], selected["configuration"])
    expected_executor_sha256 = cast(str, selected["selected_executor_sha256"])
    expected_stage_a_code_sha256 = _stage_a_code_identity()

    request_path = output_root / "request.json"
    registry_path = output_root / "work-unit-registry.json"
    run_state_path = output_root / "run-state.json"
    canonical_ledger_path = output_root / "append-only-experiment-ledger.jsonl"
    resource_summary_path = output_root / "resource-summary.json"
    resource_samples_path = output_root / "resource-samples.jsonl"
    request = load_json(request_path)
    registry = load_json(registry_path)
    run_state = load_json(run_state_path)
    resource_summary = load_json(resource_summary_path)

    _require(run_state["status"] == "COMPLETE", "Stage A final state is not complete")
    _require(request["configuration"] == configuration, "Stage A configuration changed")
    _require(
        request["selection"] == {"sequence_ranges_inclusive": [[0, 40_823]]},
        "Stage A full selection changed",
    )
    _require(
        request["warmup_selection"] == {"sequence_ranges_inclusive": [[504, 527]]},
        "Stage A warmup selection changed",
    )
    for value in (request, run_state):
        _require(value["outer_test_scores_opened"] is False, "Stage A opened outer-test data")
        _require(value["cortical_values_opened"] is False, "Stage A opened cortical data")
        _require(value["executor_sha256"] == expected_executor_sha256, "executor changed")
    _require(executor_code_identity() == expected_executor_sha256, "current executor changed")
    _require(
        request["stage_a_solver_code_sha256"] == expected_stage_a_code_sha256,
        "Stage A solver code changed",
    )
    _require(
        request["executor_backtest_registration_sha256"]
        == sha256_file(PHASE02_EXECUTOR_BACKTEST_REGISTRATION),
        "executor backtest registration changed",
    )
    _require(
        request["scientific_registration_sha256"] == PHASE02_REGISTRATION_SHA256,
        "scientific registration changed",
    )
    _require(request["work_units"] == EXPECTED_WORK_UNITS, "request unit count changed")
    _require(
        request["configuration_evaluations"] == EXPECTED_CONFIGURATION_EVALUATIONS,
        "request evaluation count changed",
    )
    _require(run_state["work_units_completed"] == EXPECTED_WORK_UNITS, "run incomplete")
    _require(run_state["work_units_total"] == EXPECTED_WORK_UNITS, "run total changed")
    _require(
        run_state["configuration_evaluations"] == EXPECTED_CONFIGURATION_EVALUATIONS,
        "run evaluation count changed",
    )
    _require(run_state["request_sha256"] == sha256_file(request_path), "request hash changed")
    _require(
        run_state["work_unit_registry_sha256"] == sha256_file(registry_path),
        "work-unit registry hash changed",
    )
    _require(
        run_state["canonical_ledger_sha256"] == sha256_file(canonical_ledger_path),
        "canonical ledger hash changed",
    )

    scientific_registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    split_registry = load_json(PHASE02_REGISTRATION_ROOT / "split-registry.json")
    expected_units = enumerate_stage_a_work_units(scientific_registration, split_registry)
    expected_registry_units = [
        {**asdict(unit), "shard_index": (unit.sequence // 2) % 3} for unit in expected_units
    ]
    registry_units = cast(list[dict[str, object]], registry["units"])
    _require(registry["executor_sha256"] == expected_executor_sha256, "registry executor changed")
    _require(registry_units == expected_registry_units, "work-unit registry content changed")

    unit_paths = sorted((output_root / "units").glob("*.json"))
    expected_unit_ids = [cast(str, unit["unit_id"]) for unit in registry_units]
    _require(len(unit_paths) == EXPECTED_WORK_UNITS, "unit file count changed")
    _require(
        {path.stem for path in unit_paths} == set(expected_unit_ids),
        "unit file coverage changed",
    )

    canonical_entries: list[dict[str, object]] = []
    with canonical_ledger_path.open(encoding="utf-8") as handle:
        for line in handle:
            canonical_entries.append(_strict_json_line(line))
    _require(len(canonical_entries) == EXPECTED_WORK_UNITS, "canonical ledger line count changed")
    canonical_by_id: dict[str, dict[str, object]] = {}
    for expected_id, entry in zip(expected_unit_ids, canonical_entries, strict=True):
        unit_id = cast(str, entry["unit_id"])
        _require(unit_id == expected_id, "canonical ledger order changed")
        _require(unit_id not in canonical_by_id, "canonical ledger contains a duplicate")
        _require(entry["configuration_count"] == 210, "ledger configuration count changed")
        _require(entry["outer_test_scores_opened"] is False, "ledger opened outer-test data")
        expected_path = output_root / "units" / f"{unit_id}.json"
        _require(entry["unit_result_path"] == str(expected_path), "ledger unit path changed")
        configuration_ids = cast(list[str], entry["configuration_ids"])
        _require(
            len(configuration_ids) == 210 and len(set(configuration_ids)) == 210,
            "ledger configuration identity changed",
        )
        canonical_by_id[unit_id] = entry

    shard_entries: dict[str, dict[str, object]] = {}
    shard_states = cast(list[dict[str, object]], run_state["shards"])
    _require(len(shard_states) == 3, "shard count changed")
    for shard_index, final_shard_state in enumerate(shard_states):
        shard_root = output_root / "shards" / f"shard-{shard_index:02d}"
        shard_state_path = shard_root / "shard-state.json"
        shard_ledger_path = shard_root / "append-only-ledger.jsonl"
        shard_state = load_json(shard_state_path)
        _require(shard_state == final_shard_state, "final/shard state mismatch")
        _require(shard_state["status"] == "COMPLETE", "shard is not complete")
        _require(shard_state["work_units_total"] == 13_608, "shard total changed")
        _require(shard_state["work_units_completed"] == 13_608, "shard incomplete")
        _require(
            shard_state["work_units_executed_this_call"] == 13_608,
            "shard did not execute the fresh complete selection",
        )
        _require(
            shard_state["ledger_sha256"] == sha256_file(shard_ledger_path), "shard ledger changed"
        )
        _require(shard_state["outer_test_scores_opened"] is False, "shard opened outer data")
        _require(shard_state["cortical_values_opened"] is False, "shard opened cortical data")
        with shard_ledger_path.open(encoding="utf-8") as handle:
            entries = [_strict_json_line(line) for line in handle]
        _require(len(entries) == 13_608, "shard ledger line count changed")
        for entry in entries:
            unit_id = cast(str, entry["unit_id"])
            _require(unit_id not in shard_entries, "duplicate unit across shard ledgers")
            _require(
                cast(int, registry_units[int(unit_id.split("_", 1)[0])]["shard_index"])
                == shard_index,
                "unit stored in wrong shard ledger",
            )
            shard_entries[unit_id] = entry
    _require(shard_entries == canonical_by_id, "canonical and shard ledgers differ")

    jobs: list[tuple[object, ...]] = []
    for registry_unit in registry_units:
        unit_id = cast(str, registry_unit["unit_id"])
        entry = canonical_by_id[unit_id]
        jobs.append(
            (
                str(output_root / "units" / f"{unit_id}.json"),
                entry["unit_result_sha256"],
                _ids_digest(cast(list[str], entry["configuration_ids"])),
                registry_unit,
                expected_executor_sha256,
                configuration,
                expected_stage_a_code_sha256,
            )
        )
    statuses: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    model_units: Counter[str] = Counter()
    unit_bytes = 0
    record_count = 0
    converged_records = 0
    solver_converged_cells = 0
    with ProcessPoolExecutor(max_workers=VERIFICATION_WORKERS) as pool:
        for summary in pool.map(_verify_unit_job, jobs, chunksize=8):
            unit_bytes += cast(int, summary["bytes"])
            record_count += cast(int, summary["records"])
            converged_records += cast(int, summary["converged_records"])
            solver_converged_cells += cast(int, summary["solver_converged_cells"])
            _merge_counts(statuses, summary["statuses"])
            _merge_counts(dispositions, summary["dispositions"])
            model_units.update([cast(str, summary["model_family"])])
    _require(record_count == EXPECTED_CONFIGURATION_EVALUATIONS, "verified record count changed")
    _require(
        model_units == {"continuous_ridge": 20_412, "event_logistic_l2": 20_412},
        "model-family unit coverage changed",
    )

    samples: list[dict[str, object]] = []
    with resource_samples_path.open(encoding="utf-8") as handle:
        samples = [_strict_json_line(line) for line in handle]
    _require(resource_summary == run_state["resource_summary"], "resource summary/state differ")
    _require(resource_summary["sample_count"] == len(samples), "resource sample count changed")
    gpu = [
        float(cast(float, sample["gpu_device_utilization_percent"]))
        for sample in samples
        if sample["gpu_device_utilization_percent"] is not None
    ]
    cpu = [float(cast(float, sample["worker_cpu_percent"])) for sample in samples]
    rss = [int(cast(int, sample["worker_rss_bytes"])) for sample in samples]
    free = [
        float(cast(float, sample["system_memory_free_percent"]))
        for sample in samples
        if sample["system_memory_free_percent"] is not None
    ]
    _assert_float_equal(
        resource_summary["gpu_device_utilization_mean_percent"], fmean(gpu), "GPU mean"
    )
    _assert_float_equal(
        resource_summary["gpu_device_utilization_median_percent"], median(gpu), "GPU median"
    )
    _assert_float_equal(resource_summary["gpu_device_utilization_max_percent"], max(gpu), "GPU max")
    _assert_float_equal(resource_summary["worker_cpu_mean_percent"], fmean(cpu), "CPU mean")
    _assert_float_equal(resource_summary["worker_cpu_max_percent"], max(cpu), "CPU max")
    _require(resource_summary["worker_rss_peak_bytes"] == max(rss), "RSS peak changed")
    _assert_float_equal(resource_summary["system_memory_free_min_percent"], min(free), "memory min")
    _assert_float_equal(
        resource_summary["system_memory_free_median_percent"], median(free), "memory median"
    )
    expected_headroom = int(cast(int, resource_summary["host_memory_bytes"]) * min(free) / 100)
    _require(
        resource_summary["estimated_minimum_memory_headroom_bytes"] == expected_headroom,
        "memory headroom changed",
    )
    _require(fmean(gpu) >= 90.0, "Stage A did not sustain the registered GPU saturation gate")
    _require(expected_headroom >= 6 * 1024**3, "Stage A violated the memory headroom gate")
    for boundary in ("pressure_before", "pressure_after"):
        pressure = cast(dict[str, object], resource_summary[boundary])
        thermal = cast(str, pressure["thermal"])
        power = cast(str, pressure["power"])
        swap = cast(str, pressure["swap"])
        _require("No thermal warning" in thermal, f"thermal gate failed: {boundary}")
        _require("No performance warning" in thermal, f"performance gate failed: {boundary}")
        _require("lowpowermode         0" in power, f"low-power mode enabled: {boundary}")
        _require(" sleep                0" in power, f"system sleep enabled: {boundary}")
        _require("used = 0.00M" in swap, f"swap was used: {boundary}")

    verification = {
        "schema_version": "veatic21_phase02_stage_a_saturated_verification_v1",
        "status": "PASS",
        "verification_workers": VERIFICATION_WORKERS,
        "verifier_source_sha256": sha256_file(Path(__file__)),
        "selected_executor_registration_sha256": sha256_file(SELECTED_EXECUTOR),
        "selected_executor_sha256": expected_executor_sha256,
        "launcher_source_sha256": selected_record["launcher_source_sha256"],
        "request_sha256": sha256_file(request_path),
        "work_unit_registry_sha256": sha256_file(registry_path),
        "run_state_sha256": sha256_file(run_state_path),
        "canonical_ledger_sha256": sha256_file(canonical_ledger_path),
        "resource_summary_sha256": sha256_file(resource_summary_path),
        "resource_samples_sha256": sha256_file(resource_samples_path),
        "work_units": EXPECTED_WORK_UNITS,
        "unique_unit_ids": len(canonical_by_id),
        "shard_ledger_lines": len(shard_entries),
        "canonical_ledger_lines": len(canonical_entries),
        "configuration_evaluations": record_count,
        "unit_result_bytes": unit_bytes,
        "model_family_units": dict(model_units),
        "record_statuses": dict(statuses),
        "record_dispositions": dict(dispositions),
        "converged_records": converged_records,
        "solver_converged_cells": solver_converged_cells,
        "elapsed_seconds": run_state["elapsed_seconds"],
        "work_units_per_second": run_state["work_units_per_second"],
        "gpu_device_utilization_mean_percent": resource_summary[
            "gpu_device_utilization_mean_percent"
        ],
        "estimated_minimum_memory_headroom_bytes": resource_summary[
            "estimated_minimum_memory_headroom_bytes"
        ],
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "gates": {
            "request_identity": "PASS",
            "registry_identity": "PASS",
            "unit_file_coverage": "PASS",
            "unit_hashes": "PASS",
            "unit_schemas": "PASS",
            "unit_provenance": "PASS",
            "configuration_coverage": "PASS",
            "shard_ledgers": "PASS",
            "canonical_ledger": "PASS",
            "resource_recalculation": "PASS",
            "gpu_saturation": "PASS",
            "memory_headroom": "PASS",
            "thermal_power_swap": "PASS",
            "access_firewall": "PASS",
        },
    }
    _write_json(output_root / "verification.json", verification)
    return verification
