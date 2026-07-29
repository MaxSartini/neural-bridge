"""Exhaustive verification for the complete hardware-saturated Stage A rescue."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import cast

from neural_bridge.veatic21.contracts import (
    PHASE02_REGISTRATION_SHA256,
    PHASE02_STAGE_A_SATURATED_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json
from neural_bridge.veatic21.phase02_stage_a import _stage_a_code_identity
from neural_bridge.veatic21.phase02_stage_a_executor import _resource_summary
from neural_bridge.veatic21.phase02_stage_a_rescue import (
    RescueCell,
    RescueUnit,
    load_rescue_registry,
    rescue_solver_code_identity,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_executor import (
    deterministic_weighted_shards,
    rescue_executor_code_identity,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_registration import (
    RESCUE_CELL_REGISTRY,
    RESCUE_UNIT_REGISTRY,
)
from neural_bridge.veatic21.phase02_stage_a_rescue_saturated import (
    RESCUE_MAIN_ROOT,
    verify_selected_rescue_executor,
)

EXPECTED_RESCUE_UNITS = 14_465
EXPECTED_RESCUE_CELLS = 113_392
MINIMUM_MEMORY_HEADROOM_BYTES = 6 * 1024**3
VERIFICATION_WORKERS = 8
EXPECTED_UNIT_KEYS = {
    "aggregation_or_pruning_performed",
    "cell_count",
    "cortical_values_opened",
    "execution_provenance",
    "feature_matrix_sha256",
    "outer_test_scores_opened",
    "records",
    "rescue_cell_registry_sha256",
    "rescue_solver_code_sha256",
    "rescue_unit",
    "rescue_unit_registry_sha256",
    "runtime_seconds",
    "scaler_sha256",
    "schema_version",
    "scientific_registration_sha256",
    "split_sha256",
    "stage_a_solver_code_sha256",
    "stage_a_unit",
    "target_thresholds_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_bytes(payload: bytes) -> dict[str, object]:
    value = json.loads(payload, parse_constant=_reject_nonfinite)
    _require(isinstance(value, dict), "expected JSON object")
    return cast(dict[str, object], value)


def _strict_json_line(line: str) -> dict[str, object]:
    value = json.loads(line, parse_constant=_reject_nonfinite)
    _require(isinstance(value, dict), "expected JSONL object")
    return cast(dict[str, object], value)


def _expected_rescue_unit(unit: RescueUnit) -> dict[str, object]:
    return {
        **asdict(unit),
        "cells": [cell.rescue_cell_identity_sha256 for cell in unit.cells],
    }


def _expected_static_record(cell: RescueCell) -> dict[str, object]:
    return {
        "rescue_sequence": cell.rescue_sequence,
        "rescue_cell_identity_sha256": cell.rescue_cell_identity_sha256,
        "original_configuration_id": cell.original_configuration_id,
        "original_unit_id": cell.original_unit_id,
        "original_unit_result_sha256": cell.original_unit_result_sha256,
        "model_family": cell.model_family,
        "candidate_id": cell.candidate_id,
        "feature_form": cell.feature_form,
        "history_depth_rows": cell.history_depth,
        "history_depth_seconds": cell.history_depth / 2,
        "regularization_index": cell.regularization_index,
        "regularization_multiplier": cell.regularization_multiplier,
        "regularization_scale": cell.regularization_scale,
        "regularization_value": cell.regularization_value,
        "train_threshold_q90": cell.train_threshold_q90,
        "train_rows": cell.train_rows,
        "validation_rows": cell.validation_rows,
        "original_base_budget": cell.original_base_budget,
        "original_maximum_budget": cell.original_maximum_budget,
        "rescue_maximum_budget": cell.rescue_maximum_budget,
        "convergence_tolerance": cell.convergence_tolerance,
        "original_solver_diagnostic": cell.original_solver_diagnostic,
    }


def _verify_record(record: dict[str, object], cell: RescueCell, path: Path) -> str:
    _require(
        record.get("schema_version") == "veatic21_phase02_stage_a_rescue_cell_result_v1",
        f"cell schema changed: {path}",
    )
    for name, expected in _expected_static_record(cell).items():
        _require(record.get(name) == expected, f"cell {name} changed: {path}")
    converged = record.get("converged")
    _require(isinstance(converged, bool), f"cell convergence flag changed: {path}")
    expected_status = "completed" if converged else "invalid"
    expected_disposition = (
        "eligible_for_inner_aggregation"
        if converged
        else "invalid_nonconverged_after_registered_maximum_budget"
    )
    _require(record.get("status") == expected_status, f"cell status changed: {path}")
    _require(
        record.get("disposition") == expected_disposition,
        f"cell disposition changed: {path}",
    )
    iterations = record.get("iterations")
    _require(
        isinstance(iterations, int)
        and not isinstance(iterations, bool)
        and cell.original_maximum_budget <= iterations <= cell.rescue_maximum_budget
        and iterations % 8 == 0,
        f"cell iteration count changed: {path}",
    )
    iterations_value = cast(int, iterations)
    if not converged:
        _require(
            iterations_value == cell.rescue_maximum_budget,
            f"invalid cell stopped before registered maximum: {path}",
        )
    diagnostic_name = (
        "relative_residual" if cell.model_family == "continuous_ridge" else "relative_gradient"
    )
    _require(record.get("diagnostic_name") == diagnostic_name, f"diagnostic changed: {path}")
    diagnostic = record.get("final_diagnostic")
    _require(
        isinstance(diagnostic, int | float)
        and not isinstance(diagnostic, bool)
        and math.isfinite(float(diagnostic))
        and float(diagnostic) >= 0,
        f"non-finite cell diagnostic: {path}",
    )
    diagnostic_value = float(cast(int | float, diagnostic))
    tolerance = cell.convergence_tolerance
    if converged:
        _require(diagnostic_value <= tolerance + 1e-7, f"false convergence: {path}")
    else:
        _require(diagnostic_value > tolerance - 1e-7, f"false nonconvergence: {path}")

    curve = cast(list[dict[str, object]], record.get("learning_curve"))
    updates = [item.get("update") for item in curve]
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in updates),
        f"learning-curve updates changed: {path}",
    )
    expected_updates = {
        value
        for value in (
            cell.original_maximum_budget,
            2 * cell.original_base_budget,
            4 * cell.original_base_budget,
            8 * cell.original_base_budget,
            12 * cell.original_base_budget,
            cell.rescue_maximum_budget,
        )
        if value <= iterations_value
    }
    if converged:
        expected_updates.add(iterations_value)
    _require(updates == sorted(expected_updates), f"learning-curve coverage changed: {path}")
    for point in curve:
        _require(
            set(point) == {"update", diagnostic_name},
            f"learning curve schema changed: {path}",
        )
        value = point[diagnostic_name]
        _require(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0,
            f"learning curve contains non-finite value: {path}",
        )

    rows = record.get("rows")
    positives = record.get("positives")
    negatives = record.get("negatives")
    _require(rows == cell.validation_rows, f"validation row ownership changed: {path}")
    _require(
        isinstance(positives, int)
        and not isinstance(positives, bool)
        and isinstance(negatives, int)
        and not isinstance(negatives, bool)
        and isinstance(rows, int)
        and not isinstance(rows, bool)
        and positives + negatives == rows,
        f"event support changed: {path}",
    )
    positives_value = cast(int, positives)
    rows_value = cast(int, rows)
    prevalence = record.get("prevalence")
    _require(
        isinstance(prevalence, int | float)
        and math.isclose(
            float(prevalence), positives_value / rows_value, rel_tol=0, abs_tol=1e-15
        ),
        f"prevalence changed: {path}",
    )
    for metric in ("raw_pr_auc", "roc_auc"):
        value = record.get(metric)
        _require(
            value is None
            or (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0 <= float(value) <= 1
            ),
            f"invalid {metric}: {path}",
        )
    brier = record.get("brier")
    if cell.model_family == "continuous_ridge":
        _require(brier is None, f"ridge cell has probability metric: {path}")
    else:
        _require(
            isinstance(brier, int | float)
            and not isinstance(brier, bool)
            and math.isfinite(float(brier))
            and 0 <= float(brier) <= 1,
            f"invalid logistic Brier score: {path}",
        )
    return cast(str, expected_disposition)


def _verify_unit_job(job: tuple[object, ...]) -> dict[str, object]:
    (
        path_text,
        expected_unit_hash,
        unit,
        shard_index,
        expected_configuration,
        expected_executor_sha256,
        expected_solver_sha256,
    ) = job
    path = Path(cast(str, path_text))
    expected = cast(RescueUnit, unit)
    payload = path.read_bytes()
    _require(
        hashlib.sha256(payload).hexdigest() == expected_unit_hash,
        f"rescue unit hash mismatch: {path}",
    )
    result = _strict_json_bytes(payload)
    _require(set(result) == EXPECTED_UNIT_KEYS, f"rescue unit schema changed: {path}")
    _require(
        result.get("schema_version") == "veatic21_phase02_stage_a_rescue_unit_result_v1",
        f"rescue unit version changed: {path}",
    )
    _require(result.get("rescue_unit") == _expected_rescue_unit(expected), f"unit changed: {path}")
    _require(result.get("cell_count") == len(expected.cells), f"cell count changed: {path}")
    _require(
        result.get("scientific_registration_sha256") == PHASE02_REGISTRATION_SHA256,
        f"scientific registration changed: {path}",
    )
    _require(
        result.get("stage_a_solver_code_sha256") == _stage_a_code_identity(),
        f"Stage A solver identity changed: {path}",
    )
    _require(
        result.get("rescue_solver_code_sha256") == expected_solver_sha256,
        f"rescue solver identity changed: {path}",
    )
    _require(
        result.get("rescue_cell_registry_sha256") == sha256_file(RESCUE_CELL_REGISTRY),
        f"cell registry changed: {path}",
    )
    _require(
        result.get("rescue_unit_registry_sha256") == sha256_file(RESCUE_UNIT_REGISTRY),
        f"unit registry changed: {path}",
    )
    for firewall in (
        "outer_test_scores_opened",
        "cortical_values_opened",
        "aggregation_or_pruning_performed",
    ):
        _require(result.get(firewall) is False, f"unit violated {firewall}: {path}")
    runtime = result.get("runtime_seconds")
    _require(
        isinstance(runtime, int | float) and math.isfinite(float(runtime)) and runtime >= 0,
        f"unit runtime changed: {path}",
    )

    original_path = PHASE02_STAGE_A_SATURATED_ROOT / "units" / f"{expected.original_unit_id}.json"
    original_payload = original_path.read_bytes()
    _require(
        hashlib.sha256(original_payload).hexdigest() == expected.original_unit_result_sha256,
        f"immutable Stage A link changed: {original_path}",
    )
    original = _strict_json_bytes(original_payload)
    _require(
        result.get("stage_a_unit") == original.get("unit"),
        f"linked Stage A unit identity changed: {path}",
    )
    for name in (
        "split_sha256",
        "feature_matrix_sha256",
        "scaler_sha256",
        "target_thresholds_sha256",
    ):
        _require(result.get(name) == original.get(name), f"linked {name} changed: {path}")

    configuration = cast(dict[str, object], expected_configuration)
    provenance = {
        "schema_version": "veatic21_phase02_stage_a_rescue_execution_v1",
        "executor_sha256": expected_executor_sha256,
        "configuration_id": configuration["id"],
        "mlx_lanes": configuration["mlx_lanes"],
        "gpu_streams_per_lane": configuration["gpu_streams_per_lane"],
        "metric_workers_per_lane": configuration["metric_workers_per_lane"],
        "cell_batch_size": configuration["cell_batch_size"],
        "compiled_update_blocks": configuration["compiled_update_blocks"],
        "fast_metrics": configuration["fast_metrics"],
        "pipeline_depth": configuration["pipeline_depth"],
        "shard_index": shard_index,
        "warmup": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    _require(result.get("execution_provenance") == provenance, f"provenance changed: {path}")
    records = cast(list[dict[str, object]], result.get("records"))
    _require(len(records) == len(expected.cells), f"record count changed: {path}")
    dispositions = Counter(
        _verify_record(record, cell, path)
        for record, cell in zip(records, expected.cells, strict=True)
    )
    identities = [cast(str, record["rescue_cell_identity_sha256"]) for record in records]
    _require(len(identities) == len(set(identities)), f"duplicate cells within unit: {path}")
    return {
        "rescue_unit_sequence": expected.rescue_unit_sequence,
        "rescue_cells": len(records),
        "rescue_unit_bytes": len(payload),
        "linked_stage_a_bytes": len(original_payload),
        "dispositions": dict(dispositions),
        "model_family": expected.model_family,
        "cell_identities": identities,
    }


def _read_ledger(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(_strict_json_line(line))
    return rows


def _thermal_pass(resources: dict[str, object]) -> bool:
    after = cast(dict[str, object], resources.get("pressure_after"))
    thermal = str(after.get("thermal", ""))
    return (
        "No thermal warning level has been recorded" in thermal
        and "No performance warning level has been recorded" in thermal
    )


def verify_phase02_stage_a_rescue_saturated_output(
    *, output_root: Path = RESCUE_MAIN_ROOT
) -> dict[str, object]:
    """Hash and structurally verify every full-rescue artifact and ledger entry."""

    output_root = reject_forbidden_runtime_path(output_root)
    _require(output_root == RESCUE_MAIN_ROOT, "complete rescue verification root changed")
    selected = verify_selected_rescue_executor()
    configuration = cast(dict[str, object], selected["configuration"])
    executor_sha256 = rescue_executor_code_identity()
    solver_sha256 = rescue_solver_code_identity()
    registry = load_rescue_registry()
    _require(len(registry) == EXPECTED_RESCUE_UNITS, "rescue unit registry count changed")
    _require(
        sum(len(unit.cells) for unit in registry) == EXPECTED_RESCUE_CELLS,
        "rescue cell registry count changed",
    )

    request_path = output_root / "request.json"
    work_registry_path = output_root / "work-unit-registry.json"
    state_path = output_root / "run-state.json"
    canonical_ledger_path = output_root / "append-only-experiment-ledger.jsonl"
    resource_summary_path = output_root / "resource-summary.json"
    resource_samples_path = output_root / "resource-samples.jsonl"
    request = load_json(request_path)
    work_registry = load_json(work_registry_path)
    state = load_json(state_path)
    resources = load_json(resource_summary_path)

    _require(state.get("status") == "COMPLETE", "complete rescue is not finished")
    _require(sha256_file(request_path) == selected["main_request_sha256"], "request changed")
    _require(request.get("configuration") == configuration, "main configuration changed")
    _require(request.get("rescue_units") == EXPECTED_RESCUE_UNITS, "request unit count changed")
    _require(request.get("rescue_cells") == EXPECTED_RESCUE_CELLS, "request cell count changed")
    for value in (request, state):
        _require(value.get("executor_sha256") == executor_sha256, "executor identity changed")
        for firewall in (
            "outer_test_scores_opened",
            "cortical_values_opened",
            "aggregation_or_pruning_performed",
        ):
            _require(value.get(firewall) is False, f"main run violated {firewall}")
    _require(state.get("rescue_units_total") == EXPECTED_RESCUE_UNITS, "run total changed")
    _require(state.get("rescue_units_completed") == EXPECTED_RESCUE_UNITS, "run incomplete")
    _require(state.get("rescue_cells") == EXPECTED_RESCUE_CELLS, "run cell count changed")
    _require(state.get("request_sha256") == sha256_file(request_path), "state request changed")
    _require(
        state.get("work_unit_registry_sha256") == sha256_file(work_registry_path),
        "state work registry changed",
    )
    _require(
        state.get("canonical_ledger_sha256") == sha256_file(canonical_ledger_path),
        "state canonical ledger changed",
    )
    _require(state.get("resource_summary") == resources, "resource summaries diverged")

    shards = deterministic_weighted_shards(registry, _integer(configuration["mlx_lanes"]))
    expected_work_units = [
        {
            "rescue_unit_sequence": unit.rescue_unit_sequence,
            "original_unit_id": unit.original_unit_id,
            "original_unit_result_sha256": unit.original_unit_result_sha256,
            "cell_count": len(unit.cells),
            "shard_index": shard_index,
        }
        for shard_index, shard in enumerate(shards)
        for unit in shard
    ]
    _require(work_registry.get("executor_sha256") == executor_sha256, "work executor changed")
    _require(work_registry.get("units") == expected_work_units, "work sharding changed")
    work_by_sequence = {
        cast(int, item["rescue_unit_sequence"]): item for item in expected_work_units
    }

    shard_ledgers: list[dict[str, object]] = []
    for shard_index, shard in enumerate(shards):
        shard_root = output_root / "shards" / f"shard-{shard_index:02d}"
        shard_state = load_json(shard_root / "shard-state.json")
        _require(shard_state.get("status") == "COMPLETE", f"shard {shard_index} incomplete")
        _require(
            shard_state.get("rescue_units_total") == len(shard)
            and shard_state.get("rescue_units_completed") == len(shard),
            f"shard {shard_index} coverage changed",
        )
        _require(
            shard_state.get("rescue_cells") == sum(len(unit.cells) for unit in shard),
            f"shard {shard_index} cell count changed",
        )
        ledger_path = shard_root / "append-only-ledger.jsonl"
        _require(
            shard_state.get("ledger_sha256") == sha256_file(ledger_path),
            f"shard {shard_index} ledger changed",
        )
        rows = _read_ledger(ledger_path)
        _require(len(rows) == len(shard), f"shard {shard_index} ledger coverage changed")
        shard_ledgers.extend(rows)

    canonical = _read_ledger(canonical_ledger_path)
    expected_sorted = sorted(
        shard_ledgers, key=lambda item: cast(int, item["rescue_unit_sequence"])
    )
    _require(canonical == expected_sorted, "canonical and shard ledgers diverged")
    sequences = [cast(int, item["rescue_unit_sequence"]) for item in canonical]
    _require(sequences == list(range(EXPECTED_RESCUE_UNITS)), "canonical unit order changed")
    _require(state.get("ledger_lines") == EXPECTED_RESCUE_UNITS, "ledger lines changed")
    _require(state.get("unique_rescue_units") == EXPECTED_RESCUE_UNITS, "unit uniqueness changed")
    _require(state.get("unique_rescue_cells") == EXPECTED_RESCUE_CELLS, "cell uniqueness changed")

    jobs: list[tuple[object, ...]] = []
    for unit, entry in zip(registry, canonical, strict=True):
        work = work_by_sequence[unit.rescue_unit_sequence]
        expected_path = output_root / "units" / (
            f"{unit.rescue_unit_sequence:05d}_{unit.original_unit_id}.json"
        )
        _require(entry.get("unit_result_path") == str(expected_path), "ledger path changed")
        _require(entry.get("cell_count") == len(unit.cells), "ledger cell count changed")
        _require(entry.get("original_unit_id") == unit.original_unit_id, "ledger unit changed")
        _require(
            entry.get("original_unit_result_sha256") == unit.original_unit_result_sha256,
            "ledger Stage A link changed",
        )
        for firewall in (
            "outer_test_scores_opened",
            "cortical_values_opened",
            "aggregation_or_pruning_performed",
        ):
            _require(entry.get(firewall) is False, f"ledger violated {firewall}")
        jobs.append(
            (
                str(expected_path),
                entry["unit_result_sha256"],
                unit,
                work["shard_index"],
                configuration,
                executor_sha256,
                solver_sha256,
            )
        )
    unit_paths = list((output_root / "units").glob("*.json"))
    _require(len(unit_paths) == EXPECTED_RESCUE_UNITS, "unexpected rescue unit file count")

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=VERIFICATION_WORKERS, mp_context=context) as pool:
        verified_units = list(pool.map(_verify_unit_job, jobs, chunksize=16))
    verified_units.sort(key=lambda item: cast(int, item["rescue_unit_sequence"]))
    cell_identities = [
        identity
        for item in verified_units
        for identity in cast(list[str], item["cell_identities"])
    ]
    _require(len(cell_identities) == EXPECTED_RESCUE_CELLS, "verified cell count changed")
    _require(len(set(cell_identities)) == EXPECTED_RESCUE_CELLS, "duplicate rescue cells")
    expected_cell_ids = [
        cell.rescue_cell_identity_sha256 for unit in registry for cell in unit.cells
    ]
    _require(cell_identities == expected_cell_ids, "verified cell order/identity changed")

    samples = _read_ledger(resource_samples_path)
    _require(len(samples) == resources.get("sample_count"), "resource sample count changed")
    recomputed = _resource_summary(cast(list[dict[str, float | int | None]], samples))
    for name, value in recomputed.items():
        _require(resources.get(name) == value, f"resource summary {name} changed")
    headroom = resources.get("estimated_minimum_memory_headroom_bytes")
    _require(
        isinstance(headroom, int) and headroom >= MINIMUM_MEMORY_HEADROOM_BYTES,
        "main rescue memory headroom gate failed",
    )
    gpu = resources.get("gpu_device_utilization_mean_percent")
    _require(
        isinstance(gpu, int | float) and float(gpu) >= 90,
        "main rescue GPU saturation gate failed",
    )
    _require(_thermal_pass(resources), "main rescue thermal/performance warning gate failed")

    dispositions: Counter[str] = Counter()
    families: Counter[str] = Counter()
    for item in verified_units:
        dispositions.update(cast(dict[str, int], item["dispositions"]))
        families[cast(str, item["model_family"])] += cast(int, item["rescue_cells"])
    verification = {
        "schema_version": "veatic21_phase02_stage_a_rescue_saturated_verification_v1",
        "status": "PASS",
        "request_sha256": sha256_file(request_path),
        "work_unit_registry_sha256": sha256_file(work_registry_path),
        "canonical_ledger_sha256": sha256_file(canonical_ledger_path),
        "run_state_sha256": sha256_file(state_path),
        "resource_summary_sha256": sha256_file(resource_summary_path),
        "resource_samples_sha256": sha256_file(resource_samples_path),
        "rescue_units": EXPECTED_RESCUE_UNITS,
        "rescue_cells": EXPECTED_RESCUE_CELLS,
        "unique_rescue_cell_identities": len(set(cell_identities)),
        "dispositions": dict(dispositions),
        "model_family_cells": dict(families),
        "rescue_unit_bytes": sum(cast(int, item["rescue_unit_bytes"]) for item in verified_units),
        "linked_stage_a_bytes_hashed": sum(
            cast(int, item["linked_stage_a_bytes"]) for item in verified_units
        ),
        "gpu_device_utilization_mean_percent": gpu,
        "estimated_minimum_memory_headroom_bytes": headroom,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "aggregation_or_pruning_performed": False,
    }
    _write_json(output_root / "verification.json", verification)
    return verification


def _integer(value: object) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), "expected integer")
    return cast(int, value)
