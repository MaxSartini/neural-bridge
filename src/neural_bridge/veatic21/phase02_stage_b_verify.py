"""Independent verification for VEATIC Phase 02 Stage B systems and main artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, cast

import numpy as np

from neural_bridge.veatic21.data import load_json, sha256_file
from neural_bridge.veatic21.phase01 import _array_digest
from neural_bridge.veatic21.phase02_metrics import binary_ranking_and_probability_metrics_fast
from neural_bridge.veatic21.phase02_stage_a import _load_inputs
from neural_bridge.veatic21.phase02_stage_b import (
    STAGE_B_BACKTEST_ROOT,
    STAGE_B_EXECUTION_REGISTRATION,
    STAGE_B_MAIN_ROOT,
    StageBWorkUnit,
    candidate_cell_id,
    iter_work_units,
    logical_candidate,
    logical_candidate_id,
    model_seed,
    prepare_stage_b_unit,
    stage_b_code_identity,
)
from neural_bridge.veatic21.phase02_stage_b_executor import (
    SELECTED_STAGE_B_EXECUTOR,
    _atomic_json,
    _registered_cell_pairs,
    _registration,
    _resolve_cells,
    _unit_bundle_paths,
)

VERIFIER_PATH = Path(__file__)


def stage_b_verifier_identity() -> str:
    return sha256_file(VERIFIER_PATH)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checkpoint_from_payload(payload: Any, prefix: str) -> dict[str, np.ndarray]:
    marker = f"{prefix}checkpoint__"
    checkpoint = {
        name[len(marker) :]: payload[name].copy()
        for name in payload.files
        if name.startswith(marker)
    }
    row_name = f"{prefix}validation_row_indices"
    _require(row_name in payload.files, "validation row identities are missing")
    checkpoint["validation_row_indices"] = payload[row_name].copy()
    return checkpoint


def _verify_cell_artifact(
    *,
    unit: StageBWorkUnit,
    candidate: dict[str, Any],
    record: dict[str, Any],
    prediction: np.ndarray,
    checkpoint: dict[str, np.ndarray],
    prepared: Any,
) -> None:
    cell_id = candidate_cell_id(unit, candidate)
    _require(record["candidate_cell_id"] == cell_id, "Stage B cell identity changed")
    _require(record["candidate"] == candidate, "Stage B candidate payload changed")
    _require(
        record["logical_candidate_id"] == logical_candidate_id(candidate, unit),
        "Stage B logical identity changed",
    )
    _require(record["model_seed"] == model_seed(unit, candidate), "Stage B seed changed")
    _require(record["split_sha256"] == unit.split_sha256, "Stage B split changed")
    _require(record["feature_matrix_sha256"] == prepared.feature_matrix_sha256, "features changed")
    _require(record["scaler_sha256"] == prepared.scaler_sha256, "scaler changed")
    _require(
        record["target_thresholds_sha256"] == prepared.target_thresholds_sha256,
        "thresholds changed",
    )
    _require(
        record["checkpoint_sha256"] == _array_digest(checkpoint),
        "Stage B checkpoint hash changed",
    )
    _require(
        np.array_equal(
            checkpoint["validation_row_indices"],
            prepared.validation_indices.astype(np.int32),
        ),
        "Stage B validation row identities changed",
    )
    prediction_digest = _array_digest(
        {
            "validation_indices": prepared.validation_indices.astype(np.int32),
            "prediction": prediction.astype(np.float32),
        }
    )
    _require(
        record["validation_prediction_sha256"] == prediction_digest,
        "Stage B prediction hash changed",
    )
    probability = candidate["family"] != "continuous_ridge"
    metrics = binary_ranking_and_probability_metrics_fast(
        prepared.labels[prepared.validation_indices],
        prediction,
        probability=probability,
    )
    _require(record["metrics"] == metrics, "Stage B metrics do not match predictions")
    solver = cast(dict[str, Any], record["solver"])
    initial = int(candidate["initial_update_budget"])
    maximum = int(
        candidate.get(
            "maximum_convergence_budget",
            candidate.get("undertraining_recovery_maximum_budget", 0),
        )
    )
    _require(initial <= int(solver["iterations"]) <= maximum, "Stage B budget changed")
    _require(record["outer_test_scores_opened"] is False, "outer-test access changed")
    _require(record["cortical_values_opened"] is False, "cortical access changed")
    _require(
        record["prospective_washout_candidates_opened"] is False,
        "washout access changed",
    )


def _coverage(registration: dict[str, Any]) -> dict[str, Any]:
    cells = _resolve_cells(_registered_cell_pairs(registration))
    protocols = {unit.protocol for unit, _ in cells}
    forms = {unit.feature_form for unit, _ in cells}
    regions = {
        (
            "low"
            if unit.history_depth_rows <= 7
            else "mid"
            if unit.history_depth_rows <= 14
            else "high"
        )
        for unit, _ in cells
    }
    families = {str(candidate["family"]) for _, candidate in cells}
    boundary = {
        (str(candidate["family"]), float(candidate["regularization_multiplier"]))
        for _, candidate in cells
        if candidate["family"] in {"continuous_ridge", "event_logistic_l2"}
    }
    elastic_l1 = {
        float(candidate["l1_ratio"])
        for _, candidate in cells
        if candidate["family"] == "event_elastic_net"
    }
    _require(protocols == {"blocked", "grouped"}, "backtest protocol coverage changed")
    _require(len(forms) == 6, "backtest feature-form coverage changed")
    _require(regions == {"low", "mid", "high"}, "backtest history coverage changed")
    _require(
        families
        == {
            "continuous_ridge",
            "event_logistic_l2",
            "event_elastic_net",
            "event_mlp",
            "event_gru",
        },
        "backtest family coverage changed",
    )
    _require(
        boundary
        == {
            ("continuous_ridge", 1e-7),
            ("continuous_ridge", 1e4),
            ("event_logistic_l2", 1e-7),
            ("event_logistic_l2", 1e4),
        },
        "backtest boundary coverage changed",
    )
    _require(elastic_l1 == {0.25, 0.5, 0.75, 1.0}, "elastic L1 coverage changed")
    elastic_regularization = {
        float(candidate["regularization_multiplier"])
        for _, candidate in cells
        if candidate["family"] == "event_elastic_net"
        and float(candidate["regularization_multiplier"]) in {1e-6, 0.1, 1000.0}
    }
    _require(
        elastic_regularization == {1e-6, 0.1, 1000.0},
        "elastic regularization coverage changed",
    )
    all_units = list(iter_work_units())
    selected_train_rows = {unit.train_rows for unit, _ in cells}
    selected_feature_counts = {unit.feature_count for unit, _ in cells}
    _require(
        {min(unit.train_rows for unit in all_units), max(unit.train_rows for unit in all_units)}
        <= selected_train_rows,
        "training-row extremes are not covered",
    )
    _require(
        {
            min(unit.feature_count for unit in all_units),
            max(unit.feature_count for unit in all_units),
        }
        <= selected_feature_counts,
        "feature-count extremes are not covered",
    )
    _require(
        {1, 21} <= {unit.history_depth_rows for unit, _ in cells},
        "history-depth extremes are not covered",
    )
    neural_axes: dict[str, dict[str, set[Any]]] = {}
    for unit, candidate in cells:
        family = str(candidate["family"])
        if family not in {"event_mlp", "event_gru"}:
            continue
        logical = logical_candidate(candidate, unit)
        axes = neural_axes.setdefault(
            family,
            {
                "layers": set(),
                "activation": set(),
                "dropout": set(),
                "optimizer": set(),
                "learning_rate_factor": set(),
                "batch_factor": set(),
                "width_factor": set(),
            },
        )
        for name in axes:
            value = logical[name]
            axes[name].add(round(float(value), 6) if isinstance(value, float) else value)
    mlp = neural_axes["event_mlp"]
    gru = neural_axes["event_gru"]
    _require(mlp["layers"] == {1, 2, 3}, "MLP layer coverage changed")
    _require(mlp["activation"] == {"relu", "gelu", "tanh"}, "MLP activation changed")
    _require(mlp["optimizer"] == {"adamw", "sgd_nesterov"}, "MLP optimizer changed")
    _require(mlp["learning_rate_factor"] == {0.25, 1.0, 4.0, 16.0}, "MLP LR changed")
    _require(mlp["batch_factor"] == {0.5, 1.0, 2.0}, "MLP batch coverage changed")
    _require(mlp["width_factor"] == {0.5, 1.0, 2.0, 4.0}, "MLP width changed")
    _require(0.0 in mlp["dropout"] and len(mlp["dropout"]) > 1, "MLP dropout changed")
    _require(gru["layers"] == {1, 2}, "GRU layer coverage changed")
    _require(gru["optimizer"] == {"adamw", "sgd_nesterov"}, "GRU optimizer changed")
    _require(gru["learning_rate_factor"] == {0.25, 1.0, 4.0, 16.0}, "GRU LR changed")
    _require(gru["batch_factor"] == {0.5, 1.0, 2.0}, "GRU batch coverage changed")
    _require(gru["width_factor"] == {0.5, 1.0, 2.0, 4.0}, "GRU width changed")
    _require(0.0 in gru["dropout"] and len(gru["dropout"]) > 1, "GRU dropout changed")
    gru_depths = [
        unit.history_depth_rows for unit, candidate in cells if candidate["family"] == "event_gru"
    ]
    _require(max(gru_depths) == 19, "maximum registered GRU depth is not covered")
    return {
        "cells": len(cells),
        "protocols": sorted(protocols),
        "feature_forms": sorted(forms),
        "history_regions": sorted(regions),
        "families": sorted(families),
        "maximum_gru_depth": max(gru_depths),
        "minimum_train_rows": min(selected_train_rows),
        "maximum_train_rows": max(selected_train_rows),
        "minimum_feature_count": min(selected_feature_counts),
        "maximum_feature_count": max(selected_feature_counts),
        "neural_axes": {
            family: {name: sorted(values) for name, values in axes.items()}
            for family, axes in neural_axes.items()
        },
    }


def _resource_eligible(value: dict[str, Any], registration: dict[str, Any]) -> bool:
    gates = cast(dict[str, Any], registration["resource_gates"])
    resources = cast(dict[str, Any], value["resources"])
    before = cast(dict[str, Any], resources["pressure_before"])
    after = cast(dict[str, Any], resources["pressure_after"])
    headroom = resources.get("estimated_minimum_memory_headroom_bytes")
    gpu = resources.get("gpu_device_utilization_max_percent")
    return (
        value["resume"]["status"] == "PASS"
        and int(resources.get("sample_count", 0)) > 0
        and headroom is not None
        and int(headroom) >= int(gates["minimum_memory_headroom_bytes"])
        and "used = 0.00M" in str(before.get("swap", ""))
        and "used = 0.00M" in str(after.get("swap", ""))
        and "No thermal warning" in str(after.get("thermal", ""))
        and "No performance warning" in str(after.get("thermal", ""))
        and gpu is not None
        and float(gpu) > 0
        and value.get("power_source") == "AC Power"
        and value.get("low_power_mode") == 0
    )


def verify_stage_b_executor_backtest() -> dict[str, Any]:
    """Re-derive registration coverage, every cell artifact, and topology selection."""

    registration = _registration()
    request = load_json(STAGE_B_BACKTEST_ROOT / "request.json")
    result = load_json(STAGE_B_BACKTEST_ROOT / "result.json")
    _require(
        request["stage_b_execution_registration_sha256"]
        == sha256_file(STAGE_B_EXECUTION_REGISTRATION),
        "Stage B registration hash changed",
    )
    _require(request["stage_b_code_sha256"] == stage_b_code_identity(), "Stage B code changed")
    coverage = _coverage(registration)
    resolved = {
        candidate_cell_id(unit, candidate): (unit, candidate)
        for unit, candidate in _resolve_cells(_registered_cell_pairs(registration))
    }
    inputs = _load_inputs()
    prepared_cache: dict[int, Any] = {}
    identities: set[str] = set()
    topology_rows: list[dict[str, Any]] = []
    expected_repetitions = int(request["repetitions"])
    for topology in cast(list[dict[str, int]], request["topologies"]):
        topology_id = (
            f"cpu{int(topology['cpu_preparation_workers']):02d}_"
            f"mlx{int(topology['mlx_stream_lanes']):02d}"
        )
        rates: list[float] = []
        eligible = True
        for repetition in range(expected_repetitions):
            root = STAGE_B_BACKTEST_ROOT / "topologies" / topology_id / f"repeat-{repetition:02d}"
            row = load_json(root / "result.json")
            _require(row["topology"] == topology, "topology payload changed")
            _require(row["cells"] == len(resolved), "topology cell count changed")
            records: list[dict[str, Any]] = []
            ledger_path = root / "append-only-ledger.jsonl"
            _require(ledger_path.is_file(), "backtest append-only ledger is missing")
            ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
            _require(
                ledger_rows == row["cell_manifests"],
                "backtest append-only ledger changed",
            )
            _require(
                row["append_only_ledger_lines"] == len(ledger_rows),
                "backtest ledger line count changed",
            )
            _require(
                row["append_only_ledger_sha256"] == sha256_file(ledger_path),
                "backtest ledger hash changed",
            )
            for manifest in cast(list[dict[str, Any]], row["cell_manifests"]):
                cell_id = str(manifest["candidate_cell_id"])
                _require(cell_id in resolved, "unregistered backtest cell")
                record_path = root / "cells" / f"{cell_id}.json"
                artifact_path = root / "cells" / f"{cell_id}.npz"
                _require(
                    sha256_file(record_path) == manifest["record_sha256"],
                    "backtest record hash changed",
                )
                _require(
                    sha256_file(artifact_path) == manifest["artifact_sha256"],
                    "backtest artifact hash changed",
                )
                record = load_json(record_path)
                unit, candidate = resolved[cell_id]
                if unit.sequence not in prepared_cache:
                    prepared_cache[unit.sequence] = prepare_stage_b_unit(unit, inputs=inputs)
                with np.load(artifact_path, allow_pickle=False) as payload:
                    prediction = payload["validation_prediction"].copy()
                    checkpoint = _checkpoint_from_payload(payload, "")
                _verify_cell_artifact(
                    unit=unit,
                    candidate=candidate,
                    record=record,
                    prediction=prediction,
                    checkpoint=checkpoint,
                    prepared=prepared_cache[unit.sequence],
                )
                records.append(record)
            normalized = hashlib.sha256(
                json.dumps(
                    sorted(
                        (
                            {key: item for key, item in record.items() if key != "runtime_seconds"}
                            for record in records
                        ),
                        key=lambda item: item["candidate_cell_id"],
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            _require(normalized == row["normalized_evidence_sha256"], "normalized hash changed")
            identities.add(normalized)
            rates.append(float(row["cells_per_second"]))
            eligible = eligible and _resource_eligible(row, registration)
        topology_rows.append(
            {
                "topology": topology,
                "topology_id": topology_id,
                "median_cells_per_second": statistics.median(rates),
                "minimum_cells_per_second": min(rates),
                "maximum_cells_per_second": max(rates),
                "eligible": eligible,
            }
        )
    _require(len(identities) == 1, "topologies did not produce identical evidence")
    _require(result["topology_summaries"] == topology_rows, "topology summaries changed")
    eligible_rows = [row for row in topology_rows if row["eligible"]]
    _require(bool(eligible_rows), "no independently eligible Stage B topology")
    fastest = max(float(row["median_cells_per_second"]) for row in eligible_rows)
    plateau = [
        row for row in eligible_rows if float(row["median_cells_per_second"]) >= 0.97 * fastest
    ]
    selected = min(
        plateau,
        key=lambda row: (
            int(row["topology"]["mlx_stream_lanes"]),
            int(row["topology"]["cpu_preparation_workers"]),
            str(row["topology_id"]),
        ),
    )
    _require(result["selected_topology"] == selected, "selected Stage B topology changed")
    verification = {
        "schema_version": "veatic21_phase02_stage_b_backtest_verification_v1",
        "status": "PASS",
        "request_sha256": sha256_file(STAGE_B_BACKTEST_ROOT / "request.json"),
        "result_sha256": sha256_file(STAGE_B_BACKTEST_ROOT / "result.json"),
        "registration_sha256": sha256_file(STAGE_B_EXECUTION_REGISTRATION),
        "stage_b_code_sha256": stage_b_code_identity(),
        "verifier_code_sha256": stage_b_verifier_identity(),
        "coverage": coverage,
        "topology_repetitions_verified": len(topology_rows) * expected_repetitions,
        "cell_artifacts_verified": len(topology_rows) * expected_repetitions * len(resolved),
        "normalized_evidence_sha256": next(iter(identities)),
        "selected_topology": selected,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    _atomic_json(STAGE_B_BACKTEST_ROOT / "verification.json", verification)
    return verification


def _read_unit_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            _require(isinstance(value, dict), "invalid Stage B main ledger row")
            rows.append(cast(dict[str, Any], value))
    return rows


def verify_stage_b_main() -> dict[str, Any]:
    """Exhaustively verify exact main-registry coverage, metrics, and artifacts."""

    _require(SELECTED_STAGE_B_EXECUTOR.is_file(), "selected Stage B executor is missing")
    request = load_json(STAGE_B_MAIN_ROOT / "request.json")
    summary = load_json(STAGE_B_MAIN_ROOT / "summary.json")
    _require(request["stage_b_code_sha256"] == stage_b_code_identity(), "Stage B code changed")
    units = list(iter_work_units())
    _require(request["work_units"] == len(units) == 40_824, "main work-unit count changed")
    inputs = _load_inputs()
    cells = 0
    eligible = 0
    invalid = 0
    protected = 0
    manifests: list[dict[str, Any]] = []
    for unit in units:
        records_path, artifacts_path, manifest_path = _unit_bundle_paths(
            STAGE_B_MAIN_ROOT, unit.sequence
        )
        manifest = load_json(manifest_path)
        _require(manifest["work_unit_id"] == unit.work_unit_id, "main work unit changed")
        _require(manifest["records_sha256"] == sha256_file(records_path), "main records changed")
        _require(
            manifest["artifacts_sha256"] == sha256_file(artifacts_path),
            "main artifacts changed",
        )
        records = _read_unit_records(records_path)
        _require(len(records) == len(unit.candidates), "main candidate coverage changed")
        prepared = prepare_stage_b_unit(unit, inputs=inputs)
        record_by_id = {str(record["candidate_cell_id"]): record for record in records}
        with np.load(artifacts_path, allow_pickle=False) as payload:
            for candidate in unit.candidates:
                cell_id = candidate_cell_id(unit, candidate)
                _require(cell_id in record_by_id, "main candidate is missing")
                prediction = payload[f"{cell_id}__prediction"].copy()
                checkpoint = _checkpoint_from_payload(payload, f"{cell_id}__")
                record = record_by_id[cell_id]
                _verify_cell_artifact(
                    unit=unit,
                    candidate=candidate,
                    record=record,
                    prediction=prediction,
                    checkpoint=checkpoint,
                    prepared=prepared,
                )
                disposition = str(record["solver"]["disposition"])
                eligible += disposition == "eligible_for_stage_b_aggregation"
                invalid += disposition == "invalid_incomplete_not_negative"
                protected += disposition == "valid_unplateaued_at_2b_protected_from_negative_claim"
                cells += 1
        manifests.append(manifest)
    _require(cells == 2_351_229, "main Stage B cell count changed")
    identity = hashlib.sha256(
        json.dumps(
            sorted(manifests, key=lambda value: value["work_unit_sequence"]),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    _require(identity == summary["manifest_identity_sha256"], "main manifest identity changed")
    canonical_ledger = STAGE_B_MAIN_ROOT / "append-only-experiment-ledger.jsonl"
    _require(
        summary["canonical_ledger_sha256"] == sha256_file(canonical_ledger),
        "main canonical ledger changed",
    )
    _require(summary["ledger_lines"] == len(units), "main ledger coverage changed")
    verification = {
        "schema_version": "veatic21_phase02_stage_b_main_verification_v1",
        "status": "PASS",
        "request_sha256": sha256_file(STAGE_B_MAIN_ROOT / "request.json"),
        "summary_sha256": sha256_file(STAGE_B_MAIN_ROOT / "summary.json"),
        "stage_b_code_sha256": stage_b_code_identity(),
        "verifier_code_sha256": stage_b_verifier_identity(),
        "work_units": len(units),
        "candidate_cells": cells,
        "eligible_cells": eligible,
        "invalid_incomplete_not_negative_cells": invalid,
        "valid_unplateaued_protected_cells": protected,
        "manifest_identity_sha256": identity,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "prospective_washout_candidates_opened": False,
    }
    _atomic_json(STAGE_B_MAIN_ROOT / "verification.json", verification)
    return verification
