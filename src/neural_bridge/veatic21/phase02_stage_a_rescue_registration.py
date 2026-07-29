"""Freeze and verify the exact Phase 02 Stage A undertrained-cell rescue registry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import cast

from neural_bridge.veatic21.contracts import (
    PHASE02_REGISTRATION_ROOT,
    PHASE02_STAGE_A_SATURATED_ROOT,
    REPOSITORY_ROOT,
)
from neural_bridge.veatic21.data import load_json, reject_forbidden_runtime_path, sha256_file
from neural_bridge.veatic21.phase00 import _write_json

RESCUE_ROOT = PHASE02_STAGE_A_SATURATED_ROOT.parent / "stage-a-convergence-rescue"
RESCUE_REGISTRATION_ROOT = RESCUE_ROOT / "registration"
RESCUE_REGISTRATION_REQUEST = RESCUE_REGISTRATION_ROOT / "request.json"
RESCUE_CELL_REGISTRY = RESCUE_REGISTRATION_ROOT / "undertrained-cell-registry.jsonl"
RESCUE_UNIT_REGISTRY = RESCUE_REGISTRATION_ROOT / "affected-unit-registry.jsonl"
RESCUE_REGISTRATION_SUMMARY = RESCUE_REGISTRATION_ROOT / "summary.json"
RESCUE_REGISTRATION_VERIFICATION = RESCUE_REGISTRATION_ROOT / "verification.json"
STAGE_A_VERIFICATION_SHA256 = "32467b1cbe223a7297cb90b4546e71ac56478c834a720ec5af90775cfc01afb4"
STAGE_A_RUN_STATE_SHA256 = "c42fa75ef13cced9177e907157d1fa2414351d5bf1d79f4118380268f059e505"
STAGE_A_LEDGER_SHA256 = "95bf4d4c18b38372ca81af0ee8210a9b18da942db12f6162c8c678a0a1b9d342"
EXPECTED_UNDERTRAINED_CELLS = 113_392
EXPECTED_UNDERTRAINED_BY_FAMILY = {
    "continuous_ridge": 112_754,
    "event_logistic_l2": 638,
}
REGISTRY_WORKERS = 8
CONFIGURATION_PATTERN = re.compile(r"__(s01_e\d{2})__reg(\d{2})$")


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


def _canonical_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _registration_code_identity() -> str:
    return sha256_file(Path(__file__))


def _stage_a_ledger_entries() -> list[dict[str, object]]:
    path = PHASE02_STAGE_A_SATURATED_ROOT / "append-only-experiment-ledger.jsonl"
    entries: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line, parse_constant=_reject_nonfinite)
            if not isinstance(value, dict):
                raise TypeError("Stage A ledger contains a non-object")
            entries.append(cast(dict[str, object], value))
    _require(len(entries) == 40_824, "Stage A canonical ledger line count changed")
    return entries


def _derive_unit_cells(job: tuple[str, str, tuple[str, ...]]) -> dict[str, object]:
    path_text, expected_sha256, targets = job
    path = Path(path_text)
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, f"unit hash changed: {path}")
    result = _strict_json_bytes(payload)
    _require(result["outer_test_scores_opened"] is False, f"outer access in unit: {path}")
    _require(result["cortical_values_opened"] is False, f"cortical access in unit: {path}")
    unit = cast(dict[str, object], result["unit"])
    solver = cast(dict[str, object], result["solver"])
    model_family = cast(str, unit["model_family"])
    base_budget = int(
        cast(int, solver["budget"] if model_family == "continuous_ridge" else solver["base_budget"])
    )
    original_maximum_budget = int(
        cast(
            int,
            solver["budget"] if model_family == "continuous_ridge" else solver["maximum_budget"],
        )
    )
    diagnostic_key = (
        "relative_residual_by_cell"
        if model_family == "continuous_ridge"
        else "relative_gradient_by_cell"
    )
    diagnostics = cast(list[list[float]], solver[diagnostic_key])
    records = cast(list[dict[str, object]], result["records"])
    cells: list[dict[str, object]] = []
    for record in records:
        if record["status"] != "undertrained":
            continue
        _require(
            record["disposition"] == "protected_from_pruning_requires_16x_budget",
            f"undertrained disposition changed: {path}",
        )
        _require(record["converged"] is False, f"undertrained record marked converged: {path}")
        configuration_id = cast(str, record["configuration_id"])
        match = CONFIGURATION_PATTERN.search(configuration_id)
        _require(match is not None, f"configuration identity changed: {configuration_id}")
        assert match is not None
        candidate_id = match.group(1)
        regularization_index = int(match.group(2))
        target_index = targets.index(candidate_id)
        train_rows = int(cast(int, record["train_rows"]))
        tolerance = 1 / math.sqrt(train_rows)
        original_diagnostic = float(diagnostics[regularization_index][target_index])
        _require(
            math.isfinite(original_diagnostic) and original_diagnostic > tolerance,
            f"undertrained solver diagnostic no longer exceeds tolerance: {configuration_id}",
        )
        cells.append(
            {
                "schema_version": "veatic21_phase02_stage_a_rescue_cell_v1",
                "original_configuration_id": configuration_id,
                "original_unit_id": unit["unit_id"],
                "original_unit_sequence": unit["sequence"],
                "original_unit_result_sha256": expected_sha256,
                "protocol": unit["protocol"],
                "split_index": unit["split_index"],
                "repeat": unit["repeat"],
                "outer_fold": unit["outer_fold"],
                "inner_fold": unit["inner_fold"],
                "feature_form": unit["feature_form"],
                "history_depth": unit["history_depth"],
                "model_family": model_family,
                "candidate_id": candidate_id,
                "target_index": target_index,
                "regularization_index": regularization_index,
                "regularization_multiplier": record["regularization_multiplier"],
                "regularization_scale": record["regularization_scale"],
                "regularization_value": record["regularization_value"],
                "train_rows": train_rows,
                "validation_rows": record["validation_rows"],
                "train_threshold_q90": record["train_threshold_q90"],
                "split_sha256": result["split_sha256"],
                "feature_matrix_sha256": result["feature_matrix_sha256"],
                "scaler_sha256": result["scaler_sha256"],
                "target_thresholds_sha256": result["target_thresholds_sha256"],
                "original_base_budget": base_budget,
                "original_maximum_budget": original_maximum_budget,
                "rescue_maximum_budget": base_budget * 16,
                "convergence_tolerance": tolerance,
                "original_solver_diagnostic": original_diagnostic,
                "rescue_initialization": "zero",
                "rescue_disposition_if_converged": "eligible_for_inner_aggregation",
                "rescue_disposition_if_unresolved": (
                    "invalid_nonconverged_after_registered_maximum_budget"
                ),
                "outer_test_scores_opened": False,
                "cortical_values_opened": False,
            }
        )
    return {
        "unit_id": unit["unit_id"],
        "unit_sequence": unit["sequence"],
        "model_family": model_family,
        "original_unit_result_sha256": expected_sha256,
        "cells": cells,
    }


def _derive_all_cells() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registration = load_json(PHASE02_REGISTRATION_ROOT / "experiment-registration.json")
    targets = tuple(cast(list[str], registration["targets"]))
    ledger = _stage_a_ledger_entries()
    jobs = [
        (
            cast(str, entry["unit_result_path"]),
            cast(str, entry["unit_result_sha256"]),
            targets,
        )
        for entry in ledger
    ]
    cells: list[dict[str, object]] = []
    affected_units: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=REGISTRY_WORKERS) as pool:
        for derived in pool.map(_derive_unit_cells, jobs, chunksize=8):
            unit_cells = cast(list[dict[str, object]], derived["cells"])
            if not unit_cells:
                continue
            configuration_ids: list[str] = []
            cell_identities: list[str] = []
            for cell in unit_cells:
                cell["rescue_sequence"] = len(cells)
                cell["rescue_cell_identity_sha256"] = _canonical_sha256(cell)
                cells.append(cell)
                configuration_ids.append(cast(str, cell["original_configuration_id"]))
                cell_identities.append(cell["rescue_cell_identity_sha256"])
            affected_units.append(
                {
                    "schema_version": "veatic21_phase02_stage_a_rescue_unit_v1",
                    "rescue_unit_sequence": len(affected_units),
                    "original_unit_sequence": derived["unit_sequence"],
                    "original_unit_id": derived["unit_id"],
                    "original_unit_result_sha256": derived["original_unit_result_sha256"],
                    "model_family": derived["model_family"],
                    "undertrained_cell_count": len(unit_cells),
                    "original_configuration_ids": configuration_ids,
                    "rescue_cell_identity_sha256s": cell_identities,
                    "outer_test_scores_opened": False,
                    "cortical_values_opened": False,
                }
            )
    return cells, affected_units


def _write_jsonl_atomic(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line, parse_constant=_reject_nonfinite)
            if not isinstance(value, dict):
                raise TypeError(f"registry line is not an object: {path}")
            values.append(cast(dict[str, object], value))
    return values


def _summary_counts(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "by_model_family": dict(Counter(cast(str, cell["model_family"]) for cell in cells)),
        "by_protocol": dict(Counter(cast(str, cell["protocol"]) for cell in cells)),
        "by_feature_form": dict(Counter(cast(str, cell["feature_form"]) for cell in cells)),
        "by_history_depth": dict(
            sorted(Counter(str(cell["history_depth"]) for cell in cells).items())
        ),
        "by_candidate_id": dict(
            sorted(Counter(cast(str, cell["candidate_id"]) for cell in cells).items())
        ),
        "by_regularization_index": dict(
            sorted(Counter(str(cell["regularization_index"]) for cell in cells).items())
        ),
    }


def run_phase02_stage_a_rescue_registration() -> dict[str, object]:
    """Build the immutable external undertrained-cell and affected-unit registries."""

    root = reject_forbidden_runtime_path(RESCUE_REGISTRATION_ROOT)
    stage_a_verification = PHASE02_STAGE_A_SATURATED_ROOT / "verification.json"
    stage_a_run_state = PHASE02_STAGE_A_SATURATED_ROOT / "run-state.json"
    stage_a_ledger = PHASE02_STAGE_A_SATURATED_ROOT / "append-only-experiment-ledger.jsonl"
    _require(
        sha256_file(stage_a_verification) == STAGE_A_VERIFICATION_SHA256,
        "Stage A verification identity changed",
    )
    _require(sha256_file(stage_a_run_state) == STAGE_A_RUN_STATE_SHA256, "run state changed")
    _require(sha256_file(stage_a_ledger) == STAGE_A_LEDGER_SHA256, "Stage A ledger changed")
    verification = load_json(stage_a_verification)
    _require(verification["status"] == "PASS", "Stage A verification did not pass")
    _require(verification["outer_test_scores_opened"] is False, "Stage A opened outer data")
    _require(verification["cortical_values_opened"] is False, "Stage A opened cortical data")
    request = {
        "schema_version": "veatic21_phase02_stage_a_rescue_registration_request_v1",
        "master_specification_version": "2.2",
        "registration_builder_sha256": _registration_code_identity(),
        "stage_a_verification_sha256": STAGE_A_VERIFICATION_SHA256,
        "stage_a_run_state_sha256": STAGE_A_RUN_STATE_SHA256,
        "stage_a_canonical_ledger_sha256": STAGE_A_LEDGER_SHA256,
        "source_undertrained_cells": EXPECTED_UNDERTRAINED_CELLS,
        "rescue_budget": "total_16_times_data_derived_base_budget",
        "rescue_initialization": "zero",
        "converged_stage_a_cells_allowed": False,
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    if RESCUE_REGISTRATION_REQUEST.exists():
        _require(load_json(RESCUE_REGISTRATION_REQUEST) == request, "rescue request changed")
    _write_json(RESCUE_REGISTRATION_REQUEST, request)
    cells, affected_units = _derive_all_cells()
    _require(len(cells) == EXPECTED_UNDERTRAINED_CELLS, "undertrained cell count changed")
    by_family = Counter(cast(str, cell["model_family"]) for cell in cells)
    _require(dict(by_family) == EXPECTED_UNDERTRAINED_BY_FAMILY, "family counts changed")
    _require(
        len({cast(str, cell["original_configuration_id"]) for cell in cells}) == len(cells),
        "duplicate undertrained configuration",
    )
    _write_jsonl_atomic(RESCUE_CELL_REGISTRY, cells)
    _write_jsonl_atomic(RESCUE_UNIT_REGISTRY, affected_units)
    summary = {
        "schema_version": "veatic21_phase02_stage_a_rescue_registration_summary_v1",
        "status": "COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "registration_request_sha256": sha256_file(RESCUE_REGISTRATION_REQUEST),
        "undertrained_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
        "affected_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
        "undertrained_cells": len(cells),
        "affected_units": len(affected_units),
        "counts": _summary_counts(cells),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
    }
    _write_json(RESCUE_REGISTRATION_SUMMARY, summary)
    return summary


def verify_phase02_stage_a_rescue_registration() -> dict[str, object]:
    """Independently re-derive and exactly compare the frozen rescue registries."""

    root = reject_forbidden_runtime_path(RESCUE_REGISTRATION_ROOT)
    _require(root.is_dir(), "rescue registration root is missing")
    request = load_json(RESCUE_REGISTRATION_REQUEST)
    summary = load_json(RESCUE_REGISTRATION_SUMMARY)
    _require(
        request["registration_builder_sha256"] == _registration_code_identity(),
        "rescue registration builder changed",
    )
    produced_cells = _read_jsonl(RESCUE_CELL_REGISTRY)
    produced_units = _read_jsonl(RESCUE_UNIT_REGISTRY)
    expected_cells, expected_units = _derive_all_cells()
    _require(
        produced_cells == expected_cells, "undertrained-cell registry differs on re-derivation"
    )
    _require(produced_units == expected_units, "affected-unit registry differs on re-derivation")
    _require(len(produced_cells) == EXPECTED_UNDERTRAINED_CELLS, "cell coverage changed")
    _require(
        [cell["rescue_sequence"] for cell in produced_cells]
        == list(range(EXPECTED_UNDERTRAINED_CELLS)),
        "rescue cell sequence changed",
    )
    for cell in produced_cells:
        identity = cast(str, cell["rescue_cell_identity_sha256"])
        without_identity = {
            key: value for key, value in cell.items() if key != "rescue_cell_identity_sha256"
        }
        _require(identity == _canonical_sha256(without_identity), "rescue cell identity changed")
        _require(cell["outer_test_scores_opened"] is False, "cell registry opened outer data")
        _require(cell["cortical_values_opened"] is False, "cell registry opened cortical data")
    _require(
        summary["registration_request_sha256"] == sha256_file(RESCUE_REGISTRATION_REQUEST),
        "rescue request hash changed",
    )
    _require(
        summary["undertrained_cell_registry_sha256"] == sha256_file(RESCUE_CELL_REGISTRY),
        "cell registry hash changed",
    )
    _require(
        summary["affected_unit_registry_sha256"] == sha256_file(RESCUE_UNIT_REGISTRY),
        "unit registry hash changed",
    )
    result = {
        "schema_version": "veatic21_phase02_stage_a_rescue_registration_verification_v1",
        "status": "PASS",
        "verification_workers": REGISTRY_WORKERS,
        "registration_builder_sha256": _registration_code_identity(),
        "registration_request_sha256": sha256_file(RESCUE_REGISTRATION_REQUEST),
        "undertrained_cell_registry_sha256": sha256_file(RESCUE_CELL_REGISTRY),
        "affected_unit_registry_sha256": sha256_file(RESCUE_UNIT_REGISTRY),
        "summary_sha256": sha256_file(RESCUE_REGISTRATION_SUMMARY),
        "undertrained_cells": len(produced_cells),
        "affected_units": len(produced_units),
        "counts": _summary_counts(produced_cells),
        "unique_original_configuration_ids": len(
            {cast(str, cell["original_configuration_id"]) for cell in produced_cells}
        ),
        "unique_rescue_cell_identities": len(
            {cast(str, cell["rescue_cell_identity_sha256"]) for cell in produced_cells}
        ),
        "outer_test_scores_opened": False,
        "cortical_values_opened": False,
        "gates": {
            "source_identity": "PASS",
            "exact_rederivation": "PASS",
            "cell_coverage": "PASS",
            "unit_grouping": "PASS",
            "sequence_identity": "PASS",
            "configuration_uniqueness": "PASS",
            "cell_identity_hashes": "PASS",
            "access_firewall": "PASS",
        },
    }
    _write_json(RESCUE_REGISTRATION_VERIFICATION, result)
    return result


COMPACT_RESCUE_REGISTRATION = REPOSITORY_ROOT / (
    "internal/active/veatic21-phase02-registration/convergence-rescue-registration.json"
)
