from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from neural_bridge.veatic21.data import sha256_file
from neural_bridge.veatic21.phase02_stage_b import (
    EXPECTED_AGGREGATION_VERIFICATION_SHA256,
    EXPECTED_WORK_REGISTRY_SHA256,
    PreparedStageBUnit,
    StageBWorkUnit,
    candidate_cell_id,
    execute_stage_b_cell,
    iter_work_units,
    logical_candidate,
    model_seed,
)
from neural_bridge.veatic21.phase02_stage_b_executor import (
    _append_unique_ledger,
    _execute_on_lane,
    _merge_main_ledgers,
    _publish_backtest_cell,
    _publish_unit_bundle,
    _registered_cell_pairs,
    _registration,
    _resolve_cells,
    _resume_audit,
    _valid_existing_bundle,
)
from neural_bridge.veatic21.phase02_stage_b_verify import _coverage

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_PATH = ROOT / (
    "internal/active/veatic21-phase02-registration/stage-b-execution-registration.json"
)
SELECTED_EXECUTOR_PATH = ROOT / (
    "internal/active/veatic21-phase02-registration/selected-stage-b-executor.json"
)
CURRENT = ROOT / "internal/handoff/CURRENT_STATE.md"


def _synthetic(
    family: str,
) -> tuple[PreparedStageBUnit, dict[str, Any]]:
    candidate: dict[str, Any]
    if family == "event_mlp":
        candidate = {
            "family": "event_mlp",
            "search_role": "one_factor_at_a_time",
            "width": 4,
            "layers": 1,
            "activation": "relu",
            "dropout": 0.0,
            "optimizer": "adamw",
            "learning_rate": 0.05,
            "batch_size": 16,
            "initial_update_budget": 8,
            "undertraining_recovery_maximum_budget": 16,
        }
        form, depth, feature_count = "current_only", 1, 1
    elif family == "event_gru":
        candidate = {
            "family": "event_gru",
            "search_role": "one_factor_at_a_time",
            "width": 4,
            "layers": 1,
            "activation": "gru_native_tanh_sigmoid",
            "dropout": 0.0,
            "optimizer": "adamw",
            "learning_rate": 0.05,
            "batch_size": 16,
            "initial_update_budget": 8,
            "undertraining_recovery_maximum_budget": 16,
        }
        form, depth, feature_count = "raw_sequence_with_availability_mask", 2, 5
    else:
        candidate = {
            "family": family,
            "search_role": "synthetic_solver_test",
            "regularization_multiplier": 1.0,
            "regularization_value": 0.1,
            "initial_update_budget": 8,
            "maximum_convergence_budget": 16,
        }
        if family == "event_elastic_net":
            candidate.pop("maximum_convergence_budget")
            candidate["l1_ratio"] = 0.5
            candidate["undertraining_recovery_maximum_budget"] = 16
        form, depth, feature_count = "current_only", 1, 1
    unit = StageBWorkUnit(
        work_unit_id="synthetic",
        sequence=0,
        scope_id="blocked_rna_o00",
        protocol="blocked",
        repeat=None,
        outer_fold=0,
        inner_fold=0,
        candidate_id="s01_e01",
        feature_set_id="synthetic",
        feature_form=form,
        history_depth_rows=depth,
        feature_count=feature_count,
        train_rows=96,
        validation_rows=32,
        split_sha256="0" * 64,
        regularization_scale=1.0,
        candidate_ids_sha256="synthetic",
        candidates=(candidate,),
    )
    rng = np.random.default_rng(11)
    if family == "event_gru":
        sequence = rng.normal(size=(128, 3, 2)).astype(np.float32)
        vector = sequence.reshape(128, 6)[:, :5]
        continuous = sequence[:, -1, 0]
    else:
        sequence = None
        vector = rng.normal(size=(128, 1)).astype(np.float32)
        continuous = vector[:, 0]
    labels = (continuous > 0).astype(np.float32)
    prepared = PreparedStageBUnit(
        unit=unit,
        target_index=0,
        x_vector=vector,
        x_linear=np.column_stack([vector, np.ones(128, dtype=np.float32)]),
        x_sequence=sequence,
        target_values=continuous.astype(np.float32),
        labels=labels,
        train_indices=np.arange(96),
        validation_indices=np.arange(96, 128),
        threshold=0.0,
        feature_matrix_sha256="feature",
        scaler_sha256="scaler",
        target_thresholds_sha256="thresholds",
        preparation_seconds=0.0,
    )
    return prepared, candidate


def test_stage_b_registration_pins_verified_exact_registry() -> None:
    registration = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    identity = registration["input_identity"]
    assert registration["registration_status"] == "prospective_before_any_stage_b_fit"
    assert identity["stage_b_work_registry_sha256"] == EXPECTED_WORK_REGISTRY_SHA256
    assert (
        identity["stage_a_aggregation_verification_sha256"]
        == EXPECTED_AGGREGATION_VERIFICATION_SHA256
    )
    assert identity["stage_b_work_units"] == 40_824
    assert identity["stage_b_candidate_cells"] == 2_351_229
    assert registration["access_boundary"]["stage_b_main_run_authorized"] is False
    current = CURRENT.read_text(encoding="utf-8")
    assert "ffaf5b86254099865768e60825db048a763140277c54050005fa640e86cca010" in current
    assert "99e288f45c2f54e514ccd98a39703406fc0adb2aada3489eae3510fb9f94b7d7" in current
    assert "255bf34330fb0e002b7436db1243f91097c656337f1e23b913ff8a4bc01c92a5" in current


def test_registered_backtest_cells_exist_and_cover_every_family_and_shape_axis() -> None:
    registration = _registration()
    assert len(_registered_cell_pairs(registration)) == 34
    coverage = _coverage(registration)
    assert coverage["cells"] == 34
    assert coverage["protocols"] == ["blocked", "grouped"]
    assert len(coverage["feature_forms"]) == 6
    assert len(coverage["families"]) == 5
    assert coverage["maximum_gru_depth"] == 19


def test_selected_stage_b_executor_pins_verified_complete_backtest() -> None:
    selected = json.loads(SELECTED_EXECUTOR_PATH.read_text(encoding="utf-8"))
    assert selected["eligible_for_main"] is True
    assert selected["execution_registration_sha256"] == (
        "ffaf5b86254099865768e60825db048a763140277c54050005fa640e86cca010"
    )
    assert selected["stage_b_code_sha256"] == (
        "99e288f45c2f54e514ccd98a39703406fc0adb2aada3489eae3510fb9f94b7d7"
    )
    assert selected["backtest_request_sha256"] == (
        "d7f6af27d80b59d8e3401d404130762af9c06d58dbba54fa2f40ba0705ad08da"
    )
    assert selected["backtest_result_sha256"] == (
        "d8c74788f4ab3cd13bf8970a5f54c212bb0f3d8b6ba74f646bb0a19d1b52410f"
    )
    assert selected["backtest_verification_sha256"] == (
        "80a587c78f234ba52ac621599ab766b4271a6946cde72dd02bd0460c8295b8ad"
    )
    assert selected["topology"] == {
        "cpu_preparation_workers": 1,
        "mlx_stream_lanes": 4,
    }
    assert selected["coverage"]["verified_cell_artifacts"] == 2_550
    assert selected["resource_gates"]["all_topologies_eligible"] is True
    assert selected["outer_test_scores_opened"] is False
    assert selected["cortical_values_opened"] is False
    current = CURRENT.read_text(encoding="utf-8")
    assert sha256_file(SELECTED_EXECUTOR_PATH) == (
        "52192d5336db1b18ec7bd6703174d42c8d9feef5cca1c7fd07fdf87aecd125e8"
    )
    assert "52192d5336db1b18ec7bd6703174d42c8d9feef5cca1c7fd07fdf87aecd125e8" in current


def test_candidate_ids_and_model_seeds_are_stable() -> None:
    unit = next(iter_work_units())
    candidate = unit.candidates[0]
    assert candidate_cell_id(unit, candidate) == candidate_cell_id(unit, candidate)
    assert model_seed(unit, candidate) == model_seed(unit, candidate)
    assert 0 <= model_seed(unit, candidate) < 2**32


def test_logical_neural_identity_normalizes_train_row_formulas() -> None:
    prepared, candidate = _synthetic("event_mlp")
    unit = prepared.unit
    normalized = logical_candidate(candidate, unit)
    assert normalized["learning_rate_factor"] == candidate["learning_rate"] * np.sqrt(96)
    assert normalized["batch_factor"] == 2.0
    assert normalized["width_factor"] == 4.0


def test_all_solver_families_emit_finite_deterministic_inner_predictions() -> None:
    for family in (
        "continuous_ridge",
        "event_logistic_l2",
        "event_elastic_net",
        "event_mlp",
        "event_gru",
    ):
        prepared, candidate = _synthetic(family)
        first = execute_stage_b_cell(prepared, candidate)
        second = execute_stage_b_cell(prepared, candidate)
        assert np.array_equal(first[1], second[1])
        assert first[0]["metrics"] == second[0]["metrics"]
        assert first[0]["checkpoint_sha256"] == second[0]["checkpoint_sha256"]
        assert np.isfinite(first[1]).all()
        assert first[0]["outer_test_scores_opened"] is False
        assert first[0]["cortical_values_opened"] is False


def test_two_mlx_streams_preserve_exact_cell_evidence() -> None:
    prepared, candidate = _synthetic("event_mlp")
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(
            pool.map(
                _execute_on_lane,
                [(prepared, candidate), (prepared, candidate)],
            )
        )
    assert np.array_equal(outputs[0][1], outputs[1][1])
    assert outputs[0][0]["metrics"] == outputs[1][0]["metrics"]
    assert outputs[0][0]["checkpoint_sha256"] == outputs[1][0]["checkpoint_sha256"]


def test_backtest_selection_resolves_only_registered_candidate_cells() -> None:
    registration = _registration()
    resolved = _resolve_cells(_registered_cell_pairs(registration))
    assert len(resolved) == 34
    assert all(
        candidate_cell_id(unit, candidate)
        == dict(_registered_cell_pairs(registration))[unit.sequence]
        for unit, candidate in resolved
    )


def test_atomic_backtest_and_main_bundles_pass_hash_checked_resume(tmp_path: Path) -> None:
    prepared, candidate = _synthetic("event_mlp")
    output = execute_stage_b_cell(prepared, candidate)
    manifest = _publish_backtest_cell(tmp_path / "backtest", *output)
    resume = _resume_audit(tmp_path / "backtest", [manifest])
    assert resume["status"] == "PASS"
    assert resume["cells_reused"] == 1

    unit_manifest = _publish_unit_bundle(
        tmp_path / "main",
        prepared.unit,
        [output],
    )
    existing = _valid_existing_bundle(tmp_path / "main", prepared.unit)
    assert existing is not None
    assert existing == unit_manifest
    assert existing["candidate_cells"] == 1
    _append_unique_ledger(
        tmp_path / "main/ledgers/shard-00.jsonl",
        existing,
        identity_key="work_unit_sequence",
    )
    ledger = _merge_main_ledgers(
        tmp_path / "main",
        lane_count=1,
        expected_sequences={0},
    )
    assert ledger["ledger_lines"] == 1
