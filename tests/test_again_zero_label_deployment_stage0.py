from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage0 as stage0


def synthetic_video_ids() -> list[str]:
    return [f"video_{index:04d}" for index in range(stage0.EXPECTED_VIDEOS)]


def test_dry_run_authorizes_stage0_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(stage0.__file__).resolve()), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "stage0_planning_contracts_only"
    assert payload["stage_a_expected_rows"] == 96
    assert payload["stage_b_expected_rows"] == 140
    assert payload["required_metrics"] == list(stage0.REQUIRED_METRICS)
    assert payload["teacher_incremental_gain_retention_minimum"] == 0.50
    assert payload["absolute_phase7_parity_required"] is False
    assert payload["model_training_performed"] is False
    assert payload["teacher_scores_generated"] is False
    assert payload["heldout_predictions_scored"] is False
    assert payload["accelerator_for_later_training"] == "mlx_gpu_mps"
    assert payload["no_cpu_fallback"] is True


def test_split_lock_is_deterministic_exact_and_video_disjoint() -> None:
    ids = synthetic_video_ids()
    first = stage0.build_split_manifest(ids)
    second = stage0.build_split_manifest(reversed(ids))
    assert first == second
    assert first["development_count"] == 696
    assert first["locked_count"] == 299
    assert [fold["test_count"] for fold in first["stage_a"]] == [232, 232, 232]
    assert [panel["test_count"] for panel in first["stage_b"]["panels"]] == [60, 60, 60, 60, 59]
    assert first["historically_untouched"] is False
    assert first["prospectively_locked_for_this_method"] is True
    stage0.validate_split_manifest(first)


def test_nested_teacher_crossfit_never_owns_outer_test_video() -> None:
    manifest = stage0.build_split_manifest(synthetic_video_ids())
    for fold in manifest["stage_a"]:
        outer_test = set(fold["test_videos"])
        crossfit_test = [set(values) for values in fold["teacher_crossfit_test_folds"]]
        assert not any(outer_test & values for values in crossfit_test)
        assert set().union(*crossfit_test) == set(fold["train_videos"])
    locked = set(manifest["locked_videos"])
    assert not any(
        locked & set(values)
        for values in manifest["stage_b"]["teacher_crossfit_test_folds"]
    )


def test_dry_run_matrix_is_96_plus_140_and_metrics_do_not_multiply_rows() -> None:
    manifest = stage0.build_split_manifest(synthetic_video_ids())
    frame = stage0.dry_run_matrix(manifest)
    assert len(frame) == 236
    assert len(frame[frame["stage"] == "stage_a"]) == 96
    assert len(frame[frame["stage"] == "stage_b"]) == 140
    assert len(frame[(frame["stage"] == "stage_a") & (frame["row_type"] == "member")]) == 72
    assert len(frame[(frame["stage"] == "stage_a") & (frame["row_type"] == "ensemble")]) == 24
    assert len(frame[(frame["stage"] == "stage_b") & (frame["row_type"] == "member")]) == 105
    assert len(frame[(frame["stage"] == "stage_b") & (frame["row_type"] == "ensemble")]) == 35
    assert set(frame[frame["stage"] == "stage_b"]["lane"]) == set(stage0.STAGE_B_LANES)


def test_matrix_fails_closed_on_missing_or_duplicate_row() -> None:
    frame = stage0.dry_run_matrix(stage0.build_split_manifest(synthetic_video_ids()))
    with pytest.raises(ValueError):
        stage0.validate_dry_run_matrix(frame.iloc[:-1])
    with pytest.raises(ValueError):
        stage0.validate_dry_run_matrix(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


def test_video_only_inference_block_has_no_ar_or_target_arrays() -> None:
    names = stage0.inference_block_field_names()
    assert not any("arousal" in name or name.startswith("ar_") for name in names)
    assert not any("target" in name or "label" in name or "teacher" in name for name in names)
    block = stage0.VideoOnlyInferenceBlock(
        row_ids=np.asarray(["a:0", "a:1"]),
        video_ids=np.asarray(["a", "a"]),
        row_indices=np.asarray([0, 1]),
        pca_sequence=np.zeros((2, 5, 256), dtype=np.float32),
        diagnostics=np.zeros((2, 53), dtype=np.float32),
        history_available=np.asarray([[True, False, False, False, False], [True, True, False, False, False]]),
        time_features=np.zeros((2, 2), dtype=np.float32),
    )
    block.validate()


@pytest.mark.parametrize(
    "field",
    [
        "arousal",
        "valence",
        "arousal_lag_1row",
        "arousal_delta_prev_4row",
        "future_arousal_max_delta_rows_4_10",
        "target_mask_future_arousal_max_delta_rows_4_10",
        "teacher_score",
        "ground_truth_lag",
        "label_available",
    ],
)
def test_inference_schema_rejects_response_target_and_teacher_fields(field: str) -> None:
    with pytest.raises(ValueError):
        stage0.validate_inference_feature_names([field])


def test_inference_schema_accepts_only_video_side_examples() -> None:
    stage0.validate_inference_feature_names(
        [
            "pca256_current_row",
            "pca256_same_video_lag_4",
            "temporal_diagnostics_53",
            "history_available_lag_4",
            "time_seconds",
            "video_time_fraction",
        ]
    )


def test_causal_history_is_same_video_and_zero_padded_at_start() -> None:
    videos = ["a", "a", "a", "b", "b"]
    rows = [0, 1, 2, 0, 1]
    indices, available = stage0.causal_history_indices(videos, rows)
    assert indices[0].tolist() == [0, -1, -1, -1, -1]
    assert available[0].tolist() == [True, False, False, False, False]
    assert indices[2, :3].tolist() == [2, 1, 0]
    assert indices[3].tolist() == [3, -1, -1, -1, -1]
    for target, row in enumerate(indices):
        for source in row[row >= 0]:
            assert videos[int(source)] == videos[target]
            assert rows[int(source)] <= rows[target]


def test_rollout_dependency_audit_accepts_predictions_and_rejects_teacher_forcing() -> None:
    valid = [
        {"video_id": "a", "row_index": 0, "state_source": "train_median_initialization", "source_row_index": None},
        {"video_id": "a", "row_index": 1, "state_source": "prediction", "source_row_index": 0},
        {"video_id": "b", "row_index": 0, "state_source": "train_median_initialization", "source_row_index": None},
    ]
    stage0.validate_rollout_dependencies(valid)
    bad = [dict(valid[0]), {**valid[1], "teacher_forced": True}]
    with pytest.raises(ValueError):
        stage0.validate_rollout_dependencies(bad)
    with pytest.raises(ValueError):
        stage0.validate_rollout_dependencies([dict(valid[0]), {**valid[1], "source_row_index": 1}])
    with pytest.raises(ValueError):
        stage0.validate_rollout_dependencies(
            [dict(valid[0]), {**valid[1], "source_video_id": "other_video"}]
        )


def test_prediction_must_be_sealed_before_label_join() -> None:
    stage0.validate_prediction_seal(
        {
            "labels_loaded_before_checksum": False,
            "prediction_sha256": "abc",
            "prediction_columns": ["video_id", "row_index", "prediction"],
        }
    )
    with pytest.raises(ValueError):
        stage0.validate_prediction_seal(
            {
                "labels_loaded_before_checksum": True,
                "prediction_sha256": "abc",
                "prediction_columns": ["video_id", "row_index", "prediction"],
            }
        )


def test_event_pr_auc_gate_fails_closed_for_single_class_slice() -> None:
    stage0.require_event_gate_defined([0, 1, 0, 1])
    with pytest.raises(ValueError):
        stage0.require_event_gate_defined([0, 0, 0])
    with pytest.raises(ValueError):
        stage0.require_event_gate_defined([1, 1])


def test_target_contract_uses_raw_future_movement_not_residual_target() -> None:
    spec = stage0.redesigned.target_specs()[0]
    assert stage0.TARGET_NAME == "future_arousal_max_delta_rows_4_10"
    assert stage0.TARGET_MASK == "target_mask_future_arousal_max_delta_rows_4_10"
    assert spec.value_column == stage0.TARGET_NAME
    assert spec.transform == "positive_delta"
    assert spec.quantile == 0.90


def test_canonical_stage0_snapshot_passes_without_authorizing_stage_a() -> None:
    root = stage0.DEFAULT_OUTPUT_ROOT
    result = json.loads((root / "stage0_result.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    target = json.loads((root / "target_identity_manifest.json").read_text(encoding="utf-8"))
    features = json.loads((root / "feature_policy_manifest.json").read_text(encoding="utf-8"))
    matrix = pd.read_csv(root / "dry_run_matrix.csv")
    assert result["stage0_pass"] is True
    assert result["stage_a_authorized"] is False
    assert result["model_training_performed"] is False
    assert result["teacher_scores_generated"] is False
    assert result["heldout_predictions_scored"] is False
    assert result["stage_a_matrix_rows"] == 96
    assert result["stage_b_matrix_rows"] == 140
    for name, digest in result["artifact_sha256"].items():
        assert stage0.file_digest(root / name) == digest
    assert split["development_count"] == 696 and split["locked_count"] == 299
    stage0.validate_split_manifest(split)
    assert target["value_column"] == stage0.TARGET_NAME
    assert target["residual_target_used"] is False
    target_without_identity = dict(target)
    recorded_identity = target_without_identity.pop("target_identity_digest")
    assert stage0.canonical_digest(target_without_identity) == recorded_identity
    assert target["builder_source_sha256"] == stage0.source_digest(stage0.redesigned.future_max_delta)
    assert target["event_scorer_source_sha256"] == stage0.source_digest(stage0.base.threshold_labels)
    assert all(record["event_pr_auc_defined"] for record in target["split_records"])
    assert all(record["first30_event_pr_auc_defined"] for record in target["split_records"])
    assert features["pca"]["width"] == 256
    assert features["pca"]["locked_or_outer_test_rows_in_fit"] is False
    assert features["pca"]["teacher_basis_policy"].startswith("fit_inside_each_nested")
    assert features["pca"]["reuse_incompatible_phase7_fold_score_matrices"] is False
    stage0.validate_dry_run_matrix(matrix)
