import json
from pathlib import Path

import numpy as np
import pytest

from backend.scripts import run_veatic_frozen_tensor_trained_heads_benchmark as wrapper
from backend.scripts import veatic_frozen_tensor_adapter as adapter
from backend.scripts import veatic_frozen_tensor_trained_heads as trained


def _row(video_id: str, index: int, y: float) -> dict:
    return {
        "video_id": video_id,
        "time_start_seconds": float(index),
        "frame_index": index,
        "targets": {"arousal": y},
    }


def _contract(tmp_path: Path, *, representation: str = adapter.PRIMARY_CANDIDATE) -> tuple[adapter.FrozenTensorContract, list[dict]]:
    all_rows = [_row(str(index // 2), index, float(index % 2)) for index in range(10)]
    train_rows = all_rows[:6]
    test_rows = all_rows[6:]
    contract_dir = tmp_path / representation / "grouped_0" / adapter.target_dir_name("arousal__future_spike_1_3s", 0.05)
    tracked_dir = tmp_path / "tracked" / representation
    contract_dir.mkdir(parents=True)
    tracked_dir.mkdir(parents=True)
    train_x = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
            [0.2, 0.2],
            [1.2, 1.2],
        ],
        dtype=np.float64,
    )
    test_x = np.asarray([[0.05, 0.0], [1.05, 1.0], [0.25, 0.2], [1.25, 1.2]], dtype=np.float64)
    if representation == adapter.PRIMARY_CANDIDATE:
        np.save(contract_dir / "X_sequence_train.npy", np.repeat(train_x[:, None, :], 3, axis=1))
        np.save(contract_dir / "X_sequence_test.npy", np.repeat(test_x[:, None, :], 3, axis=1))
        np.save(contract_dir / "sequence_mask_train.npy", np.ones((6, 3), dtype=np.float64))
        np.save(contract_dir / "sequence_mask_test.npy", np.ones((4, 3), dtype=np.float64))
    return (
        adapter.FrozenTensorContract(
            representation_name=representation,
            feature_name=adapter.FEATURE_ALIASES.get(representation, representation),
            split="grouped_0",
            target_name="arousal__future_spike_1_3s",
            threshold=0.05,
            contract_dir=contract_dir,
            tracked_dir=tracked_dir,
            train_x=train_x,
            test_x=test_x,
            train_y=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float64),
            test_y=np.asarray([0, 1, 0, 1], dtype=np.float64),
            train_row_metadata=train_rows,
            test_row_metadata=test_rows,
            representation_metadata={"uses_future_features": False},
            split_metadata={"train_test_video_overlap": []},
            target_metadata={"target_name": "arousal__future_spike_1_3s", "threshold": 0.05, "task_type": "binary"},
            checksum_manifest={"files": []},
            leakage_contract={"no_future_rows_in_primary": True},
        ),
        all_rows,
    )


def test_trained_head_wrapper_rejects_exclude_video_83_option():
    with pytest.raises(SystemExit):
        wrapper.build_parser().parse_args(["--exclude-video-83"])


def test_without_83_paths_are_rejected():
    with pytest.raises(ValueError, match="exclude-video-83"):
        adapter.reject_exclude_83_paths(Path("outputs/without_83_attempt"))


def test_logistic_head_selection_uses_inner_validation_only():
    rows = [_row(str(index // 2), index, float(index % 2)) for index in range(12)]
    train_x = np.asarray([[index, index % 2] for index in range(12)], dtype=np.float64)
    train_y = np.asarray([index % 2 for index in range(12)], dtype=np.float64)
    test_x = train_x[:4]

    bundle = trained.fit_logistic_grid("logistic_l2", train_x, train_y, test_x, rows, seed=3)

    assert bundle.selection["test_labels_used_for_selection"] is False
    assert "train_only" in bundle.selection["inner_validation_strategy"]
    assert bundle.selection["selected_hyperparameters"]["C"] in trained.LOGISTIC_C_GRID
    assert bundle.selection["training_backend"] == "torch_mps"
    assert bundle.selection["device"] == "mps"
    assert bundle.selection["selected_hyperparameters"]["solver"] == "torch_mps_adam"


def test_ar_and_controls_are_recomputed_fresh(monkeypatch, tmp_path):
    contract, all_rows = _contract(tmp_path)
    calls = []

    def fake_ar(all_manifest_rows, rows, target, *, include_current):
        calls.append({"row_count": len(rows), "target": target, "include_current": include_current})
        return np.column_stack([np.ones(len(rows)), np.arange(len(rows), dtype=np.float64)])

    monkeypatch.setattr(trained.bench, "autoregressive_features", fake_ar)

    lane_rows, leakage_checks, _selection = trained.score_collapsed_contract(
        fresh_run_id="unit",
        contract=contract,
        all_rows=all_rows,
        heads=("ridge_score",),
        seed=5,
    )

    assert calls == [
        {"row_count": 6, "target": "arousal", "include_current": True},
        {"row_count": 4, "target": "arousal", "include_current": True},
    ]
    assert any(row["canonical_control"] for row in lane_rows)
    assert any(check["check_name"] == "computed_controls_fresh" and check["status"] == "pass" for check in leakage_checks)


def test_residualized_lane_uses_train_residual(monkeypatch, tmp_path):
    contract, all_rows = _contract(tmp_path)
    seen = {}
    original = trained.fit_residual_correction_bundle

    def spy(head_name, train_x, residual_train, test_x, **kwargs):
        seen["residual_len"] = len(residual_train)
        seen["test_width"] = test_x.shape[1]
        return original(head_name, train_x, residual_train, test_x, **kwargs)

    monkeypatch.setattr(trained, "fit_residual_correction_bundle", spy)

    lane_rows, _checks, _selection = trained.score_collapsed_contract(
        fresh_run_id="unit",
        contract=contract,
        all_rows=all_rows,
        heads=("ridge_score",),
        seed=7,
    )

    assert seen["residual_len"] == 6
    assert any(row["model_lane"] == "residualized_AR_plus_PCA128" for row in lane_rows)


def test_sequence_flattening_preserves_row_count():
    sequence = np.zeros((5, 3, 128), dtype=np.float64)

    flat = trained.flatten_sequence(sequence)

    assert flat.shape == (5, 384)


def test_temporal_pooling_produces_rows_by_128():
    sequence = np.zeros((5, 3, 128), dtype=np.float64)

    pooled = trained.temporal_pool_sequence(sequence, (0.2, 0.3, 0.5))

    assert pooled.shape == (5, 128)


def test_topk_rows_are_marked_cautionary():
    flags = trained.lane_flags("topk_vertices_512", "AR_plus_topk_vertices_512")

    assert flags["uses_topk"] is True
    assert flags["uses_supervised_features"] is True
    assert flags["cautionary"] is True


def _gate_row(split: str, lane: str, value: float, *, control: bool = False, head: str = "logistic_l2") -> dict:
    feature_name = "PCA64_delta" if "PCA64_delta" in lane else "PCA128"
    return {
        "split_name": split,
        "target_name": "arousal__future_spike_1_3s",
        "threshold": 0.05,
        "model_lane": lane,
        "feature_name": feature_name,
        "head_name": head,
        "pr_auc": value,
        "canonical_control": control,
        "cautionary": False,
    }


def test_promotion_cannot_pass_without_beating_ar_and_controls():
    rows = []
    for fold in range(5):
        split = f"grouped_{fold}"
        rows.extend(
            [
                _gate_row(split, "AR_only", 0.5),
                _gate_row(split, "PCA128_only", 0.4),
                _gate_row(split, "PCA64_delta_only", 0.3),
                _gate_row(split, "AR_plus_PCA128", 0.49),
                _gate_row(split, "AR_plus_PCA64_delta", 0.45),
                _gate_row(split, "residualized_AR_plus_PCA128", 0.48),
                _gate_row(split, "residualized_AR_plus_PCA64_delta", 0.44),
                _gate_row(split, "AR_plus_random_PCA128", 0.6, control=True),
                _gate_row(split, "residualized_AR_plus_random_PCA128", 0.6, control=True),
            ]
        )

    gates, candidates = trained.compute_gate_checks(rows, [])

    assert any(gate["gate_name"] == "AR_plus_PCA128 > AR_only" and gate["status"] == "fail" for gate in gates)
    assert all(candidate["category"] != "promoted_incremental_neural_value" for candidate in candidates)


def test_required_run_fields_fail_if_any_core_flag_false():
    payload = dict(trained.required_run_fields())
    payload["computed_controls_fresh"] = False

    with pytest.raises(ValueError, match="computed_controls_fresh"):
        trained.assert_required_run_fields(payload)
