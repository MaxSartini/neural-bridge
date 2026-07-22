from __future__ import annotations

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import FeatureRows, LabelRows, SubstrateIdentity
from neural_bridge.veatic21.preregistration import (
    benchmark_partition_mask,
    build_event_preregistration,
    calibrate_event_preregistration,
)


def _canonical_fixture() -> tuple[SubstrateIdentity, FeatureRows, LabelRows]:
    video_ids = tuple(str(index) for index in range(124))
    counts = np.full(124, 166, dtype=np.int64)
    counts[:73] += 1
    videos = np.concatenate(
        [np.full(count, video, dtype="U3") for video, count in zip(video_ids, counts, strict=True)]
    )
    rows = np.concatenate([np.arange(count) for count in counts]).astype(np.int32)
    times = rows.astype(np.float32) / 2.0
    eligible = np.ones(len(rows), dtype=np.bool_)
    offset = 0
    for video_index, count in enumerate(counts):
        eligible[offset : offset + 7] = False
        if video_index < 55:
            eligible[offset + 7] = False
        offset += int(count)
    assert int((~eligible).sum()) == 923
    phase = np.asarray([int(video) * 0.17 for video in videos])
    arousal = (np.sin(times * 0.21 + phase) + 0.12 * np.sin(times * 0.91)).astype(np.float32)
    features = FeatureRows(
        video_id=videos,
        row_index=rows,
        time_seconds=times,
        quality_eligible=eligible,
        representations={"diagnostics_only": np.zeros((len(rows), 1), dtype=np.float32)},
    )
    labels = LabelRows(
        video_id=videos,
        row_index=rows,
        time_seconds=times,
        arousal=arousal,
        valence=np.cos(times * 0.13 + phase).astype(np.float32),
    )
    identity = SubstrateIdentity(
        video_ids=video_ids,
        row_count=20_657,
        exclusion_count=923,
        row_hz=2.0,
        vjepa_artifact_id="veatic-2.1-vjepa-fixture",
        vjepa_sha256_tree="a" * 64,
        vjepa_file_count=124,
        vjepa_size_bytes=1,
        tribe_artifact_id="veatic-2.1-tribe-fixture",
        tribe_sha256_tree="b" * 64,
        tribe_file_count=124,
        tribe_size_bytes=1,
        row_plan_sha256="c" * 64,
        source_tree_sha256="d" * 64,
        encoder_model_sha256="e" * 64,
    )
    return identity, features, labels


def _label_subset(labels: LabelRows, mask: np.ndarray) -> LabelRows:
    return LabelRows(
        video_id=labels.video_id[mask],
        row_index=labels.row_index[mask],
        time_seconds=labels.time_seconds[mask],
        arousal=labels.arousal[mask],
        valence=labels.valence[mask],
    )


def test_preregistration_locks_label_blind_temporal_benchmark_split() -> None:
    identity, features, _ = _canonical_fixture()

    first = build_event_preregistration(identity, features)
    second = build_event_preregistration(identity, features)

    assert first == second
    assert first["label_values_accessed"] is False
    split = first["split"]
    train_mask = benchmark_partition_mask(features, split, "train")
    test_mask = benchmark_partition_mask(features, split, "test")
    assert len(split["video_ids"]) == 124
    assert len(split["per_video_boundaries"]) == 124
    assert int(train_mask.sum()) == split["benchmark_train_rows"]
    assert int(test_mask.sum()) == split["benchmark_test_rows"]
    assert int(train_mask.sum() + test_mask.sum()) == 19_734
    assert set(features.video_id[train_mask]) == set(identity.video_ids)
    assert set(features.video_id[test_mask]) == set(identity.video_ids)
    assert not (train_mask & test_mask).any()
    assert first["split"]["requested_train_fraction"] == 0.70
    assert first["label_access"]["production_refit"].endswith("all_124_after_benchmark")
    assert first["label_access"]["production_refit_is_claim_evidence"] is False
    assert first["target_calibration"]["source"] == "benchmark_train_arousal_labels_only"
    assert first["training"]["comparison_seed_panel"] == [20_260_722, 20_260_723, 20_260_724]
    assert len(first["training"]["stability_seed_panel"]) == 9
    assert "without_changing" in first["training"]["freshness_policy"]
    assert first["representations"]["pca"]["actual_width"].startswith("smallest_train")
    assert "all_124" in first["representations"]["pca"]["production_refit_scope"]
    assert first["representations"]["pca"]["cache_reuse_across"] == [
        "targets",
        "quantiles",
        "heads",
        "seeds",
    ]
    assert first["representations"]["pca"]["fixed_width_candidates"] == [64, 128, 256, 512]
    assert "inner_validation" in first["representations"]["pca"]["winner"]
    assert first["representations"]["supervised_projection"]["pca_is_baseline_not_ceiling"]
    two_stage = first["heads"]["two_stage_contract"]
    assert two_stage["teacher_distillation_default"] is False
    assert two_stage["stage_1_does_not_require_video_only_to_beat_ar"] is True
    assert "label_assisted" in two_stage["stage_1"]
    assert "zero_label" in two_stage["stage_2"]
    assert first["selection"]["stage_order"][-1].startswith("single_sealed")
    assert first["heads"]["two_stage_contract"]["veatic_evidence_selects_or_rejects"] is True
    assert first["training"]["again_numeric_configuration_reuse"] is False
    assert first["training"]["checkpoint_eligibility"].endswith("epoch_1")
    assert first["training"]["minimum_epochs_before_termination"] == 50
    assert first["training"]["last_checkpoint_preference"] is False
    assert "400_plus_epochs" in first["training"]["termination"]
    safe_residual = first["heads"]["ar_contract"]["safe_residual_candidate"]
    assert "positive" in safe_residual["checkpoint_rule"]
    assert safe_residual["benchmark_labels_used_for_fallback"] is False
    assert "row_outcome_oracle" in safe_residual["fallback_scope"]


def test_calibration_uses_all_benchmark_train_labels_and_rejects_test() -> None:
    identity, features, labels = _canonical_fixture()
    preregistration = build_event_preregistration(identity, features)
    benchmark_train_mask = benchmark_partition_mask(
        features, preregistration["split"], "train"
    )
    benchmark_train_features = features.subset(benchmark_train_mask)
    benchmark_train_labels = _label_subset(labels, benchmark_train_mask)

    calibration = calibrate_event_preregistration(
        preregistration, benchmark_train_features, benchmark_train_labels
    )

    assert calibration["benchmark_train_video_count"] == 124
    assert calibration["benchmark_test_video_count"] == 124
    assert calibration["benchmark_train_row_count"] == int(benchmark_train_mask.sum())
    assert calibration["benchmark_test_labels_accessed"] is False
    assert len(calibration["target_hypotheses"]) >= 2
    assert all(
        target["benchmark_train_support"]["fold_support"]
        for target in calibration["target_hypotheses"]
    )
    assert calibration["movement_curve"]["median_absolute_change"][-1] > 0.0

    with pytest.raises(ValueError, match="exactly the eligible benchmark-train rows"):
        calibrate_event_preregistration(preregistration, features, labels)
