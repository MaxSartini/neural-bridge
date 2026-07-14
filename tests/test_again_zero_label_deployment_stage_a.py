from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from backend.scripts import again_zero_label_deployment_stage_a as stage_a
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage0 as stage0
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage_a as runner


def test_freeze_is_exactly_stage_a_and_keeps_stage_b_locked() -> None:
    manifest = stage_a.implementation_freeze_manifest()
    assert manifest["authorization"] == stage_a.AUTHORIZATION
    assert manifest["rows"] == {"member": 72, "ensemble": 24, "total": 96}
    assert manifest["stage_b_authorized"] is False
    assert manifest["cpu_fallback"] is False
    assert manifest["pca"]["nested_teacher_train_only"] is True


def test_preflight_matches_canonical_stage0_locks() -> None:
    preflight = runner.implementation_preflight(
        stage0_root=runner.DEFAULT_STAGE0_ROOT,
        dense_root=stage0.DEFAULT_DENSE_ROOT,
        external_phase4_root=runner.DEFAULT_EXTERNAL_PHASE4_ROOT,
        output_root=runner.DEFAULT_OUTPUT_ROOT,
    )
    assert preflight["target_identity_digest"] == "446906dff30be33f204de0f973207975"
    assert preflight["development_split_digest"] == "cf65a766cd827e6201544dd753049cb4"
    assert preflight["locked_split_digest"] == "ded8bc2bf079fef91ae5c253b9a9ac2e"


def test_video_feature_schema_contains_no_response_or_target_arrays() -> None:
    names = {field.name for field in fields(stage_a.VideoFeatures)}
    assert not names & {"arousal", "target", "teacher_score", "ar_score", "ar_reg"}


def test_causal_sequence_zero_pads_and_never_crosses_video() -> None:
    current = np.arange(18, dtype=np.float32).reshape(6, 3)
    rows = np.arange(6, dtype=np.int64)
    videos = np.asarray(["a", "a", "a", "b", "b", "b"])
    sequence, mask = stage_a._causal_sequence_with_mask(current, rows, videos)
    sequence = sequence.reshape(6, stage_a.WINDOW_ROWS, 3)
    assert mask[0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert mask[3].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert np.array_equal(sequence[3, -1], current[3])
    assert not np.any(sequence[3, :-1])


def test_rollout_features_use_only_earlier_predictions_and_reset() -> None:
    prediction = np.asarray([0.1, 0.2, 0.3, 0.7, 0.8], dtype=np.float32)
    videos = np.asarray(["a", "a", "a", "b", "b"])
    features, audit = stage_a.own_prediction_ar_features(prediction, videos, 0.5)
    assert features[0].tolist() == pytest.approx([0.1, 0.5, 0.5, 0.5, -0.4, -0.4, -0.4])
    assert features[3].tolist() == pytest.approx([0.7, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
    assert audit["teacher_forcing_ratio"] == 0.0
    assert audit["cross_video_state_carry"] == 0
    assert audit["all_finite"] is True


def test_video_mapping_is_deterministic_and_deranged() -> None:
    videos = ["a", "b", "c", "d"]
    counts = {"a": 5, "b": 7, "c": 6, "d": 8}
    first = stage_a.deterministic_video_mapping(videos, counts, "unit")
    second = stage_a.deterministic_video_mapping(videos, counts, "unit")
    assert first == second
    assert all(source != donor for source, donor in first.items())
    assert set(first) == set(first.values())


def test_sequence_reassignment_preserves_recipient_shape() -> None:
    videos = np.asarray(["a"] * 3 + ["b"] * 5 + ["c"] * 4)
    values = np.arange(len(videos) * 2, dtype=np.float32).reshape(len(videos), 2)
    reassigned, mapping = stage_a.reassign_video_sequences(values, videos, "unit")
    assert reassigned.shape == values.shape
    assert np.isfinite(reassigned).all()
    assert all(source != donor for source, donor in mapping.items())


def test_event_scorer_fails_closed_for_single_class_first30() -> None:
    train = np.linspace(0, 1, 100)
    test = np.linspace(0, 1, 20)
    prediction = test.copy()
    times = np.asarray(list(range(5)) + list(range(30, 45)), dtype=float)
    with pytest.raises(ValueError):
        stage_a.score_prediction(
            train_values=train,
            test_values=test,
            prediction=prediction,
            time_seconds=times,
        )


def test_ensemble_requires_three_aligned_members() -> None:
    with pytest.raises(ValueError):
        stage_a.ensemble_predictions([np.zeros(3), np.zeros(3)])
    with pytest.raises(ValueError):
        stage_a.ensemble_predictions([np.zeros(3), np.zeros(4), np.zeros(3)])
    assert np.array_equal(
        stage_a.ensemble_predictions([np.ones(3), np.ones(3) * 2, np.ones(3) * 3]),
        np.ones(3) * 2,
    )


def _synthetic_rows(*, h1_pass: bool = True) -> pd.DataFrame:
    lane_values = {
        "video_distilled_temporal": (0.18 if h1_pass else 0.11, 0.08, 0.19),
        "video_closed_loop_rollout": (0.12, 0.055, 0.13),
        "video_supervised_temporal": (0.13, 0.06, 0.14),
        "video_supervised_current_row": (0.125, 0.058, 0.135),
        "no_video_closed_loop_persistence": (0.10, 0.05, 0.12),
        "sequence_shuffled_video": (0.11, 0.052, 0.125),
        "video_label_permutation": (0.105, 0.051, 0.122),
        "phase7_ar_assisted_teacher_ceiling": (0.24, 0.10, 0.24),
    }
    rows = []
    for fold in stage_a.OUTER_FOLDS:
        for lane in stage_a.LANES:
            spearman, top5, event = lane_values[lane]
            for row_type, groups in (
                ("member", [str(seed) for seed in stage_a.SEEDS]),
                ("ensemble", [stage_a.GROUP_NAME]),
            ):
                for group in groups:
                    member_penalty = 0.002 if row_type == "member" and lane == "video_distilled_temporal" else 0.0
                    row = {
                        "stage": "stage_a",
                        "split_digest": f"fold{fold}",
                        "fold": fold,
                        "lane": lane,
                        "row_type": row_type,
                        "seed_or_group": group,
                        "cold_start_policy": "row0_zero_history_no_label_burnin",
                        "pooled_continuous_spearman": spearman - member_penalty,
                        "top_5pct_true_future_movement_lift": top5 - member_penalty,
                        "training_q90_future_event_pr_auc": event - member_penalty,
                        "first30_pooled_continuous_spearman": spearman,
                        "first30_top_5pct_true_future_movement_lift": top5,
                        "first30_training_q90_future_event_pr_auc": event,
                        "teacher_forcing_ratio": 0.0,
                        "cross_video_state_carry": 0,
                        "rollout_all_finite": True,
                    }
                    for metric in stage_a.REQUIRED_METRICS:
                        row[f"compat_{metric}"] = row[metric]
                    rows.append(row)
    return pd.DataFrame(rows)


def test_verdict_passes_only_complete_conjunctive_candidate() -> None:
    result = stage_a.compute_stage_a_verdict(_synthetic_rows(), audit_pass=True)
    assert result["rows_actual"] == 96
    assert result["stage_a_pass"] is True
    assert result["locked_winner"] == "video_distilled_temporal"
    assert result["stage_b_authorized"] is False


def test_verdict_fails_closed_on_scope_or_audit() -> None:
    frame = _synthetic_rows()
    assert stage_a.compute_stage_a_verdict(frame.iloc[:-1], True)["stage_a_pass"] is False
    assert stage_a.compute_stage_a_verdict(frame, False)["stage_a_pass"] is False


def test_verdict_does_not_rescue_candidate_that_loses_one_endpoint() -> None:
    result = stage_a.compute_stage_a_verdict(_synthetic_rows(h1_pass=False), True)
    assert result["candidate_results"]["video_distilled_temporal"]["qualified"] is False
    assert result["stage_a_pass"] is False


def test_scored_key_includes_cold_start_policy() -> None:
    row = _synthetic_rows().iloc[0].to_dict()
    key = stage_a.scored_row_key(row)
    assert len(key) == 7
    assert key[-1] == "row0_zero_history_no_label_burnin"


def test_required_metrics_are_spike_and_continuous_conjunctively() -> None:
    assert stage_a.REQUIRED_METRICS == (
        "pooled_continuous_spearman",
        "top_5pct_true_future_movement_lift",
        "training_q90_future_event_pr_auc",
    )
