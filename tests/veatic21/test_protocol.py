from __future__ import annotations

import hashlib
from typing import Any, cast

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import (
    CandidateSpec,
    CellSpec,
    FeatureRows,
    FrozenRecipe,
    LabelRows,
    TargetSpec,
)
from neural_bridge.veatic21.protocol import (
    assert_row_alignment,
    build_video_splits,
    causal_ar_features,
    event_labels,
    fit_event_threshold,
    freeze_final_recipe,
    freeze_winner,
    future_target_values,
    target_support_mask,
)


def test_nested_video_splits_are_stable_balanced_and_group_disjoint() -> None:
    video_ids = np.repeat([f"v{index:02d}" for index in range(17)], 3)

    first = build_video_splits(video_ids)
    repeated = build_video_splits(video_ids[::-1])
    changed_seed = build_video_splits(video_ids, split_seed=20_260_722)

    assert first == repeated
    assert len(first) == 5
    assert len({split.digest for split in first}) == 5
    assert [split.digest for split in first] != [split.digest for split in changed_seed]
    for split in first:
        outer_train = set(split.train_video_ids)
        outer_test = set(split.test_video_ids)
        assert outer_train.isdisjoint(outer_test)
        assert outer_train | outer_test == set(video_ids)
        assert len(split.inner_splits) == 3
        inner_validation: list[str] = []
        for train, validation in split.inner_splits:
            assert set(train).isdisjoint(validation)
            assert set(train) | set(validation) == outer_train
            inner_validation.extend(validation)
        assert sorted(inner_validation) == sorted(outer_train)
        assert len(split.digest) == 64


def _aligned_rows() -> tuple[FeatureRows, LabelRows, TargetSpec]:
    video_id = np.array(["a", "a", "a", "b", "b", "b"])
    row_index = np.array([0, 1, 2, 0, 1, 2])
    time_seconds = row_index.astype(np.float64) / 2.0
    features = FeatureRows(
        video_id=video_id,
        row_index=row_index,
        time_seconds=time_seconds,
        quality_eligible=np.ones(6, dtype=bool),
        representations={"tribe_cortical": np.ones((6, 2))},
    )
    labels = LabelRows(
        video_id=video_id,
        row_index=row_index,
        time_seconds=time_seconds,
        arousal=np.array([0.0, 1.0, 4.0, 10.0, 11.0, 13.0]),
        valence=np.zeros(6),
    )
    return features, labels, TargetSpec("synthetic", "arousal", (1, 2), 0.5)


def test_targets_and_causal_ar_use_row_identity_without_crossing_videos() -> None:
    features, labels, target = _aligned_rows()
    expected_support = np.array([True, False, False, True, False, False])

    assert_row_alignment(features, labels)
    np.testing.assert_array_equal(target_support_mask(features, target), expected_support)
    np.testing.assert_array_equal(
        target_support_mask(features.video_id, features.row_index, target), expected_support
    )
    future = future_target_values(labels, target)
    np.testing.assert_allclose(future[expected_support], [4.0, 3.0])
    assert np.isnan(future[~expected_support]).all()

    ar_values, available = causal_ar_features(labels, target, lag_rows=(1, 2))
    assert ar_values.shape == available.shape == (6, 2)
    np.testing.assert_array_equal(available[0], [False, False])
    np.testing.assert_array_equal(available[2], [True, True])
    np.testing.assert_array_equal(available[3], [False, False])
    np.testing.assert_allclose(ar_values[2], [1.0, 0.0])
    np.testing.assert_allclose(ar_values[3], [0.0, 0.0])

    misaligned = LabelRows(
        video_id=labels.video_id[::-1],
        row_index=labels.row_index,
        time_seconds=labels.time_seconds,
        arousal=labels.arousal,
        valence=labels.valence,
    )
    with pytest.raises(ValueError, match="not exactly aligned"):
        assert_row_alignment(features, misaligned)


def test_event_threshold_is_owned_only_by_declared_training_rows() -> None:
    target = TargetSpec("synthetic", "arousal", (1,), 0.5)
    train_mask = np.array([True, True, False])
    first = np.array([1.0, 2.0, 10_000.0])
    second = np.array([1.0, 2.0, -10_000.0])

    assert fit_event_threshold(first, train_mask, target) == 1.5
    assert fit_event_threshold(second, train_mask, target) == 1.5
    np.testing.assert_array_equal(event_labels(first, 1.5), [False, True, True])

    with pytest.raises(ValueError, match="future support"):
        fit_event_threshold(np.array([1.0, np.nan]), np.array([True, True]), target)


def _candidate_grid() -> tuple[CandidateSpec, CandidateSpec]:
    return (
        CandidateSpec(
            "alpha", "tribe_cortical", 2, 1.0, pca_solver="incremental", pca_batch_rows=4
        ),
        CandidateSpec("beta", "tribe_cortical", 3, 1.0, pca_solver="incremental", pca_batch_rows=4),
    )


def _inner_scores() -> list[dict[str, object]]:
    return [
        {
            "candidate": candidate,
            "inner_fold": fold,
            "pooled_pr_auc": base + fold / 100.0,
            "frozen_ar_pr_auc": 0.4,
            "delta_vs_frozen_ar": base - 0.4,
            "threshold": 0.5,
            "train_rows": 100,
            "validation_rows": 50,
        }
        for candidate, base in (("alpha", 0.7), ("beta", 0.6))
        for fold in range(3)
    ]


def test_winner_and_final_recipe_freezes_reject_leakage_and_incomplete_grids() -> None:
    candidates = _candidate_grid()
    target = TargetSpec("synthetic", "arousal", (1,), 0.5)
    cell = CellSpec(target=target, outer_fold=1, seed=17)
    inner = _inner_scores()

    winner = freeze_winner(candidates, inner, cell=cell, split_digest="a" * 64)
    repeated = freeze_winner(candidates, inner[::-1], cell=cell, split_digest="a" * 64)
    assert winner.candidate.name == "alpha"
    assert winner.digest == repeated.digest

    leaking = [dict(row) for row in inner]
    leaking[0]["outer_pr_auc"] = 1.0
    with pytest.raises(ValueError, match="held-out result"):
        freeze_winner(candidates, leaking, cell=cell, split_digest="a" * 64)
    with pytest.raises(ValueError, match="incomplete"):
        freeze_winner(candidates, inner[:-1], cell=cell, split_digest="a" * 64)

    discovery = [
        {
            "candidate": candidate,
            "outer_fold": fold,
            "pooled_pr_auc": base + fold / 100.0,
            "discovery_digest": hashlib.sha256(f"fold-{fold}".encode()).hexdigest(),
        }
        for fold in range(5)
        for candidate, base in (("alpha", 0.7), ("beta", 0.6))
    ]
    recipe = freeze_final_recipe(
        candidates,
        discovery,
        refit_seed=91,
    )
    reordered = freeze_final_recipe(
        candidates,
        discovery[::-1],
        refit_seed=91,
    )
    assert recipe.candidate.name == "alpha"
    assert recipe.refit_seed == 91
    assert recipe.promotable is False
    assert recipe.digest == reordered.digest
    assert recipe.digest != freeze_final_recipe(candidates, discovery, refit_seed=92).digest

    with pytest.raises(ValueError, match="no preregistered promotion gate"):
        freeze_final_recipe(candidates, discovery, promotable=True)
    with pytest.raises(ValueError, match="foundation cells cannot be marked promotable"):
        CellSpec(target=target, outer_fold=1, seed=17, promotable=True)
    with pytest.raises(ValueError, match="foundation recipes cannot be marked promotable"):
        FrozenRecipe(
            candidate=candidates[0],
            discovery_digests=("a" * 64,) * 5,
            outer_fold_count=5,
            selection_metric="pooled_pr_auc",
            tie_break="candidate_name_ascending",
            refit_seed=91,
            promotable=True,
            digest="b" * 64,
        )

    diagnostics = CandidateSpec(
        name="diagnostics-primary",
        representation=cast(Any, "diagnostics_only"),
        pca_width=1,
        regularization_c=1.0,
    )
    with pytest.raises(ValueError, match="primary representation"):
        diagnostics.validate()

    with pytest.raises(ValueError, match="incomplete"):
        freeze_final_recipe(candidates, discovery[:-1], refit_seed=91)
    discovery[0]["confirmation_pr_auc"] = 1.0
    with pytest.raises(ValueError, match="held-out result"):
        freeze_final_recipe(candidates, discovery, refit_seed=91)
