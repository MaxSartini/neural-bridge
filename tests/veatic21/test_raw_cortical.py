from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.raw_cortical import (
    expand_control_to_width,
    fit_raw_discriminant_mlx,
    predict_raw_discriminant_mlx,
    shape_matched_random,
    within_partition_video_shuffle,
    within_video_label_permutation,
)


def test_raw_discriminant_uses_all_columns_and_scores_separated_rows_on_gpu() -> None:
    features = np.asarray(
        [
            [-2.0, 0.0, -1.0],
            [-1.0, 0.2, -2.0],
            [1.0, -0.1, 2.0],
            [2.0, 0.1, 1.0],
        ],
        dtype=np.float16,
    )
    labels = np.asarray((0, 0, 1, 1), dtype=np.int8)
    model = fit_raw_discriminant_mlx(features, labels)
    scores = predict_raw_discriminant_mlx(model, features)

    assert model.device == "gpu:0"
    assert model.mean.shape == (3,)
    assert np.min(scores[labels == 1]) > np.max(scores[labels == 0])


def test_small_nuisance_control_is_tiled_to_exact_declared_width() -> None:
    base = np.asarray(((1.0, 2.0), (3.0, 4.0)))
    expanded = expand_control_to_width(base, width=5)

    assert expanded.dtype == np.float16
    assert expanded.tolist() == [[1.0, 2.0, 1.0, 2.0, 1.0], [3.0, 4.0, 3.0, 4.0, 3.0]]


def test_shuffle_stays_within_video_partition_and_is_nonidentity() -> None:
    partition = np.arange(12)
    video_id = np.repeat(np.arange(3), 4)
    source = within_partition_video_shuffle(partition, video_id, seed=7)

    assert not np.array_equal(source, partition)
    assert np.array_equal(video_id[source], video_id[partition])
    assert set(source) == set(partition)


def test_label_permutation_preserves_video_and_global_support() -> None:
    indices = np.arange(12)
    video_id = np.repeat(np.arange(3), 4)
    labels = np.tile(np.asarray((0, 0, 1, 1), dtype=np.int8), 3)
    permuted = within_video_label_permutation(labels, video_id, indices, seed=9)

    assert not np.array_equal(permuted, labels)
    assert int(permuted.sum()) == int(labels.sum())
    for video in np.unique(video_id):
        mask = video_id == video
        assert int(permuted[mask].sum()) == int(labels[mask].sum())


def test_shape_matched_random_is_deterministic_and_exact() -> None:
    first = shape_matched_random(5, 7, seed=11, chunk_rows=2)
    second = shape_matched_random(5, 7, seed=11, chunk_rows=2)

    assert first.shape == (5, 7)
    assert first.dtype == np.float16
    assert np.array_equal(first, second)
    assert set(np.unique(first)) == {-1.0, 1.0}
