from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.scripts.veatic21_pca import (
    ALLOWED_WIDTHS,
    DATASET_ID,
    Veatic21PcaAccessor,
    Veatic21PcaError,
    fit_or_load_pca,
)


def _synthetic_problem(
    *, seed: int = 17, feature_width: int = 140
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows_per_video = 40
    video_count = 5
    row_count = rows_per_video * video_count
    matrix = rng.normal(size=(row_count, feature_width)).astype(np.float32)
    # Stable video and time structure prevents the family transforms from
    # becoming accidental aliases while preserving a comfortably full rank.
    time = np.arange(rows_per_video, dtype=np.float32)[:, None]
    for video in range(video_count):
        start = video * rows_per_video
        stop = start + rows_per_video
        matrix[start:stop] += np.float32(video * 0.07)
        matrix[start:stop, :8] += time * np.float32(0.003 + video * 0.0002)
    video_ids = np.repeat(
        np.asarray([str(index) for index in range(video_count)], dtype=np.str_),
        rows_per_video,
    )
    train_rows = np.arange(0, rows_per_video * 4, dtype=np.int64)
    held_out_rows = np.arange(rows_per_video * 4, row_count, dtype=np.int64)
    quality = np.ones(row_count, dtype=bool)
    quality[[3, 47]] = False
    return matrix, video_ids, train_rows, held_out_rows, quality


def _fit(
    tmp_path: Path,
    matrix: np.ndarray,
    video_ids: np.ndarray,
    train_rows: np.ndarray,
    held_out_rows: np.ndarray,
    quality: np.ndarray,
    *,
    family: str = "current",
    width: int = 64,
    cache_digest: str = "cache-seal-a",
    contract_digest: str = "contract-seal-a",
    artifact_key: str = "fold-1",
    seed: int = 901,
):
    accessor = Veatic21PcaAccessor(
        matrix,
        video_ids,
        base_family=family,
        cache_digest=cache_digest,
    )
    fitted = fit_or_load_pca(
        accessor,
        train_row_indices=train_rows,
        held_out_row_indices=held_out_rows,
        quality_mask=quality,
        output_root=tmp_path,
        artifact_key=artifact_key,
        width=width,
        seed=seed,
        contract_digest=contract_digest,
        batch_size=31,
        oversampling=8,
        power_iterations=1,
    )
    return accessor, fitted


def test_accessor_implements_causal_family_semantics() -> None:
    matrix = np.arange(16, dtype=np.float32).reshape(8, 2)
    video_ids = np.asarray(["a"] * 4 + ["b"] * 4)

    current = Veatic21PcaAccessor(
        matrix, video_ids, base_family="current", cache_digest="cache"
    )
    delta = Veatic21PcaAccessor(
        matrix, video_ids, base_family="delta", cache_digest="cache"
    )
    temporal = Veatic21PcaAccessor(
        matrix, video_ids, base_family="temporal_mean_2s", cache_digest="cache"
    )

    np.testing.assert_array_equal(current.batch(np.asarray([0, 5])), matrix[[0, 5]])
    np.testing.assert_array_equal(delta.batch(np.asarray([1, 5])), matrix[[1, 5]] - matrix[[0, 4]])
    with pytest.raises(Veatic21PcaError, match="family-invalid"):
        delta.batch(np.asarray([4]))
    np.testing.assert_allclose(
        temporal.batch(np.asarray([0, 2, 3, 4, 6])),
        np.stack(
            [
                matrix[0],
                matrix[0:3].mean(axis=0),
                matrix[0:4].mean(axis=0),
                matrix[4],
                matrix[4:7].mean(axis=0),
            ]
        ),
    )


def test_held_out_values_cannot_change_fitted_pca(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    changed = matrix.copy()
    changed[held_out_rows] = changed[held_out_rows] * np.float32(999.0) + np.float32(12345.0)

    _, first = _fit(
        tmp_path / "first",
        matrix,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        cache_digest="cache-original",
    )
    _, second = _fit(
        tmp_path / "second",
        changed,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        cache_digest="cache-heldout-mutated",
    )

    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.components, second.components)
    np.testing.assert_array_equal(first.singular_values, second.singular_values)
    identity = first.metadata["identity"]
    assert identity["dataset_id"] == DATASET_ID
    assert identity["no_held_out_participation_audit"] is True
    assert identity["held_out_row_overlap_count"] == 0
    assert identity["held_out_video_overlap_count"] == 0
    assert set(identity["train_video_ids"]).isdisjoint(identity["held_out_video_ids"])


def test_grouped_video_overlap_fails_before_fit(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    # Row 159 and row 120 belong to the same training video.  Supplying one as
    # held out must fail even though the row arrays themselves do not overlap.
    partial_train = train_rows[:-1]
    partial_held_out = np.sort(np.concatenate([np.asarray([159]), held_out_rows])).astype(np.int64)
    accessor = Veatic21PcaAccessor(
        matrix, video_ids, base_family="current", cache_digest="cache"
    )
    with pytest.raises(Veatic21PcaError, match="held-out videos"):
        fit_or_load_pca(
            accessor,
            train_row_indices=partial_train,
            held_out_row_indices=partial_held_out,
            quality_mask=quality,
            output_root=tmp_path,
            artifact_key="overlap",
            width=64,
            seed=1,
            contract_digest="contract",
        )


def test_resume_requires_exact_scientific_identity(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    accessor, first = _fit(
        tmp_path, matrix, video_ids, train_rows, held_out_rows, quality
    )
    resumed = fit_or_load_pca(
        accessor,
        train_row_indices=train_rows,
        held_out_row_indices=held_out_rows,
        quality_mask=quality,
        output_root=tmp_path,
        artifact_key="fold-1",
        width=64,
        seed=901,
        contract_digest="contract-seal-a",
        batch_size=31,
        oversampling=8,
        power_iterations=1,
    )
    assert first.cache_hit is False
    assert resumed.cache_hit is True
    np.testing.assert_array_equal(resumed.components, first.components)

    with pytest.raises(Veatic21PcaError, match="resume identity mismatch"):
        fit_or_load_pca(
            accessor,
            train_row_indices=train_rows,
            held_out_row_indices=held_out_rows,
            quality_mask=quality,
            output_root=tmp_path,
            artifact_key="fold-1",
            width=64,
            seed=901,
            contract_digest="changed-contract",
            batch_size=31,
            oversampling=8,
            power_iterations=1,
        )


def test_width_and_family_are_distinct_artifact_identities(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    _, current64 = _fit(
        tmp_path,
        matrix,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        family="current",
        width=64,
    )
    _, temporal64 = _fit(
        tmp_path,
        matrix,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        family="temporal_mean_2s",
        width=64,
    )
    _, current128 = _fit(
        tmp_path,
        matrix,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        family="current",
        width=128,
    )

    assert ALLOWED_WIDTHS == (64, 128, 256)
    assert len({current64.component_path, temporal64.component_path, current128.component_path}) == 3
    identities = {
        fitted.metadata["identity_sha256"]
        for fitted in (current64, temporal64, current128)
    }
    assert len(identities) == 3
    assert current64.components.shape == (64, matrix.shape[1])
    assert current128.components.shape == (128, matrix.shape[1])


def test_quality_excluded_outlier_never_participates_in_fit(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    changed = matrix.copy()
    changed[3] = np.float32(1e20)

    _, first = _fit(
        tmp_path / "first",
        matrix,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        cache_digest="cache-before-outlier",
    )
    _, second = _fit(
        tmp_path / "second",
        changed,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        cache_digest="cache-after-outlier",
    )

    assert 3 not in first.fit_row_indices
    assert first.metadata["identity"]["quality_excluded_train_row_count"] == 2
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.components, second.components)
    np.testing.assert_array_equal(first.singular_values, second.singular_values)


def test_transform_preserves_arbitrary_alignment_and_delta_mask(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    accessor, fitted = _fit(
        tmp_path,
        matrix,
        video_ids,
        train_rows,
        held_out_rows,
        quality,
        family="delta",
    )
    requested = np.asarray([161, 5, 160, 3, 42, 0, 199, 161], dtype=np.int64)
    transformed = fitted.transform(accessor, requested, batch_size=3)

    np.testing.assert_array_equal(transformed.row_indices, requested)
    np.testing.assert_array_equal(
        transformed.family_valid_mask,
        np.asarray([True, True, False, True, True, False, True, True]),
    )
    assert np.isnan(transformed.values[[2, 5]]).all()
    assert np.isfinite(transformed.values[[0, 1, 3, 4, 6, 7]]).all()
    # Row 3 was excluded from fitting for quality, but remains transformable
    # solely to retain the aligned causal grid.
    assert np.isfinite(transformed.values[3]).all()
    for position in (0, 1, 3, 4, 6, 7):
        one = fitted.transform(accessor, requested[[position]])
        # BLAS may accumulate a 1-row and 3-row matmul in a slightly different
        # order; row identity and numerical alignment must still be exact to
        # float32 precision.
        np.testing.assert_allclose(
            transformed.values[position], one.values[0], rtol=2e-5, atol=2e-6
        )
    np.testing.assert_array_equal(transformed.values[0], transformed.values[7])
    assert transformed.metadata["alignment_preserved"] is True


def test_transform_rejects_cache_or_row_video_provenance_drift(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    accessor, fitted = _fit(
        tmp_path, matrix, video_ids, train_rows, held_out_rows, quality
    )
    assert np.isfinite(fitted.transform(accessor, held_out_rows[:2]).values).all()

    wrong_cache = Veatic21PcaAccessor(
        matrix,
        video_ids,
        base_family="current",
        cache_digest="a-different-cache-seal",
    )
    with pytest.raises(Veatic21PcaError, match="cache_digest"):
        fitted.transform(wrong_cache, held_out_rows[:2])

    relabeled = np.repeat(np.asarray([f"x{index}" for index in range(5)]), 40)
    wrong_layout = Veatic21PcaAccessor(
        matrix,
        relabeled,
        base_family="current",
        cache_digest="cache-seal-a",
    )
    with pytest.raises(Veatic21PcaError, match="row_video_layout_digest"):
        fitted.transform(wrong_layout, held_out_rows[:2])


def test_shape_nonfinite_and_insufficient_rank_fail_closed(tmp_path: Path) -> None:
    matrix, video_ids, train_rows, held_out_rows, quality = _synthetic_problem()
    with pytest.raises(Veatic21PcaError, match="row-aligned"):
        Veatic21PcaAccessor(
            matrix, video_ids[:-1], base_family="current", cache_digest="cache"
        )

    nonfinite = matrix.copy()
    nonfinite[10, 2] = np.nan
    accessor = Veatic21PcaAccessor(
        nonfinite, video_ids, base_family="current", cache_digest="nonfinite-cache"
    )
    with pytest.raises(Veatic21PcaError, match="non-finite"):
        fit_or_load_pca(
            accessor,
            train_row_indices=train_rows,
            held_out_row_indices=held_out_rows,
            quality_mask=quality,
            output_root=tmp_path / "nonfinite",
            artifact_key="fold",
            width=64,
            seed=1,
            contract_digest="contract",
            oversampling=8,
            power_iterations=0,
        )

    row_signal = np.linspace(-1.0, 1.0, len(matrix), dtype=np.float32)[:, None]
    low_rank = np.repeat(row_signal, matrix.shape[1], axis=1)
    low_rank_accessor = Veatic21PcaAccessor(
        low_rank,
        video_ids,
        base_family="current",
        cache_digest="low-rank-cache",
    )
    with pytest.raises(Veatic21PcaError, match="insufficient numerical rank"):
        fit_or_load_pca(
            low_rank_accessor,
            train_row_indices=train_rows,
            held_out_row_indices=held_out_rows,
            quality_mask=quality,
            output_root=tmp_path / "low-rank",
            artifact_key="fold",
            width=64,
            seed=1,
            contract_digest="contract",
            batch_size=31,
            oversampling=8,
            power_iterations=0,
        )
