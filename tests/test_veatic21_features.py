from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from backend.scripts import veatic21_features as features


def _source_arrays(width: int = 64) -> tuple[np.ndarray, ...]:
    row_idx = np.asarray([10, 11, 12, 13, 30, 31, 32], dtype=np.int64)
    video_id = np.asarray(["video_a"] * 4 + ["video_b"] * 3)
    time_seconds = np.asarray([0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0], dtype=np.float32)
    pca = (
        np.arange(len(row_idx) * width, dtype=np.float32).reshape(len(row_idx), width)
        / 100.0
    )
    diagnostics = (
        np.arange(len(row_idx) * features.DIAGNOSTIC_WIDTH, dtype=np.float32).reshape(
            len(row_idx), features.DIAGNOSTIC_WIDTH
        )
        / 1000.0
    )
    return row_idx, video_id, time_seconds, pca, diagnostics


def _build(width: int = 64) -> features.Veatic21Features:
    row_idx, video_id, time_seconds, pca, diagnostics = _source_arrays(width)
    return features.build_veatic21_features(
        row_idx=row_idx,
        video_id=video_id,
        time_seconds=time_seconds,
        pca_scores=pca,
        diagnostics=diagnostics,
        pca_width=width,
        pca_row_idx=row_idx.copy(),
        diagnostic_row_idx=row_idx.copy(),
    )


@pytest.mark.parametrize("width", features.ALLOWED_PCA_WIDTHS)
def test_variable_pca_widths_have_deterministic_sealed_schemas(width: int) -> None:
    block = _build(width)
    expected_temporal_width = features.WINDOW_ROWS * width + features.DIAGNOSTIC_WIDTH + 7
    expected_current_width = width + features.DIAGNOSTIC_WIDTH + 3

    assert block.x_temporal.shape == (7, expected_temporal_width)
    assert block.x_current.shape == (7, expected_current_width)
    assert len(block.feature_names) == expected_temporal_width
    assert len(block.current_feature_names) == expected_current_width
    assert block.schema_digest == features.feature_schema_digest(width)
    assert features.feature_schema_digest(width) == features.feature_schema_digest(width)
    assert features.audit_veatic21_features(block).passed is True


def test_schema_digests_are_width_specific_and_invalid_widths_fail() -> None:
    digests = {features.feature_schema_digest(width) for width in features.ALLOWED_PCA_WIDTHS}
    assert len(digests) == len(features.ALLOWED_PCA_WIDTHS)
    assert all(len(digest) == 64 for digest in digests)
    with pytest.raises(ValueError, match="Unsupported PCA width"):
        features.feature_names(32)


def test_temporal_sequence_is_same_video_causal_and_preserves_row0_cold_start() -> None:
    block = _build(64)
    _, _, _, pca, diagnostics = _source_arrays(64)
    sequence = block.x_temporal[:, : features.WINDOW_ROWS * 64].reshape(7, 5, 64)

    expected_masks = np.asarray(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(block.history_mask, expected_masks)
    assert np.array_equal(sequence[4, :-1], np.zeros((4, 64), dtype=np.float32))
    assert np.array_equal(sequence[4, -1], pca[4])
    assert np.array_equal(sequence[5, -2], pca[4])
    assert np.array_equal(sequence[5, -1], pca[5])
    assert np.array_equal(block.x_current[:, :64], pca)
    assert np.array_equal(block.x_current[:, 64 : 64 + 53], diagnostics)
    assert block.x_temporal[0, -1] == pytest.approx(0.0)
    assert block.x_temporal[3, -1] == pytest.approx(1.0)
    assert block.x_temporal[4, -1] == pytest.approx(0.0)
    assert block.x_temporal[6, -1] == pytest.approx(1.0)


def test_builder_api_and_schema_contain_no_response_or_label_inputs() -> None:
    parameters = set(inspect.signature(features.build_veatic21_features).parameters)
    forbidden = {"arousal", "valence", "target", "label", "teacher", "ar_score"}
    assert parameters.isdisjoint(forbidden)
    block = _build()
    all_names = block.temporal_feature_names + block.current_feature_names
    assert not any(
        token in name.lower()
        for name in all_names
        for token in features.FORBIDDEN_RESPONSE_TOKENS
    )


def test_audit_fails_closed_on_forbidden_response_name() -> None:
    block = _build()
    bad_names = list(block.temporal_feature_names)
    bad_names[0] = "observed_arousal_lag1"
    corrupted = replace(block, temporal_feature_names=tuple(bad_names))
    with pytest.raises(features.Veatic21FeatureAuditError, match="Forbidden response-side"):
        features.audit_veatic21_features(corrupted)


def test_audit_fails_closed_on_cross_video_history() -> None:
    block = _build()
    corrupted_x = block.x_temporal.copy()
    lag1_slice = slice(3 * block.pca_width, 4 * block.pca_width)
    corrupted_x[4, lag1_slice] = block.x_current[3, : block.pca_width]
    corrupted = replace(block, x_temporal=corrupted_x)
    with pytest.raises(features.Veatic21FeatureAuditError, match="cross-video history"):
        features.audit_veatic21_features(corrupted)


def test_builder_fails_closed_on_source_row_misalignment() -> None:
    row_idx, video_id, time_seconds, pca, diagnostics = _source_arrays()
    shifted_pca_rows = row_idx + 1
    with pytest.raises(features.Veatic21FeatureAuditError, match="pca_row_idx is misaligned"):
        features.build_veatic21_features(
            row_idx=row_idx,
            video_id=video_id,
            time_seconds=time_seconds,
            pca_scores=pca,
            diagnostics=diagnostics,
            pca_row_idx=shifted_pca_rows,
            diagnostic_row_idx=row_idx,
        )
    with pytest.raises(features.Veatic21FeatureAuditError, match="diagnostics is misaligned"):
        features.build_veatic21_features(
            row_idx=row_idx,
            video_id=video_id,
            time_seconds=time_seconds,
            pca_scores=pca,
            diagnostics=diagnostics[:-1],
        )


def test_builder_rejects_ambiguous_row_and_video_ordering() -> None:
    row_idx, video_id, time_seconds, pca, diagnostics = _source_arrays()
    with pytest.raises(features.Veatic21FeatureAuditError, match="strictly increasing"):
        features.build_veatic21_features(
            row_idx=np.asarray([10, 11, 10, 13, 30, 31, 32]),
            video_id=video_id,
            time_seconds=time_seconds,
            pca_scores=pca,
            diagnostics=diagnostics,
        )

    interleaved_video = video_id.copy()
    interleaved_video[2] = "video_b"
    with pytest.raises(features.Veatic21FeatureAuditError, match="multiple blocks"):
        features.build_veatic21_features(
            row_idx=row_idx,
            video_id=interleaved_video,
            time_seconds=np.asarray([0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.5]),
            pca_scores=pca,
            diagnostics=diagnostics,
        )


def test_non_finite_inputs_and_outputs_fail_closed() -> None:
    row_idx, video_id, time_seconds, pca, diagnostics = _source_arrays()
    bad_pca = pca.copy()
    bad_pca[2, 3] = np.nan
    with pytest.raises(features.Veatic21FeatureAuditError, match="non-finite"):
        features.build_veatic21_features(
            row_idx=row_idx,
            video_id=video_id,
            time_seconds=time_seconds,
            pca_scores=bad_pca,
            diagnostics=diagnostics,
        )

    block = _build()
    bad_current = block.x_current.copy()
    bad_current[0, 0] = np.inf
    with pytest.raises(features.Veatic21FeatureAuditError, match="non-finite"):
        features.audit_veatic21_features(replace(block, x_current=bad_current))
