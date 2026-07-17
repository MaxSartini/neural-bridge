from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from backend.scripts import veatic21_controls as controls
from backend.scripts import veatic21_features as features


def _block(
    video_lengths: dict[str, int],
    *,
    width: int = 64,
    row_start: int = 0,
    pca_offset: float = 0.0,
    diagnostic_offset: float = 0.0,
) -> features.Veatic21Features:
    rows = sum(video_lengths.values())
    row_idx = np.arange(row_start, row_start + rows, dtype=np.int64)
    video_id: list[str] = []
    time_seconds: list[float] = []
    pca_parts: list[np.ndarray] = []
    cursor = 0
    for video_number, (video, length) in enumerate(video_lengths.items()):
        video_id.extend([video] * length)
        time_seconds.extend(np.arange(length, dtype=np.float32).tolist())
        row_level = np.arange(length, dtype=np.float32).reshape(-1, 1)
        column_level = np.arange(width, dtype=np.float32).reshape(1, -1) / 1000.0
        pca_parts.append(
            pca_offset + video_number * 100.0 + row_level * 10.0 + column_level
        )
        cursor += length
    pca = np.concatenate(pca_parts, axis=0).astype(np.float32)
    diagnostics = (
        diagnostic_offset
        + np.arange(rows * features.DIAGNOSTIC_WIDTH, dtype=np.float32).reshape(
            rows, features.DIAGNOSTIC_WIDTH
        )
        / 100.0
    )
    return features.build_veatic21_features(
        row_idx=row_idx,
        video_id=np.asarray(video_id),
        time_seconds=np.asarray(time_seconds, dtype=np.float32),
        pca_scores=pca,
        diagnostics=diagnostics,
        pca_width=width,
        pca_row_idx=row_idx,
        diagnostic_row_idx=row_idx,
    )


def _pair(width: int = 64) -> tuple[features.Veatic21Features, features.Veatic21Features]:
    return (
        _block({"train_a": 4, "train_b": 6, "train_c": 5}, width=width),
        _block(
            {"test_a": 5, "test_b": 3},
            width=width,
            row_start=100,
            pca_offset=1000.0,
            diagnostic_offset=5.0,
        ),
    )


def _namespace(lane: str, *, endpoint: str = "continuous") -> controls.ControlNamespace:
    return controls.ControlNamespace(
        target="future_arousal_max_delta_rows_4_10",
        fold=2,
        seed=20260716,
        endpoint=endpoint,
        lane=lane,
    )


def _pca(block: features.Veatic21Features) -> np.ndarray:
    return block.x_current[:, : block.pca_width]


def _diagnostics(block: features.Veatic21Features) -> np.ndarray:
    width = block.pca_width
    return block.x_current[:, width : width + features.DIAGNOSTIC_WIDTH]


def _reseal(record: dict[str, object]) -> dict[str, object]:
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    return {**unsigned, "record_digest": controls.canonical_digest(unsigned)}


def test_namespace_is_complete_deterministic_and_strict() -> None:
    namespace = _namespace("ar_plus_random_pca_residual")
    assert namespace.as_dict() == {
        "target": "future_arousal_max_delta_rows_4_10",
        "fold": 2,
        "seed": 20260716,
        "endpoint": "continuous",
        "lane": "ar_plus_random_pca_residual",
    }
    assert namespace.digest == _namespace("ar_plus_random_pca_residual").digest
    assert len(namespace.digest) == 64
    with pytest.raises(controls.Veatic21ControlError, match="non-empty trimmed"):
        controls.ControlNamespace("target", 0, 1, "continuous", " lane")
    with pytest.raises(controls.Veatic21ControlError, match="non-negative integer"):
        controls.ControlNamespace("target", -1, 1, "continuous", "lane")


@pytest.mark.parametrize("width", features.ALLOWED_PCA_WIDTHS)
def test_sequence_shuffle_is_whole_video_resampled_and_pca_only(width: int) -> None:
    train, test = _pair(width)
    namespace = _namespace("ar_plus_shuffled_pca_residual")
    control = controls.build_sequence_shuffled_pca_control(
        train,
        test,
        namespace=namespace,
        privileged=True,
        frozen_ar_identity="frozen-ar-member-2",
    )
    repeated = controls.build_sequence_shuffled_pca_control(
        train,
        test,
        namespace=namespace,
        privileged=True,
        frozen_ar_identity="frozen-ar-member-2",
    )

    assert np.array_equal(control.train.row_idx, train.row_idx)
    assert np.array_equal(control.test.row_idx, test.row_idx)
    assert np.array_equal(control.train.video_id, train.video_id)
    assert np.array_equal(_diagnostics(control.train), _diagnostics(train))
    assert np.array_equal(control.train.history_mask, train.history_mask)
    assert np.array_equal(control.train.x_temporal[:, -2:], train.x_temporal[:, -2:])
    assert np.array_equal(control.train.x_current, repeated.train.x_current)
    assert control.record == repeated.record

    mapping = control.record["construction"]["train_donor_mapping"]
    assert all(recipient != donor for recipient, donor in mapping.items())
    for recipient, donor in mapping.items():
        recipient_rows = np.flatnonzero(train.video_id == recipient)
        donor_rows = np.flatnonzero(train.video_id == donor)
        assert np.allclose(_pca(control.train)[recipient_rows[0]], _pca(train)[donor_rows[0]])
        assert np.allclose(_pca(control.train)[recipient_rows[-1]], _pca(train)[donor_rows[-1]])
    assert control.record["frozen_ar_identity"]["identity_digest"] == "frozen-ar-member-2"
    assert controls.audit_feature_control(control).passed is True


def test_sequence_shuffle_allows_self_map_only_for_single_video_split() -> None:
    train = _block({"only_train": 4})
    test = _block({"only_test": 3}, row_start=50, pca_offset=500.0)
    control = controls.build_sequence_shuffled_pca_control(
        train,
        test,
        namespace=_namespace("sequence_shuffled_supervised_temporal", endpoint="zero_label"),
    )
    assert control.record["construction"]["train_donor_mapping"] == {
        "only_train": "only_train"
    }
    assert np.array_equal(_pca(control.train), _pca(train))


def test_mapping_self_assignment_tamper_fails_closed() -> None:
    train, test = _pair()
    control = controls.build_sequence_shuffled_pca_control(
        train,
        test,
        namespace=_namespace("sequence_shuffled_supervised_temporal", endpoint="zero_label"),
    )
    record = dict(control.record)
    construction = dict(record["construction"])
    mapping = dict(construction["train_donor_mapping"])
    recipient = next(iter(mapping))
    mapping[recipient] = recipient
    construction["train_donor_mapping"] = mapping
    construction["train_mapping_digest"] = controls.canonical_digest(mapping)
    record["construction"] = construction
    corrupted = replace(control, record=_reseal(record))
    with pytest.raises(controls.Veatic21ControlError, match="not a permutation|self-map"):
        controls.audit_feature_control(corrupted)


def test_random_pca_uses_only_outer_train_fit_distribution() -> None:
    train, test = _pair(128)
    fit_mask = np.ones(len(train.row_idx), dtype=bool)
    fit_mask[[0, 7, 14]] = False
    namespace = _namespace("ar_plus_random_pca_residual")
    control = controls.build_matched_random_pca_control(
        train,
        test,
        namespace=namespace,
        frozen_ar_identity="ar-identity-random",
        train_fit_mask=fit_mask,
    )

    changed_test = features.build_veatic21_features(
        row_idx=test.row_idx,
        video_id=test.video_id,
        time_seconds=test.time_seconds,
        pca_scores=_pca(test) + 99_999.0,
        diagnostics=_diagnostics(test),
        pca_width=test.pca_width,
    )
    changed = controls.build_matched_random_pca_control(
        train,
        changed_test,
        namespace=namespace,
        frozen_ar_identity="ar-identity-random",
        train_fit_mask=fit_mask,
    )
    assert np.array_equal(_pca(control.test), _pca(changed.test))
    assert control.record["construction"]["test_values_used_for_fit"] is False
    assert control.record["construction"]["fit_scope"] == "outer_train_only"

    expected_mean = np.mean(_pca(train)[fit_mask], axis=0)
    expected_std = np.std(_pca(train)[fit_mask], axis=0)
    assert np.allclose(control.parameter_arrays["fit_mean"], expected_mean, atol=1e-5)
    assert np.allclose(control.parameter_arrays["fit_std"], expected_std, atol=1e-5)
    assert np.allclose(np.mean(_pca(control.train)[fit_mask], axis=0), expected_mean, atol=1e-4)
    assert np.allclose(np.std(_pca(control.train)[fit_mask], axis=0), expected_std, atol=1e-4)
    assert np.array_equal(_diagnostics(control.train), _diagnostics(train))


def test_train_video_mean_uses_global_train_fallback_for_unseen_videos() -> None:
    train, test = _pair()
    fit_mask = train.video_id != "train_c"
    control = controls.build_train_only_video_mean_control(
        train,
        test,
        namespace=_namespace("ar_plus_train_only_video_mean_residual"),
        frozen_ar_identity="ar-identity-mean",
        train_fit_mask=fit_mask,
    )
    global_mean = np.mean(_pca(train)[fit_mask], axis=0)
    assert np.allclose(control.parameter_arrays["global_mean"], global_mean)
    assert control.record["construction"]["training_videos_without_fit_rows"] == [
        "train_c"
    ]
    assert np.allclose(_pca(control.train)[train.video_id == "train_c"], global_mean)
    assert np.allclose(_pca(control.test), global_mean)
    assert np.array_equal(_diagnostics(control.test), _diagnostics(test))
    assert control.record["construction"]["held_out_global_fallback_rows"] == len(
        test.row_idx
    )


def test_diagnostics_only_and_no_video_views_preserve_the_sealed_shape() -> None:
    train, test = _pair(256)
    diagnostics_only = controls.build_diagnostics_only_control(
        train,
        test,
        namespace=_namespace("diagnostics_only_supervised_temporal", endpoint="zero_label"),
    )
    no_video = controls.build_no_video_control(
        train,
        test,
        namespace=_namespace("no_video_supervised_temporal", endpoint="zero_label"),
    )
    assert np.count_nonzero(_pca(diagnostics_only.train)) == 0
    assert np.array_equal(_diagnostics(diagnostics_only.train), _diagnostics(train))
    assert np.count_nonzero(_pca(no_video.train)) == 0
    assert np.count_nonzero(_diagnostics(no_video.train)) == 0
    assert np.array_equal(no_video.train.history_mask, train.history_mask)
    assert np.array_equal(no_video.train.x_temporal[:, -2:], train.x_temporal[:, -2:])
    assert diagnostics_only.train.x_temporal.shape == train.x_temporal.shape
    assert no_video.test.x_current.shape == test.x_current.shape


def test_response_leakage_overlap_and_nonfinite_sources_fail_closed() -> None:
    train, test = _pair()
    names = list(train.temporal_feature_names)
    names[0] = "future_valence_teacher"
    leaking = replace(train, temporal_feature_names=tuple(names))
    with pytest.raises(ValueError, match="Forbidden response-side"):
        controls.build_no_video_control(
            leaking,
            test,
            namespace=_namespace("no_video", endpoint="zero_label"),
        )

    overlap = _block({"train_a": 3, "test_x": 3}, row_start=1000)
    with pytest.raises(controls.Veatic21ControlError, match="video overlap"):
        controls.build_no_video_control(
            train,
            overlap,
            namespace=_namespace("no_video", endpoint="zero_label"),
        )

    corrupted_x = train.x_current.copy()
    corrupted_x[0, 0] = np.nan
    nonfinite = replace(train, x_current=corrupted_x)
    with pytest.raises(ValueError, match="non-finite"):
        controls.build_no_video_control(
            nonfinite,
            test,
            namespace=_namespace("no_video", endpoint="zero_label"),
        )


def test_privileged_controls_require_identity_and_zero_label_rejects_it() -> None:
    train, test = _pair()
    with pytest.raises(controls.Veatic21ControlError, match="requires the exact frozen AR"):
        controls.build_sequence_shuffled_pca_control(
            train,
            test,
            namespace=_namespace("ar_plus_shuffled_pca_residual"),
            privileged=True,
        )
    with pytest.raises(controls.Veatic21ControlError, match="must not carry"):
        controls.build_diagnostics_only_control(
            train,
            test,
            namespace=_namespace("diagnostics_only", endpoint="zero_label"),
            frozen_ar_identity="stale-ar-identity",
        )


def test_whole_video_label_permutation_is_deterministic_and_nearest_resampled() -> None:
    lengths = {"video_a": 3, "video_b": 5, "video_c": 4}
    block = _block(lengths)
    target_parts = [
        np.asarray([0, 1, 0], dtype=np.float32),
        np.asarray([1, 1, 0, 0, 1], dtype=np.float32),
        np.asarray([0, 0, 1, 1], dtype=np.float32),
    ]
    target = np.concatenate(target_parts)
    namespace = _namespace("label_permutation_supervised_temporal", endpoint="binary")
    control = controls.build_whole_video_label_permutation(
        row_idx=block.row_idx,
        video_id=block.video_id,
        time_seconds=block.time_seconds,
        target=target,
        namespace=namespace,
    )
    repeated = controls.build_whole_video_label_permutation(
        row_idx=block.row_idx,
        video_id=block.video_id,
        time_seconds=block.time_seconds,
        target=target,
        namespace=namespace,
    )
    assert np.array_equal(control.target, repeated.target)
    assert np.array_equal(control.row_idx, block.row_idx)
    assert set(np.unique(control.target)) <= {0.0, 1.0}
    mapping = control.record["construction"]["donor_mapping"]
    assert all(recipient != donor for recipient, donor in mapping.items())
    for recipient, donor in mapping.items():
        recipient_rows = np.flatnonzero(block.video_id == recipient)
        donor_rows = np.flatnonzero(block.video_id == donor)
        assert control.target[recipient_rows[0]] == target[donor_rows[0]]
        assert control.target[recipient_rows[-1]] == target[donor_rows[-1]]
    assert controls.audit_target_control(control).passed is True

    bad_target = target.copy()
    bad_target[-1] = np.nan
    with pytest.raises(controls.Veatic21ControlError, match="target must be finite"):
        controls.build_whole_video_label_permutation(
            row_idx=block.row_idx,
            video_id=block.video_id,
            time_seconds=block.time_seconds,
            target=bad_target,
            namespace=namespace,
        )


def test_feature_persistence_roundtrip_seals_parameters_and_reuse_identity(tmp_path) -> None:
    train, test = _pair()
    namespace = _namespace("ar_plus_random_pca_residual")
    control = controls.build_matched_random_pca_control(
        train,
        test,
        namespace=namespace,
        frozen_ar_identity="ar-persisted-identity",
    )
    paths = controls.persist_feature_control(control, tmp_path)
    loaded = controls.load_feature_control(
        paths.manifest,
        expected_namespace=namespace,
        expected_frozen_ar_identity="ar-persisted-identity",
    )
    assert loaded.record == control.record
    assert np.array_equal(loaded.train.x_temporal, control.train.x_temporal)
    assert set(loaded.parameter_arrays) == set(control.parameter_arrays)
    assert controls.persist_feature_control(control, tmp_path) == paths

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["record"]["parameter_arrays"]["fit_mean"]["digest"]
    assert manifest["record"]["frozen_ar_identity"] == {
        "identity_digest": "ar-persisted-identity"
    }
    with pytest.raises(controls.Veatic21ControlError, match="reuse identity mismatch"):
        controls.load_feature_control(
            paths.manifest,
            expected_frozen_ar_identity="different-ar-identity",
        )
    wrong_namespace = controls.ControlNamespace(
        target=namespace.target,
        fold=3,
        seed=namespace.seed,
        endpoint=namespace.endpoint,
        lane=namespace.lane,
    )
    with pytest.raises(controls.Veatic21ControlError, match="namespace mismatch"):
        controls.load_feature_control(paths.manifest, expected_namespace=wrong_namespace)


def test_existing_feature_artifact_is_never_overwritten_on_source_mismatch(tmp_path) -> None:
    train, test = _pair()
    namespace = _namespace("ar_plus_random_pca_residual")
    original = controls.build_matched_random_pca_control(
        train,
        test,
        namespace=namespace,
        frozen_ar_identity="ar-reuse",
    )
    paths = controls.persist_feature_control(original, tmp_path)
    before = paths.manifest.read_bytes()

    changed_test = features.build_veatic21_features(
        row_idx=test.row_idx,
        video_id=test.video_id,
        time_seconds=test.time_seconds,
        pca_scores=_pca(test) + 1.0,
        diagnostics=_diagnostics(test),
        pca_width=test.pca_width,
    )
    changed = controls.build_matched_random_pca_control(
        train,
        changed_test,
        namespace=namespace,
        frozen_ar_identity="ar-reuse",
    )
    with pytest.raises(controls.Veatic21ControlError, match="reuse record mismatch"):
        controls.persist_feature_control(changed, tmp_path)
    assert paths.manifest.read_bytes() == before


def test_target_persistence_roundtrip_and_identity_mismatch(tmp_path) -> None:
    block = _block({"a": 4, "b": 4})
    payload = {"model_family": "neural_ar7", "checkpoint_digest": "checkpoint"}
    unsigned_identity = {
        **payload,
    }
    identity = {
        **unsigned_identity,
        "identity_digest": __import__("hashlib").blake2b(
            json.dumps(
                unsigned_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest(),
    }
    namespace = _namespace("ar_plus_label_permutation_residual", endpoint="binary")
    control = controls.build_whole_video_label_permutation(
        row_idx=block.row_idx,
        video_id=block.video_id,
        time_seconds=block.time_seconds,
        target=np.linspace(0.0, 1.0, len(block.row_idx), dtype=np.float32),
        namespace=namespace,
        privileged=True,
        frozen_ar_identity=identity,
    )
    paths = controls.persist_target_control(control, tmp_path)
    loaded = controls.load_target_control(
        paths.manifest,
        expected_namespace=namespace,
        expected_frozen_ar_identity=identity,
    )
    assert loaded.record == control.record
    assert np.array_equal(loaded.target, control.target)
    with pytest.raises(controls.Veatic21ControlError, match="reuse identity mismatch"):
        controls.load_target_control(
            paths.manifest,
            expected_frozen_ar_identity="wrong-identity",
        )

