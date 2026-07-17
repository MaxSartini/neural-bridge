from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.scripts.veatic21_compact_cache import (
    ENCODE_POLICY,
    EXPECTED_VIDEO_IDS,
    MODEL_SHA256,
    PAYLOAD_SCHEMA,
    POSTPASS_SCHEMA,
    PREDICTION_WIDTH,
    QUALITY_FLAG_KEYS,
    ROW_SCALAR_BOUNDS,
    ROW_SCALAR_DTYPES,
    ROW_PLAN_SHA256,
    RUN_SCHEMA,
    SELECTED_STATE_INDICES,
    UPLOAD_SCHEMA,
    UPSTREAM_SCHEMA,
    Veatic21CompactCache,
    Veatic21CompactCacheError,
)


EXPECTED_ROW_SCALARS = (
    "time_seconds",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "source_arousal",
    "source_valence",
    "arousal",
    "valence",
    "luma_mean",
    "luma_std",
    "frame_luma_std_mean",
    "motion_absdiff_mean",
    "black_frame_fraction",
    "duplicate_frame_fraction",
    "quality_black_frame_flag",
    "quality_duplicate_frame_flag",
    "quality_exclusion_flag",
    "quality_weight_suggested",
)


def _json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(rows: int, offset: float) -> dict[str, np.ndarray]:
    time_seconds = (np.arange(rows, dtype=np.float32) / np.float32(2.0)).astype(np.float32)
    trailing = np.arange(63, -1, -1, dtype=np.float32) / np.float32(16.0)
    sample_times = np.maximum(0.0, time_seconds[:, None] - trailing[None, :]).astype(np.float32)
    position = (time_seconds * np.float32(25.0)).astype(np.float32)
    floor = np.floor(position).astype(np.int32)
    ceil = np.ceil(position).astype(np.int32)
    alpha = (position - floor.astype(np.float32)).astype(np.float32)
    arousal = np.linspace(-0.5 + offset, 0.5 + offset, rows, dtype=np.float32)
    valence = np.linspace(0.5 + offset, -0.5 + offset, rows, dtype=np.float32)
    zeros = np.zeros(rows, dtype=np.float32)
    flags = np.zeros(rows, dtype=np.uint8)
    return {
        "time_seconds": time_seconds,
        "sample_frame_indices": np.rint(sample_times * np.float32(16.0)).astype(np.int32),
        "sample_time_seconds": sample_times,
        "selected_state_indices": np.asarray(SELECTED_STATE_INDICES, dtype=np.int16),
        "source_frame_position": position,
        "source_floor_frame_index": floor,
        "source_ceil_frame_index": ceil,
        "source_interp_alpha": alpha,
        "source_arousal": arousal,
        "source_valence": valence,
        "arousal": arousal.copy(),
        "valence": valence.copy(),
        "luma_mean": zeros.copy(),
        "luma_std": zeros.copy(),
        "frame_luma_std_mean": zeros.copy(),
        "motion_absdiff_mean": zeros.copy(),
        "black_frame_fraction": zeros.copy(),
        "duplicate_frame_fraction": zeros.copy(),
        "quality_black_frame_flag": flags.copy(),
        "quality_duplicate_frame_flag": flags.copy(),
        "quality_exclusion_flag": flags.copy(),
        "quality_weight_suggested": np.ones(rows, dtype=np.float32),
        "cortical_prediction": np.zeros((rows, PREDICTION_WIDTH), dtype=np.float16),
        "tribe_grouped_video_feature": np.zeros((rows, 2, 1408), dtype=np.float16),
        "temporal_diagnostics53": np.zeros((rows, 53), dtype=np.float32),
    }


def _write_upstream(root: Path, video_id: str, arrays: dict[str, np.ndarray]) -> Path:
    video_root = root / video_id
    video_root.mkdir(parents=True)
    npz_path = video_root / "vjepa21_hidden_states.npz"
    upstream_arrays = {
        key: value
        for key, value in arrays.items()
        if key not in {"cortical_prediction", "tribe_grouped_video_feature"}
    }
    upstream_arrays["features"] = np.zeros(
        (len(arrays["time_seconds"]), 20, 1, 1408), dtype=np.float16
    )
    np.savez_compressed(npz_path, **upstream_arrays)
    rows = len(arrays["time_seconds"])
    manifest = {
        "cache_keys": sorted(upstream_arrays),
        "cache_sha256": _sha(npz_path),
        "causal_sample_span_seconds": 3.9375,
        "decode_hz": 16.0,
        "dtype": "float16",
        "encode_policy": ENCODE_POLICY,
        "feature_shape": [rows, 20, 1, 1408],
        "frames_per_clip": 64,
        "image_size": 256,
        "model_sha256": MODEL_SHA256,
        "persisted_full_temporal_tensors": False,
        "row_count": rows,
        "row_hz": 2.0,
        "row_plan_sha256": ROW_PLAN_SHA256,
        "schema": UPSTREAM_SCHEMA,
        "selected_state_indices": list(SELECTED_STATE_INDICES),
        "status": "complete",
        "temporal_diagnostics53_shape": [rows, 53],
        "time_end_seconds": (rows - 1) / 2.0,
        "time_start_seconds": 0.0,
        "tribe_imported_or_run": False,
        "video_id": video_id,
        "video_name": f"{video_id}.mp4",
        "video_sha256": "a" * 64,
        "vjepa_only": True,
    }
    _json(video_root / "manifest.json", manifest)
    _json(video_root / "status.json", manifest)
    _json(video_root / "preprocessing.json", {"video_id": video_id})
    (video_root / "rows.csv").write_text("row_index\n0\n", encoding="utf-8")

    payload_files = []
    for name in (
        "manifest.json",
        "preprocessing.json",
        "rows.csv",
        "status.json",
        "vjepa21_hidden_states.npz",
    ):
        path = video_root / name
        payload_files.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    payload = {
        "schema_version": PAYLOAD_SCHEMA,
        "video_id": video_id,
        "file_count": len(payload_files),
        "total_bytes": sum(item["bytes"] for item in payload_files),
        "files": payload_files,
    }
    payload_path = video_root / "_PAYLOAD_SHA256.json"
    _json(payload_path, payload)
    payload_bytes = payload_path.read_bytes()
    marker = {
        "schema_version": UPLOAD_SCHEMA,
        "status": "complete",
        "video_id": video_id,
        "payload_manifest": "_PAYLOAD_SHA256.json",
        "payload_manifest_bytes": len(payload_bytes),
        "payload_manifest_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "payload_file_count": len(payload_files),
        "payload_total_bytes": payload["total_bytes"],
    }
    _json(video_root / "_UPLOAD_COMPLETE.json", marker)
    return npz_path


def _make_cache(tmp_path: Path, row_counts: tuple[int, ...] = (2, 3)) -> tuple[Veatic21CompactCache, Path, Path, Path]:
    cache_root = tmp_path / "compact"
    per_video = cache_root / "per_video"
    upstream_root = tmp_path / "upstream"
    identity_path = tmp_path / "identity.jsonl"
    video_ids = tuple(str(index) for index in range(len(row_counts)))
    identity_rows = []
    for index, (video_id, rows) in enumerate(zip(video_ids, row_counts)):
        arrays = _arrays(rows, offset=float(index) / 10.0)
        upstream_npz = _write_upstream(upstream_root, video_id, arrays)
        video_root = per_video / video_id
        video_root.mkdir(parents=True)
        output_npz = video_root / "tribe_v2_cortical_predictions.npz"
        np.savez_compressed(output_npz, **arrays)
        postpass = {
            "schema_version": POSTPASS_SCHEMA,
            "status": "complete",
            "video_id": video_id,
            "row_count": rows,
            "cortical_shape": [rows, PREDICTION_WIDTH],
            "grouped_shape": [rows, 2, 1408],
            "input_cache": str(upstream_npz.resolve()),
            "output_cache": str(output_npz.resolve()),
            "runtime_seconds": 0.1,
            "finished_at": "2026-07-16T00:00:00+00:00",
        }
        _json(video_root / "manifest.json", postpass)
        _json(video_root / "status.json", postpass)
        # Deliberately incompatible old-row data: only identity may be consumed.
        identity_rows.append(
            {
                "dataset": "veatic",
                "video_id": video_id,
                "media_path": f"<external-assets-root>/videos/{video_id}.mp4",
                "source_annotation": {
                    "arousal": f"<external-assets-root>/ratings/{video_id}_arousal.csv",
                    "valence": f"<external-assets-root>/ratings/{video_id}_valence.csv",
                },
                "time_start_seconds": 999.0,
                "sampling_frequency_hz": 1.0,
                "targets": {"arousal": 999.0, "valence": 999.0},
            }
        )
    per_video.mkdir(parents=True, exist_ok=True)
    _json(
        per_video / "run_status.json",
        {
            "schema_version": RUN_SCHEMA,
            "expected_videos": len(video_ids),
            "completed_videos": len(video_ids),
            "failures": {},
            "updated_at": "2026-07-16T00:00:00+00:00",
        },
    )
    identity_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in identity_rows),
        encoding="utf-8",
    )
    adapter = Veatic21CompactCache(
        cache_root,
        upstream_root=upstream_root,
        identity_manifest_path=identity_path,
        expected_video_ids=video_ids,
        expected_total_rows=sum(row_counts),
    )
    return adapter, cache_root, upstream_root, identity_path


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


def _snapshot(paths: tuple[Path, ...]) -> dict[str, tuple[int, str]]:
    files = sorted(path for root in paths for path in root.rglob("*") if path.is_file())
    return {str(path): (path.stat().st_mtime_ns, _sha(path)) for path in files}


def test_validates_and_exposes_new_dense_contract_without_prediction_copy(tmp_path: Path) -> None:
    adapter, _cache, _upstream, _identity = _make_cache(tmp_path)
    report = adapter.validate()
    block = adapter.load_video("0")

    assert report.status == "pass"
    assert report.video_ids == ("0", "1")
    assert report.total_rows == 5
    assert report.prediction_width == PREDICTION_WIDTH
    assert block.row_count == 2
    assert block.columns["predictions"] is block.columns["cortical_prediction"]
    assert block.predictions is block.cortical_prediction
    assert block.predictions.dtype == np.float16
    assert not block.predictions.flags.writeable
    assert not block.columns["row_index"].flags.writeable
    assert np.array_equal(block.columns["row_index"], np.arange(2, dtype=np.int32))
    assert np.array_equal(block.time_seconds, np.asarray([0.0, 0.5], dtype=np.float32))
    assert not np.any(block.arousal == 999.0)
    assert block.identity.media_path.endswith("/0.mp4")
    assert block.provenance.row_plan_sha256 == ROW_PLAN_SHA256
    assert block.provenance.model_sha256 == MODEL_SHA256


def test_adapter_does_not_modify_any_cache_or_identity_file(tmp_path: Path) -> None:
    adapter, cache_root, upstream_root, identity_path = _make_cache(tmp_path)
    before = _snapshot((cache_root, upstream_root, identity_path.parent))
    adapter.validate()
    adapter.load_video("1")
    after = _snapshot((cache_root, upstream_root, identity_path.parent))
    assert after == before


def test_default_contract_is_exactly_124_video_ids() -> None:
    assert EXPECTED_VIDEO_IDS == tuple(str(index) for index in range(124))


def test_row_scalar_contract_is_explicit_and_complete() -> None:
    assert tuple(ROW_SCALAR_DTYPES) == EXPECTED_ROW_SCALARS
    assert tuple(ROW_SCALAR_BOUNDS) == EXPECTED_ROW_SCALARS
    assert QUALITY_FLAG_KEYS == (
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
    )


@pytest.mark.parametrize("key", EXPECTED_ROW_SCALARS)
def test_fails_closed_on_non_1d_row_scalar_shape(tmp_path: Path, key: str) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    path = cache_root / "per_video/0/tribe_v2_cortical_predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        value = bundle[key]
    _rewrite_npz(path, **{key: value.reshape(value.shape[0], 1)})
    with pytest.raises(Veatic21CompactCacheError, match=rf"scalar shape mismatch.*{key}"):
        adapter.load_video("0")


@pytest.mark.parametrize("key", EXPECTED_ROW_SCALARS)
def test_fails_closed_on_row_scalar_dtype_drift(tmp_path: Path, key: str) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    path = cache_root / "per_video/0/tribe_v2_cortical_predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        value = bundle[key]
    wrong_dtype = np.float64 if np.issubdtype(value.dtype, np.floating) else np.int64
    _rewrite_npz(path, **{key: value.astype(wrong_dtype)})
    with pytest.raises(
        Veatic21CompactCacheError,
        match=rf"({key} dtype mismatch|scalar dtype mismatch.*{key})",
    ):
        adapter.load_video("0")


@pytest.mark.parametrize("key", EXPECTED_ROW_SCALARS)
def test_fails_closed_on_row_scalar_domain_violation(tmp_path: Path, key: str) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    path = cache_root / "per_video/0/tribe_v2_cortical_predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        value = bundle[key].copy()
    lower, upper = ROW_SCALAR_BOUNDS[key]
    if key in QUALITY_FLAG_KEYS:
        value[0] = np.uint8(2)
        pattern = rf"boolean-domain mismatch.*{key}"
    elif upper is not None:
        value[0] = np.asarray(upper + 1, dtype=value.dtype)
        pattern = rf"scalar domain mismatch.*{key}"
    else:
        assert lower is not None
        value[0] = np.asarray(lower - 1, dtype=value.dtype)
        pattern = rf"scalar domain mismatch.*{key}"
    _rewrite_npz(path, **{key: value})
    with pytest.raises(Veatic21CompactCacheError, match=pattern):
        adapter.load_video("0")


@pytest.mark.parametrize(
    ("key", "mutation", "pattern"),
    (
        (
            "sample_frame_indices",
            lambda value: value.astype(np.int64),
            "sample frame dtype mismatch",
        ),
        (
            "sample_time_seconds",
            lambda value: value.astype(np.float64),
            "sample time dtype mismatch",
        ),
        (
            "selected_state_indices",
            lambda value: value.astype(np.int32),
            "selected-state dtype mismatch",
        ),
    ),
)
def test_fails_closed_on_support_array_dtype_drift(
    tmp_path: Path,
    key: str,
    mutation: Any,
    pattern: str,
) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    path = cache_root / "per_video/0/tribe_v2_cortical_predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        value = bundle[key]
    _rewrite_npz(path, **{key: mutation(value)})
    with pytest.raises(Veatic21CompactCacheError, match=pattern):
        adapter.load_video("0")


def test_fails_closed_when_run_is_incomplete_or_id_is_missing(tmp_path: Path) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    run_path = cache_root / "per_video/run_status.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["completed_videos"] = 1
    _json(run_path, run)
    with pytest.raises(Veatic21CompactCacheError, match="not complete"):
        adapter.validate()

    run["completed_videos"] = 2
    _json(run_path, run)
    (cache_root / "per_video/1").rename(cache_root / "per_video/missing-1")
    with pytest.raises(Veatic21CompactCacheError, match="coverage mismatch"):
        adapter.load_video("0")


@pytest.mark.parametrize("fault", ["timestamp", "width", "nonfinite"])
def test_fails_closed_on_invalid_dense_arrays(tmp_path: Path, fault: str) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    path = cache_root / "per_video/0/tribe_v2_cortical_predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        times = bundle["time_seconds"]
        predictions = bundle["cortical_prediction"]
    if fault == "timestamp":
        bad = times.copy()
        bad[1] = np.float32(0.75)
        _rewrite_npz(path, time_seconds=bad)
        pattern = "exact 2 Hz"
    elif fault == "width":
        _rewrite_npz(path, cortical_prediction=predictions[:, :-1])
        pattern = "cortical width"
    else:
        bad = predictions.copy()
        bad[0, 0] = np.float16(np.nan)
        _rewrite_npz(path, cortical_prediction=bad)
        pattern = "non-finite"
    with pytest.raises(Veatic21CompactCacheError, match=pattern):
        adapter.load_video("0")


def test_fails_closed_on_manifest_row_count_and_upstream_provenance(tmp_path: Path) -> None:
    adapter, cache_root, upstream_root, _identity = _make_cache(tmp_path)
    manifest_path = cache_root / "per_video/0/manifest.json"
    status_path = cache_root / "per_video/0/status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 999
    _json(manifest_path, manifest)
    _json(status_path, manifest)
    with pytest.raises(Veatic21CompactCacheError, match="row total|row_count"):
        adapter.load_video("0")

    manifest["row_count"] = 2
    _json(manifest_path, manifest)
    _json(status_path, manifest)
    upstream_manifest_path = upstream_root / "0/manifest.json"
    upstream_status_path = upstream_root / "0/status.json"
    upstream_manifest = json.loads(upstream_manifest_path.read_text(encoding="utf-8"))
    upstream_manifest["row_plan_sha256"] = "0" * 64
    _json(upstream_manifest_path, upstream_manifest)
    _json(upstream_status_path, upstream_manifest)
    with pytest.raises(Veatic21CompactCacheError, match="row-plan provenance"):
        adapter.load_video("0")


def test_fails_closed_if_postpass_changes_authoritative_cache_labels(tmp_path: Path) -> None:
    adapter, cache_root, _upstream, _identity = _make_cache(tmp_path)
    path = cache_root / "per_video/0/tribe_v2_cortical_predictions.npz"
    with np.load(path, allow_pickle=False) as bundle:
        arousal = bundle["arousal"]
    changed = arousal.copy()
    changed[0] += np.float32(0.25)
    _rewrite_npz(path, arousal=changed, source_arousal=changed.copy())
    with pytest.raises(Veatic21CompactCacheError, match="changed authoritative upstream array"):
        adapter.load_video("0")
