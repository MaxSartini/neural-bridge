from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from neural_bridge.veatic21.benchmark import benchmark_bundle_topologies
from neural_bridge.veatic21.bundle import (
    ALIGNMENT_FILES,
    FORBIDDEN_NAME,
    TRIBE_FILES,
    BundleError,
    assemble_bundle,
    assert_safe_delete_target,
    verify_bundle,
)
from neural_bridge.veatic21.phase00 import run_phase00, verify_phase00


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n")


def _make_sources(root: Path, ids: tuple[str, ...]) -> tuple[Path, Path]:
    tribe_root = root / "tribe"
    alignment_root = root / "alignment"
    for video_id in ids:
        tribe = tribe_root / video_id
        alignment = alignment_root / video_id
        tribe.mkdir(parents=True)
        alignment.mkdir(parents=True)
        row_count = 3
        rows = []
        for row_index in range(row_count):
            time_seconds = row_index / 2
            rows.append(
                {
                    "video_id": video_id,
                    "row_index": row_index,
                    "time_seconds": time_seconds,
                    "row_hz": 2.0,
                    "source_frame_position": row_index * 4.0,
                    "source_floor_frame_index": row_index * 4,
                    "source_ceil_frame_index": row_index * 4,
                    "source_interp_alpha": 0.0,
                    "arousal": row_index / 10,
                    "valence": -row_index / 10,
                }
            )
        with (alignment / "rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        values = {
            key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
            for key in (
                "time_seconds",
                "source_frame_position",
                "source_floor_frame_index",
                "source_ceil_frame_index",
                "source_interp_alpha",
                "arousal",
                "valence",
            )
        }
        np.savez_compressed(
            tribe / "tribe_v2_cortical_predictions.npz",
            time_seconds=values["time_seconds"],
            source_frame_position=values["source_frame_position"],
            source_floor_frame_index=values["source_floor_frame_index"],
            source_ceil_frame_index=values["source_ceil_frame_index"],
            source_interp_alpha=values["source_interp_alpha"],
            arousal=values["arousal"],
            valence=values["valence"],
            cortical_prediction=np.full((row_count, 20_484), 0.25, dtype=np.float16),
            temporal_diagnostics53=np.full((row_count, 53), 0.5, dtype=np.float32),
        )
        status = {"status": "complete", "video_id": video_id, "row_count": row_count}
        _write_json(tribe / "status.json", status)
        _write_json(tribe / "manifest.json", status)
        _write_json(alignment / "status.json", status)
        _write_json(alignment / "manifest.json", status)
        _write_json(alignment / "preprocessing.json", {"row_hz": 2.0})
        (alignment / FORBIDDEN_NAME).write_bytes(b"must never enter the bundle")
    return tribe_root, alignment_root


def test_assembly_is_parallel_verified_atomic_and_excludes_hidden_states(
    tmp_path: Path,
) -> None:
    expected = ("0", "1")
    tribe_root, alignment_root = _make_sources(tmp_path, expected)
    output_root = tmp_path / "bundle"
    hidden_hash_before = (alignment_root / "0" / FORBIDDEN_NAME).read_bytes()

    result = assemble_bundle(
        tribe_root=tribe_root,
        vjepa_root=alignment_root,
        output_root=output_root,
        expected_ids=expected,
        workers=2,
    )

    assert result["status"] == "pass"
    assert result["video_count"] == 2
    assert result["total_rows"] == 6
    assert result["atomic_publication"] is True
    assert result["sealed_read_only"] is True
    assert not list(output_root.rglob(FORBIDDEN_NAME))
    assert (alignment_root / "0" / FORBIDDEN_NAME).read_bytes() == hidden_hash_before
    for video_id in expected:
        names = {path.name for path in (output_root / "per_video" / video_id).iterdir()}
        assert names == {
            *TRIBE_FILES.values(),
            *ALIGNMENT_FILES.values(),
            "input-manifest.json",
        }

    verified = verify_bundle(
        output_root=output_root,
        expected_ids=expected,
        workers=2,
    )
    assert verified["protected_source_hashes_unchanged"] is True

    registration = tmp_path / "registration.json"
    _write_json(registration, {"phase": "00"})
    again_root = tmp_path / "again"
    again_root.mkdir()
    phase00_root = tmp_path / "phase00"
    phase00 = run_phase00(
        bundle_root=output_root,
        output_root=phase00_root,
        expected_ids=expected,
        workers=2,
        registration_path=registration,
        protected_roots=(tribe_root, alignment_root, output_root, again_root),
    )
    assert phase00["status"] == "pass"
    assert phase00["video_count"] == 2
    assert verify_phase00(output_root=phase00_root) == phase00


def test_assembly_refuses_overwrite(tmp_path: Path) -> None:
    expected = ("0",)
    tribe_root, alignment_root = _make_sources(tmp_path, expected)
    output_root = tmp_path / "bundle"
    output_root.mkdir()
    with pytest.raises(BundleError, match="refusing to overwrite"):
        assemble_bundle(
            tribe_root=tribe_root,
            vjepa_root=alignment_root,
            output_root=output_root,
            expected_ids=expected,
        )


def test_topology_benchmark_uses_every_requested_video_and_cleans_scratch(
    tmp_path: Path,
) -> None:
    expected = ("0",)
    tribe_root, alignment_root = _make_sources(tmp_path, expected)
    scratch = tmp_path / "scratch"
    result = benchmark_bundle_topologies(
        worker_counts=(1,),
        repeats=1,
        tribe_root=tribe_root,
        vjepa_root=alignment_root,
        scratch_parent=scratch,
        expected_ids=expected,
    )
    assert result["runs"][0]["video_count"] == 1
    assert result["runs"][0]["verified_video_count"] == 1
    assert not list(scratch.iterdir())


@pytest.mark.parametrize(
    "target",
    [
        Path("/Volumes/onn. Drive/Neural Bridge Artifacts/features"),
        Path("/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2"),
        Path("/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/child"),
        Path("/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again"),
    ],
)
def test_deletion_guard_rejects_ancestors_roots_and_descendants(target: Path) -> None:
    with pytest.raises(BundleError, match="protected root"):
        assert_safe_delete_target(target)
