import csv
from pathlib import Path

from backend.scripts.again_boundary_manifest import (
    BOUNDARY_POLICY,
    build_boundary_aligned_manifest,
    load_annotation_series,
    run_builder,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def boundary_row(video_id: str = "v1", annotation_end: float = 8.6) -> dict[str, object]:
    return {
        "dataset_name": "AGAIN_cleaned",
        "video_id": video_id,
        "video_path": f"/tmp/{video_id}.webm",
        "game": "Game",
        "participant_id": "p1",
        "session_id": "s1",
        "video_duration_seconds": 12.0,
        "annotation_start_seconds": 0.0,
        "annotation_end_seconds": annotation_end,
        "video_minus_annotation_end_seconds": 3.4,
        "recommended_encode_start_seconds": 0.0,
        "recommended_encode_end_seconds": annotation_end,
        "recommended_benchmark_start_seconds": 0.0,
        "recommended_benchmark_end_seconds": annotation_end,
        "target_safe_end_future_1_3s_seconds": annotation_end - 3.0,
        "trim_start_seconds": 0.0,
        "trim_end_seconds": 3.4,
        "one_hz_grid_start_second": 0,
        "one_hz_grid_end_second": int(annotation_end),
        "one_hz_target_safe_end_second": int(annotation_end - 3.0),
        "boundary_confidence": "medium",
        "recommended_policy": BOUNDARY_POLICY,
        "notes": "post_annotation_tail_dynamic_manual_review;video_extends_beyond_annotation",
    }


def annotation_rows(video_id: str = "v1") -> list[dict[str, object]]:
    return [
        {
            "dataset_name": "AGAIN_cleaned",
            "video_id": video_id,
            "video_path": f"/tmp/{video_id}.webm",
            "time_start_seconds": t,
            "frame_index": int(t * 4),
            "timestamp_index": str(t),
            "arousal": t / 10.0,
            "valence": "",
            "participant_id": "p1",
            "session_id": "s1",
            "game": "Game",
            "genre": "Genre",
            "aggregate_method": "none_participant_session_label",
            "split_group": "not_assigned",
            "source_metadata": "{}",
            "alignment_status": "duration_mismatch_gt_1s",
        }
        for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.6]
    ]


def test_builder_uses_annotation_boundary_not_fixed_video_minus_three():
    annotations = {
        "v1": load_annotation_series_from_rows(annotation_rows("v1"))["v1"],
    }

    rows, video_summary, summary = build_boundary_aligned_manifest([boundary_row("v1", 8.6)], annotations)

    assert len(video_summary) == 1
    assert summary["manifest_rows"] == 9
    assert rows[-1]["time_start_seconds"] == 8.0
    assert rows[-1]["recommended_benchmark_end_seconds"] == 8.6
    assert rows[-1]["target_feasible_future_spike_1_3s"] is False
    assert rows[5]["target_feasible_future_spike_1_3s"] is True
    assert rows[6]["target_feasible_future_spike_1_3s"] is False
    assert rows[0]["alignment_policy"] == BOUNDARY_POLICY
    assert rows[0]["boundary_notes"].startswith("post_annotation_tail_dynamic")


def test_run_builder_writes_guarded_manifest_outputs(tmp_path: Path):
    boundary_path = tmp_path / "boundaries.csv"
    proposal_path = tmp_path / "proposal.csv"
    output_root = tmp_path / "out"
    write_csv(boundary_path, [boundary_row("v1", 8.6)])
    write_csv(proposal_path, annotation_rows("v1"))

    manifest = run_builder(
        boundary_recommendations_path=boundary_path,
        manifest_proposal_path=proposal_path,
        output_root=output_root,
    )

    assert manifest["videos_in_manifest"] == 1
    assert manifest["manifest_rows"] == 9
    assert manifest["tribe_encoding_run"] is False
    assert manifest["models_trained"] is False
    assert manifest["veatic_outputs_modified"] is False
    assert manifest["final_benchmark_manifest_created"] is False
    assert (output_root / "again_boundary_aligned_1hz_manifest.csv").exists()
    with (output_root / "again_boundary_aligned_1hz_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["alignment_policy"] == BOUNDARY_POLICY
    assert rows[0]["target_feasible_future_spike_1_3s"] == "true"
    assert rows[-1]["target_feasible_future_spike_1_3s"] == "false"


def load_annotation_series_from_rows(rows: list[dict[str, object]]):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "proposal.csv"
        write_csv(path, rows)
        return load_annotation_series(path)
