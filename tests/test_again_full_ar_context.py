import csv
from pathlib import Path

from backend.scripts.again_full_ar_context import FullArContextConfig, run_full_ar_context


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for video_index in range(4):
        video_id = f"again_v{video_index}"
        for t in range(12):
            spike = (video_index + t) % 5 == 0 and t <= 8
            rows.append(
                {
                    "dataset_name": "AGAIN_cleaned",
                    "video_id": video_id,
                    "video_path": f"/tmp/{video_id}.webm",
                    "time_start_seconds": float(t),
                    "arousal": 0.2 + 0.05 * ((video_index + t) % 4),
                    "target_feasible_future_spike_1_3s": str(t <= 8).lower(),
                    "future_spike_1_3s_ge_0.05": str(spike).lower() if t <= 8 else "",
                    "future_spike_1_3s_ge_0.075": str(spike and t % 2 == 0).lower() if t <= 8 else "",
                    "game": "Game",
                    "genre": "Genre",
                    "alignment_policy": "use_annotation_covered_video_time_only",
                }
            )
    return rows


def test_full_ar_context_is_context_only_not_sparse_pca_comparison(tmp_path: Path):
    manifest_path = tmp_path / "again_manifest.csv"
    output_root = tmp_path / "again_full_ar_context"
    write_csv(manifest_path, manifest_rows())

    manifest = run_full_ar_context(
        output_root=output_root,
        config=FullArContextConfig(
            manifest_path=manifest_path,
            report_date="test",
            n_splits=2,
        ),
    )

    assert manifest["benchmark_mode"] == "again_full_995_ar_context_only"
    assert manifest["tribe_encoding_run"] is False
    assert manifest["sparse_pca128_features_used"] is False
    assert manifest["direct_sparse_pca128_comparison_made"] is False
    assert manifest["veatic_outputs_modified"] is False
    with (output_root / "again_full_ar_context_lane_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["model_lane"] for row in rows} == {"AR_only"}
    assert all(row["comparable_to_sparse_pca128"] == "false" for row in rows)
