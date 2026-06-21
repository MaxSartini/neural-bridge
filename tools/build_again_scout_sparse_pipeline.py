#!/usr/bin/env python3
"""Build AGAIN scout/sparse-TRIBE planning artifacts.

This command is intentionally non-training and non-dense-encoding. It can build
preflight/control artifacts, label-safe telemetry-change features, candidate
window regions, and sparse ViT-G/TRIBE teacher queues.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.scripts.again_scout_sparse_pipeline import (  # noqa: E402
    build_run_manifest,
    build_sparse_teacher_queue,
    control_contract_rows,
    default_again_dataset_root,
    default_boundary_manifest_root,
    default_selector_configs,
    default_sparse_teacher_budgets,
    event_mask_contract_rows,
    external_root,
    group_clean_annotations,
    load_manifest_rows,
    merge_candidate_regions,
    model_registry,
    select_candidate_timestamps,
    split_contract_rows,
    telemetry_change_feature_rows,
    write_csv_rows,
    write_json,
    write_preflight_outputs,
)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["preflight", "cheap-telemetry", "candidate-windows", "teacher-queue", "all"],
        default="preflight",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Tracked output root. Defaults to outputs/again_scout_sparse_pipeline_<timestamp>/",
    )
    parser.add_argument("--again-root", type=Path, default=default_again_dataset_root())
    parser.add_argument("--manifest-root", type=Path, default=default_boundary_manifest_root())
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument(
        "--external-cache-root",
        type=Path,
        default=None,
        help="AGAIN-only external cache root for sparse teacher queues.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or Path("outputs") / f"again_scout_sparse_pipeline_{timestamp()}"
    external_cache_root = (
        args.external_cache_root
        or external_root() / "benchmarks" / "again" / f"scout_sparse_pipeline_{timestamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=False)

    manifest_path = args.manifest_root / "again_boundary_aligned_1hz_manifest.csv"
    clean_data_path = args.again_root / "annotations" / "clean_data.csv"
    metadata_path = args.again_root / "metadata" / "cleaned_session_video_metadata.csv"

    write_preflight_outputs(output_root, limit_videos=args.limit_videos)
    manifest = build_run_manifest(output_root, stage=args.stage, limit_videos=args.limit_videos)
    manifest["again_root"] = str(args.again_root)
    manifest["boundary_manifest_path"] = str(manifest_path)
    manifest["external_cache_root"] = str(external_cache_root)
    write_json(output_root / "run_manifest.json", manifest)

    if args.stage == "preflight":
        print(f"output_root={output_root}")
        print("stage=preflight")
        return 0

    manifest_rows = load_manifest_rows(manifest_path, limit_videos=args.limit_videos)
    annotations_by_video = group_clean_annotations(
        clean_data_path,
        metadata_path,
        limit_videos=args.limit_videos,
    )
    telemetry_rows = telemetry_change_feature_rows(
        manifest_rows=manifest_rows,
        annotations_by_video=annotations_by_video,
    )
    telemetry_path = output_root / "again_telemetry_change_features.csv"
    write_csv_rows(telemetry_path, telemetry_rows)

    if args.stage == "cheap-telemetry":
        print(f"output_root={output_root}")
        print(f"telemetry_rows={len(telemetry_rows)}")
        return 0

    selected = select_candidate_timestamps(telemetry_rows, default_selector_configs())
    regions = merge_candidate_regions(selected)
    write_csv_rows(output_root / "again_candidate_timestamps.csv", selected)
    write_csv_rows(output_root / "again_candidate_regions.csv", regions)

    if args.stage == "candidate-windows":
        print(f"output_root={output_root}")
        print(f"candidate_timestamps={len(selected)}")
        print(f"candidate_regions={len(regions)}")
        return 0

    registry = model_registry()
    teacher = next(spec for spec in registry if spec.name == "vjepa21_vitg_tribe_sparse_teacher")
    queue = build_sparse_teacher_queue(
        regions,
        default_sparse_teacher_budgets(),
        teacher_model=teacher,
        output_external_root=external_cache_root,
    )
    write_csv_rows(output_root / "again_sparse_tribe_teacher_queue.csv", queue)
    write_csv_rows(output_root / "again_control_contracts.csv", control_contract_rows())
    write_csv_rows(output_root / "again_split_contracts.csv", split_contract_rows())
    write_csv_rows(output_root / "again_event_mask_contracts.csv", event_mask_contract_rows())

    print(f"output_root={output_root}")
    print(f"telemetry_rows={len(telemetry_rows)}")
    print(f"candidate_timestamps={len(selected)}")
    print(f"candidate_regions={len(regions)}")
    print(f"sparse_teacher_queue_rows={len(queue)}")
    print("dense_again_vitg_run=false")
    print("models_trained=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
