"""Build a window-level benchmark manifest from local Emo-FilM annotations."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = "/Volumes/onn. Drive/Neural Bridge/datasets/emofilm_annotations"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tsv_gz(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return np.asarray(data, dtype=np.float32)


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite": 0}
    return {
        "count": int(array.size),
        "finite": int(finite.size),
        "missing": int(array.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def derivative_files(root: Path) -> list[Path]:
    return sorted((root / "derivatives").glob("Annot_*_stim.json"))


def film_id_from_derivative(path: Path) -> str:
    return path.stem.removeprefix("Annot_").removesuffix("_stim")


def build_schema_report(root: Path, rows: list[dict[str, Any]], output_manifest: Path) -> dict[str, Any]:
    participants = []
    participants_path = root / "participants.tsv"
    if participants_path.exists():
        lines = participants_path.read_text(encoding="utf-8").splitlines()
        participants = [line.split("\t")[0] for line in lines[1:] if line.strip()]

    derivative_reports = []
    target_names: list[str] = []
    global_values: list[np.ndarray] = []
    for metadata_path in derivative_files(root):
        film_id = film_id_from_derivative(metadata_path)
        metadata = load_json(metadata_path)
        data = load_tsv_gz(metadata_path.with_suffix(".tsv.gz"))
        target_names = metadata.get("Columns", target_names)
        global_values.append(data)
        derivative_reports.append(
            {
                "film_id": film_id,
                "rows": int(data.shape[0]),
                "targets": int(data.shape[1]),
                "sampling_frequency_hz": metadata.get("SamplingFrequency"),
                "start_time_seconds": metadata.get("StartTime"),
                "value_summary": summarize_array(data),
            }
        )

    per_subject_files = list(root.glob("sub-*/beh/*_stim.json"))
    recordings = sorted(
        {
            part.removeprefix("recording-")
            for path in per_subject_files
            for part in path.stem.split("_")
            if part.startswith("recording-")
        }
    )
    tasks = sorted(
        {
            part.removeprefix("task-")
            for path in per_subject_files
            for part in path.stem.split("_")
            if part.startswith("task-")
        }
    )

    all_values = np.concatenate([arr.reshape(-1) for arr in global_values]) if global_values else np.asarray([])
    return {
        "schema_version": "emofilm_annotation_schema_report_v1",
        "dataset_root": str(root),
        "manifest": str(output_manifest),
        "dataset_description": load_json(root / "dataset_description.json") if (root / "dataset_description.json").exists() else {},
        "targets": {
            "available_targets": target_names,
            "target_count": len(target_names),
            "types": "continuous derivative aggregate annotations; per-subject raw traces are continuous single-recording ratings",
            "primary_default_targets": [
                target for target in (
                    "Anxiety",
                    "Fear",
                    "Sad",
                    "Happiness",
                    "Surprise",
                    "Disgust",
                    "Calm",
                    "Good",
                    "Bad",
                    "IntenseEmotion",
                )
                if target in target_names
            ],
        },
        "rating_scale": {
            "derivative_aggregate": {
                "description": "Aggregated derivative annotations are continuous z-like values, not bounded 0-100 raw gauge values.",
                "summary": summarize_array(all_values),
            },
            "per_subject_raw": {
                "description": "Per-subject recording files are continuous gauge traces; observed local range includes 0-100 and occasional -1 sentinel values.",
                "observed_global_minmax_from_quick_scan": [-1.0, 100.0],
            },
        },
        "timestamp_window_structure": {
            "sampling_frequency_hz": 1.0,
            "window_seconds": 1.0,
            "start_time_seconds": 0,
            "row_to_time_mapping": "row_index / SamplingFrequency + StartTime",
            "label_level": "time-series / one row per 1-second film window",
        },
        "stimuli": {
            "film_count": len(derivative_reports),
            "film_ids": [entry["film_id"] for entry in derivative_reports],
            "per_film": derivative_reports,
            "total_windows": len(rows),
        },
        "participants": {
            "participant_count": len(participants),
            "participant_ids_present": bool(participants),
            "participant_ids": participants,
            "per_subject_annotation_files": len(per_subject_files),
            "per_subject_tasks": tasks,
            "per_subject_recordings": recordings,
        },
        "missing_values": {
            "manifest_rows_with_missing_targets": int(
                sum(any(not np.isfinite(float(value)) for value in row["targets"].values()) for row in rows)
            ),
            "derivative_files_missing_values": int(
                sum(entry["value_summary"]["missing"] > 0 for entry in derivative_reports)
            ),
        },
        "benchmark_design": {
            "primary_task": "film_id + 1-second time window -> aggregate human affect/emotion rating",
            "within_film_temporal_holdout": "Train/test on different time windows from the same film; labelled within-film only.",
            "leave_film_out_holdout": "Hold out entire films; labelled cross-film generalisation.",
            "openlav_mixing": False,
            "heavy_tribe_required_for_schema": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output", default="benchmarks/emofilm/emofilm_annotation_manifest.jsonl")
    parser.add_argument("--schema-report", default="benchmarks/emofilm/emofilm_annotation_schema_report.json")
    parser.add_argument("--limit-films", type=int)
    parser.add_argument("--limit-windows-per-film", type=int)
    parser.add_argument("--targets", default="")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_targets = [target.strip() for target in args.targets.split(",") if target.strip()]

    rows: list[dict[str, Any]] = []
    files = derivative_files(root)
    if args.limit_films:
        files = files[: args.limit_films]
    for metadata_path in files:
        metadata = load_json(metadata_path)
        target_names = metadata["Columns"]
        target_indices = list(range(len(target_names)))
        if selected_targets:
            missing = sorted(set(selected_targets) - set(target_names))
            if missing:
                raise ValueError(f"Unknown Emo-FilM targets {missing}; available={target_names}")
            target_indices = [target_names.index(target) for target in selected_targets]
        data = load_tsv_gz(metadata_path.with_suffix(".tsv.gz"))
        film_id = film_id_from_derivative(metadata_path)
        sampling_frequency = float(metadata.get("SamplingFrequency", 1.0))
        start_time = float(metadata.get("StartTime", 0.0))
        row_count = data.shape[0]
        if args.limit_windows_per_film:
            row_count = min(row_count, args.limit_windows_per_film)
        for window_index in range(row_count):
            time_start = start_time + window_index / sampling_frequency
            time_end = time_start + 1.0 / sampling_frequency
            targets = {
                target_names[index]: float(data[window_index, index])
                for index in target_indices
            }
            rows.append(
                {
                    "schema_version": "emofilm_annotation_window_v1",
                    "dataset": "emofilm",
                    "film_id": film_id,
                    "stimulus_id": f"{film_id}:{window_index:06d}",
                    "window_index": window_index,
                    "time_start_seconds": time_start,
                    "time_end_seconds": time_end,
                    "sampling_frequency_hz": sampling_frequency,
                    "targets": targets,
                    "source_annotation": str(metadata_path.with_suffix(".tsv.gz")),
                    "target_source": "derivative_aggregate_annotation",
                }
            )

    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    schema_report = build_schema_report(root, rows, output)
    schema_path = Path(args.schema_report).expanduser().resolve()
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema_report, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output), "schema_report": str(schema_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
