"""Join cached OpenLAV neuro-response IRs to official video-level labels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scientific_contract(contract: dict) -> dict:
    """Normalize legacy statuses by excluding exact runtime-only controls."""
    return {
        key: value
        for key, value in contract.items()
        if key not in {"mps_memory_fraction", "video_attention_query_chunk_size"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_tools/export/video_data.csv",
    )
    parser.add_argument(
        "--cache-dir",
        default="/Volumes/onn. Drive/Neural Bridge/benchmarks/openlav/tribe_cache",
    )
    parser.add_argument("--output", default="benchmarks/openlav/calibration_manifest.jsonl")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--expected-video-frames", type=int, default=64)
    parser.add_argument(
        "--expected-extraction-contract",
        default="official_64_frame_exact_chunked_attention",
    )
    parser.add_argument("--expected-feature-schema", default="neuro_calibration_features_v2")
    args = parser.parse_args()

    with Path(args.labels).expanduser().open(newline="", encoding="utf-8-sig") as handle:
        labels = list(csv.DictReader(handle))
    bounds = {
        axis: {
            "min": min(float(row[axis]) for row in labels),
            "max": max(float(row[axis]) for row in labels),
        }
        for axis in ("valence", "arousal")
    }
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    rows = []
    dropped = []
    contract_hashes = set()
    for label in sorted(labels, key=lambda row: row["video_code"]):
        code = label["video_code"]
        status_path = cache_dir / code / "cache_status.json"
        feature_path = cache_dir / code / "neuro_response_ir.npz"
        metadata_path = cache_dir / code / "neuro_response_ir.json"
        if not status_path.exists() or not feature_path.exists() or not metadata_path.exists():
            dropped.append({"stimulus_id": code, "reason": "missing_cache_artifact"})
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            ir_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            dropped.append({"stimulus_id": code, "reason": f"invalid_cache_metadata: {exc}"})
            continue
        if not status.get("complete"):
            dropped.append({
                "stimulus_id": code,
                "reason": "cache_incomplete",
                "error": status.get("error"),
            })
            continue
        contract = scientific_contract(status.get("model_contract", {}))
        if contract.get("video_num_frames") != args.expected_video_frames:
            dropped.append({
                "stimulus_id": code,
                "reason": "unexpected_video_frame_contract",
                "actual": contract.get("video_num_frames"),
                "expected": args.expected_video_frames,
            })
            continue
        if contract.get("video_extraction_contract") != args.expected_extraction_contract:
            dropped.append({
                "stimulus_id": code,
                "reason": "unexpected_extraction_contract",
                "actual": contract.get("video_extraction_contract"),
                "expected": args.expected_extraction_contract,
            })
            continue
        feature_contract = ir_metadata.get("feature_contract", {})
        if feature_contract.get("schema_version") != args.expected_feature_schema:
            dropped.append({
                "stimulus_id": code,
                "reason": "unexpected_feature_schema",
                "actual": feature_contract.get("schema_version"),
                "expected": args.expected_feature_schema,
            })
            continue
        feature_names = feature_contract.get("feature_names", [])
        feature_names_hash = stable_hash(feature_names)
        contract_hash = stable_hash(contract)
        contract_hashes.add(contract_hash)
        targets = {}
        raw_targets = {}
        for axis, limits in bounds.items():
            raw = float(label[axis])
            span = limits["max"] - limits["min"]
            targets[axis] = (raw - limits["min"]) / span if span else 0.5
            raw_targets[axis] = raw
        rows.append({
            "stimulus_id": code,
            "feature_path": str(feature_path),
            "group": label["source_URL"] or code,
            "targets": targets,
            "raw_targets": raw_targets,
            "target_contract": {
                "schema_version": "openlav_video_affect_targets_v1",
                "source_columns": {"valence": "valence", "arousal": "arousal"},
                "aggregation": "official OpenLAV video-level aggregate factor scores",
                "normalization": "dataset_minmax_v1",
            },
            "cache_contract": {
                "cache_key": status.get("cache_key"),
                "stimulus_sha256": status.get("stimulus_sha256"),
                "model_contract": contract,
                "model_contract_hash": contract_hash,
                "runtime_contract": status.get("runtime_contract", {
                    key: status.get("model_contract", {}).get(key)
                    for key in ("mps_memory_fraction", "video_attention_query_chunk_size")
                    if key in status.get("model_contract", {})
                }),
                "ir_schema_version": ir_metadata.get("schema_version"),
                "feature_schema_version": feature_contract.get("schema_version"),
                "feature_names_hash": feature_names_hash,
                "source_output_sha256": ir_metadata.get("source", {}).get("source_sha256"),
            },
            "metadata": {
                "author": label["author"],
                "source_url": label["source_URL"],
                "language_free": label["language_free"],
                "modal_emotion": label["modal_emotion"],
            },
        })

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "openlav_calibration_manifest_v2",
        "manifest": str(output),
        "rows": len(rows),
        "official_stimuli": len(labels),
        "dropped_rows": dropped,
        "dropped_count": len(dropped),
        "grouping": "source_URL",
        "contract_audit": {
            "expected_video_frames": args.expected_video_frames,
            "expected_extraction_contract": args.expected_extraction_contract,
            "expected_feature_schema": args.expected_feature_schema,
            "unique_model_contract_hashes": sorted(contract_hashes),
            "mixed_model_contracts": len(contract_hashes) > 1,
        },
        "target_normalization": {
            "method": "dataset_minmax_v1",
            "bounds": bounds,
            "warning": (
                "OpenLAV factor scores have no fixed theoretical [0,1] scale. "
                "These fixed dataset bounds are recorded for reproducibility."
            ),
        },
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if len(contract_hashes) > 1:
        raise SystemExit("Mixed model contracts detected in accepted manifest rows")
    if args.require_complete and dropped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
