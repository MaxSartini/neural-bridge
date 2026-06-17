"""Freeze and verify the VEATIC-124 v2 evidence bundle without re-encoding.

The snapshot intentionally copies lightweight tracked evidence and cache
metadata only. Raw TRIBE arrays stay external in the live cache and are checked
by presence/count/size unless --include-raw-cache-checksums is requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ID = "veatic_124_v2_20260616"
TRACKED_MANIFEST = ROOT / "benchmarks" / "veatic" / "veatic_v2_evidence_manifest.json"
CACHE_RELATIVE = Path("benchmarks") / "veatic" / "tribe_cache"
SNAPSHOT_RELATIVE = Path("evidence_snapshots") / SNAPSHOT_ID
MODALITY_ORDER = ("text", "audio", "video")

EXPLICIT_TRACKED_FILES = {
    "AGENTS.md",
    "README.md",
    "REQUIREMENTS.md",
    "ROADMAP.md",
    "docs/README.md",
    "docs/PROJECT_MEMORY.md",
    "docs/current_project_state.md",
    "docs/external_assets_manifest.md",
    "docs/veatic_v2_evidence_summary.md",
    "reports/current_artifact_port_audit_20260617.md",
    "backend/scripts/run_veatic_strict_benchmark.py",
}

TRACKED_PREFIXES = (
    "benchmarks/veatic/",
    "outputs/veatic_124_temporal_context_v2_20260616_1557/",
    "outputs/veatic_124_temporal_fairness_20260616_1509/",
)

TRACKED_BENCHMARK_PREFIXES = (
    "benchmarks/veatic/veatic_124",
    "benchmarks/veatic/veatic_manifest_124",
    "benchmarks/veatic/veatic_neuro_benchmark_124video",
)

CACHE_METADATA_NAMES = ("cache_status.json", "tribe_summary.json", "tribe_segments.json")

SUPERSEDED_ARTIFACTS = [
    {
        "scope": "repo docs",
        "status": "deleted",
        "pattern": "pre-v2 VEATIC 5/20/50-video handoffs and acceleration audits",
        "reason": "They contradicted the VEATIC-124 v2 baseline and are available only through git history.",
    },
    {
        "scope": "repo scripts",
        "status": "deleted",
        "pattern": "old smoke-test and transition scripts not needed for the v2 strict suite",
        "reason": "The consolidated strict benchmark suite is the current entrypoint.",
    },
    {
        "scope": "external",
        "status": "retained_non_authoritative",
        "pattern": "benchmarks/veatic/tribe_cache_mlx",
        "reason": "Retained only as MLX hotswap/parity archaeology; not part of the frozen v2 evidence bundle.",
    },
    {
        "scope": "external",
        "status": "retained_non_authoritative",
        "pattern": "benchmarks/veatic/tribe_cache_multimodal_pilot",
        "reason": "Retained as the guarded 83/84 multimodal pilot; not part of the proven v2 baseline.",
    },
    {
        "scope": "external",
        "status": "retained_non_authoritative",
        "pattern": "benchmarks/veatic/tribe_smoke",
        "reason": "Retained only as local smoke-test residue if present; not authoritative evidence.",
    },
]


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def local_env_value(name: str) -> str | None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")
    return None


def external_root() -> Path:
    value = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT") or local_env_value("NEURAL_BRIDGE_EXTERNAL_ROOT")
    if not value:
        raise SystemExit("NEURAL_BRIDGE_EXTERNAL_ROOT is not set and not present in .env")
    return Path(value).expanduser()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_evidence_files() -> list[Path]:
    paths: list[Path] = []
    for rel in git_lines("ls-files"):
        if rel == TRACKED_MANIFEST.relative_to(ROOT).as_posix():
            continue
        include = rel in EXPLICIT_TRACKED_FILES
        include = include or rel.startswith(TRACKED_PREFIXES)
        include = include or rel.startswith(TRACKED_BENCHMARK_PREFIXES)
        if include:
            path = ROOT / rel
            if path.is_file():
                paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def tracked_records() -> list[dict[str, Any]]:
    records = []
    for path in tracked_evidence_files():
        rel = path.relative_to(ROOT).as_posix()
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role_for_tracked_file(rel),
            }
        )
    return records


def role_for_tracked_file(rel: str) -> str:
    if rel.startswith("benchmarks/veatic/veatic_manifest_124"):
        return "veatic_124_manifest"
    if rel.startswith("benchmarks/veatic/veatic_124"):
        return "veatic_124_benchmark_output"
    if rel.startswith("benchmarks/veatic/veatic_neuro_benchmark_124video"):
        return "veatic_124_neuro_benchmark_output"
    if rel.startswith("outputs/veatic_124_temporal_context_v2"):
        return "temporal_context_v2_output"
    if rel.startswith("outputs/veatic_124_temporal_fairness"):
        return "temporal_fairness_reference_output"
    if rel.startswith("docs/") or rel in {"README.md", "ROADMAP.md", "REQUIREMENTS.md", "AGENTS.md"}:
        return "current_orientation_report"
    if rel.startswith("reports/"):
        return "repo_audit_report"
    return "reproducibility_code"


def cache_dir(root: Path) -> Path:
    return root / CACHE_RELATIVE


def natural_cache_dirs(path: Path) -> list[Path]:
    return sorted(
        [child for child in path.iterdir() if child.is_dir() and child.name.isdigit()],
        key=lambda item: int(item.name),
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def cache_records(root: Path, include_raw_checksums: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = cache_dir(root)
    if not path.is_dir():
        raise SystemExit(f"cache directory not found: {path}")

    records: list[dict[str, Any]] = []
    modality_counts: dict[str, int] = {}
    completed_status_count = 0
    raw_count = 0
    for item in natural_cache_dirs(path):
        video_id = item.name
        raw_path = item / "tribe_raw_output.npz"
        status_path = item / "cache_status.json"
        summary_path = item / "tribe_summary.json"
        status = read_json(status_path)
        summary = read_json(summary_path)
        if status.get("complete") is True:
            completed_status_count += 1
        if raw_path.exists():
            raw_count += 1

        event_quality = summary.get("event_quality") or {}
        missing = (
            bool(event_quality.get("missing_text", True)),
            bool(event_quality.get("missing_audio", True)),
            bool(event_quality.get("missing_video", True)),
        )
        present = "+".join(name for name, is_missing in zip(MODALITY_ORDER, missing) if not is_missing) or "none"
        modality_counts[present] = modality_counts.get(present, 0) + 1

        raw_record: dict[str, Any] = {
            "video_id": video_id,
            "raw_output_present": raw_path.exists(),
            "raw_output_bytes": raw_path.stat().st_size if raw_path.exists() else None,
            "raw_output_sha256": sha256_file(raw_path) if include_raw_checksums and raw_path.exists() else None,
            "cache_status_present": status_path.exists(),
            "cache_status_complete": status.get("complete"),
            "tribe_summary_present": summary_path.exists(),
            "present_modalities": present,
            "metadata_files": [],
        }
        for name in CACHE_METADATA_NAMES:
            metadata = item / name
            if metadata.exists():
                raw_record["metadata_files"].append(
                    {
                        "path": metadata.relative_to(path).as_posix(),
                        "bytes": metadata.stat().st_size,
                        "sha256": sha256_file(metadata),
                    }
                )
        records.append(raw_record)

    summary = {
        "cache_relative_path": CACHE_RELATIVE.as_posix(),
        "cache_video_dirs": len(records),
        "raw_output_count": raw_count,
        "complete_status_count": completed_status_count,
        "modality_counts": dict(sorted(modality_counts.items())),
        "raw_cache_checksums_included": include_raw_checksums,
    }
    return records, summary


def build_manifest(root: Path, include_raw_checksums: bool = False) -> dict[str, Any]:
    cache_inventory, cache_summary = cache_records(root, include_raw_checksums=include_raw_checksums)
    tracked = tracked_records()
    return {
        "schema_version": "neural_bridge_veatic_v2_evidence_bundle_v1",
        "snapshot_id": SNAPSHOT_ID,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "claim_scope": (
            "VEATIC-124 v2 video-dominant cortical/TRIBE arousal event and spike ranking; "
            "not a full text+audio+video multimodal claim."
        ),
        "external_snapshot_relative_path": SNAPSHOT_RELATIVE.as_posix(),
        "authoritative_cache_relative_path": CACHE_RELATIVE.as_posix(),
        "authoritative_tracked_files": tracked,
        "cache_summary": cache_summary,
        "cache_inventory": cache_inventory,
        "superseded_artifacts": SUPERSEDED_ARTIFACTS,
        "verification": {
            "command": "npm run evidence:verify",
            "does_not_reencode_video": True,
            "raw_cache_default": "presence-size-metadata-checks",
            "raw_cache_sha256_command": "python3 backend/scripts/freeze_veatic_v2_evidence.py --verify --include-raw-cache-checksums",
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row if key != "metadata_files"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def remove_write_bits(path: Path) -> None:
    mode = path.stat().st_mode
    if path.is_dir():
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    else:
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def protect_tree(path: Path) -> None:
    for item in sorted(path.rglob("*"), reverse=True):
        remove_write_bits(item)
    remove_write_bits(path)


def copy_tracked_files(snapshot: Path, manifest: dict[str, Any]) -> None:
    target_root = snapshot / "tracked_files"
    for record in manifest["authoritative_tracked_files"]:
        src = ROOT / record["path"]
        dst = target_root / record["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_cache_metadata(snapshot: Path, root: Path) -> None:
    src_cache = cache_dir(root)
    dst_cache = snapshot / "cache_metadata"
    for item in natural_cache_dirs(src_cache):
        for name in CACHE_METADATA_NAMES:
            src = item / name
            if src.exists():
                dst = dst_cache / item.name / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def create_snapshot(args: argparse.Namespace) -> int:
    root = external_root()
    snapshot = root / SNAPSHOT_RELATIVE
    if snapshot.exists():
        raise SystemExit(f"snapshot already exists: {snapshot}")

    manifest = build_manifest(root, include_raw_checksums=args.include_raw_cache_checksums)
    write_json(TRACKED_MANIFEST, manifest)

    snapshot.mkdir(parents=True)
    copy_tracked_files(snapshot, manifest)
    copy_cache_metadata(snapshot, root)
    write_json(snapshot / "manifest.json", manifest)
    write_json(snapshot / "checksums.json", {"tracked_files": manifest["authoritative_tracked_files"]})
    write_csv(snapshot / "cache_inventory.csv", manifest["cache_inventory"])
    write_json(snapshot / "superseded_artifacts.json", {"items": SUPERSEDED_ARTIFACTS})
    protect_tree(snapshot)

    print(
        json.dumps(
            {
                "status": "created",
                "snapshot": str(snapshot),
                "tracked_manifest": str(TRACKED_MANIFEST),
                "tracked_files": len(manifest["authoritative_tracked_files"]),
                "cache_video_dirs": manifest["cache_summary"]["cache_video_dirs"],
                "raw_output_count": manifest["cache_summary"]["raw_output_count"],
            },
            indent=2,
        )
    )
    return 0


def verify_file_record(record: dict[str, Any], base: Path, errors: list[str], label: str) -> None:
    path = base / record["path"]
    if not path.exists():
        errors.append(f"missing {label}: {record['path']}")
        return
    if path.stat().st_size != int(record["bytes"]):
        errors.append(f"size mismatch {label}: {record['path']}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        errors.append(f"sha256 mismatch {label}: {record['path']}")


def verify_snapshot_copy(manifest: dict[str, Any], snapshot: Path, errors: list[str]) -> None:
    if not snapshot.is_dir():
        errors.append(f"missing external snapshot directory: {snapshot}")
        return
    external_manifest = snapshot / "manifest.json"
    if not external_manifest.exists():
        errors.append(f"missing external snapshot manifest: {external_manifest}")
    for record in manifest["authoritative_tracked_files"]:
        verify_file_record(record, snapshot / "tracked_files", errors, "snapshot copy")
    for record in manifest["cache_inventory"]:
        video_id = str(record["video_id"])
        for metadata in record.get("metadata_files", []):
            path = snapshot / "cache_metadata" / metadata["path"]
            if not path.exists():
                errors.append(f"missing snapshot cache metadata: {video_id}/{metadata['path']}")
                continue
            if path.stat().st_size != int(metadata["bytes"]):
                errors.append(f"size mismatch snapshot cache metadata: {video_id}/{metadata['path']}")
            if sha256_file(path) != metadata["sha256"]:
                errors.append(f"sha256 mismatch snapshot cache metadata: {video_id}/{metadata['path']}")


def verify_live_cache(manifest: dict[str, Any], root: Path, include_raw_checksums: bool, errors: list[str]) -> dict[str, Any]:
    live_inventory, live_summary = cache_records(root, include_raw_checksums=include_raw_checksums)
    expected_by_id = {str(row["video_id"]): row for row in manifest["cache_inventory"]}
    live_by_id = {str(row["video_id"]): row for row in live_inventory}
    if set(expected_by_id) != set(live_by_id):
        errors.append("live cache video-id set differs from frozen manifest")
    for key in ("cache_video_dirs", "raw_output_count", "complete_status_count", "modality_counts"):
        if live_summary.get(key) != manifest["cache_summary"].get(key):
            errors.append(f"live cache summary mismatch: {key}")
    for video_id, expected in expected_by_id.items():
        live = live_by_id.get(video_id)
        if not live:
            continue
        for key in ("raw_output_present", "raw_output_bytes", "cache_status_present", "cache_status_complete", "tribe_summary_present", "present_modalities"):
            if live.get(key) != expected.get(key):
                errors.append(f"live cache mismatch video={video_id} field={key}")
        if include_raw_checksums and expected.get("raw_output_sha256") and live.get("raw_output_sha256") != expected.get("raw_output_sha256"):
            errors.append(f"live raw cache sha256 mismatch video={video_id}")
    return live_summary


def verify_snapshot(args: argparse.Namespace) -> int:
    if not TRACKED_MANIFEST.exists():
        raise SystemExit(f"tracked evidence manifest missing: {TRACKED_MANIFEST}")
    root = external_root()
    manifest = json.loads(TRACKED_MANIFEST.read_text(encoding="utf-8"))
    if args.include_raw_cache_checksums and not manifest["cache_summary"].get("raw_cache_checksums_included"):
        raise SystemExit(
            "raw cache checksums were not recorded in this frozen manifest; "
            "create a new snapshot with --include-raw-cache-checksums to enable that check"
        )
    snapshot = root / Path(manifest["external_snapshot_relative_path"])
    errors: list[str] = []

    for record in manifest["authoritative_tracked_files"]:
        verify_file_record(record, ROOT, errors, "tracked file")
    verify_snapshot_copy(manifest, snapshot, errors)
    live_summary = verify_live_cache(manifest, root, args.include_raw_cache_checksums, errors)

    if errors:
        for error in errors:
            print(f"error: {error}")
        print(f"evidence_bundle\tfail\t{len(errors)}")
        return 1

    print(
        json.dumps(
            {
                "status": "pass",
                "snapshot_id": manifest["snapshot_id"],
                "snapshot": str(snapshot),
                "tracked_files": len(manifest["authoritative_tracked_files"]),
                "cache_video_dirs": live_summary["cache_video_dirs"],
                "raw_output_count": live_summary["raw_output_count"],
                "modality_counts": live_summary["modality_counts"],
                "raw_cache_checksums_checked": args.include_raw_cache_checksums,
                "does_not_reencode_video": True,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze or verify the VEATIC-124 v2 evidence bundle.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create-snapshot", action="store_true", help="Create the protected external snapshot and tracked manifest.")
    action.add_argument("--verify", action="store_true", help="Verify tracked evidence, snapshot copies, and live cache metadata.")
    parser.add_argument(
        "--include-raw-cache-checksums",
        action="store_true",
        help="Also hash the large tribe_raw_output.npz files. Slower, but still no re-encoding.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.create_snapshot:
        return create_snapshot(args)
    if args.verify:
        return verify_snapshot(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
