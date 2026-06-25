#!/usr/bin/env python3
"""Create the lightweight Neural Bridge Phase 0-5 evidence bundle."""

from __future__ import annotations

import csv
import fnmatch
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "evidence_bundle_phase0_to_phase5_20260625"
BUNDLE = ROOT / BUNDLE_NAME
MAX_SIZE = 70 * 1024 * 1024
EXT_ROOT = Path(
    os.environ.get(
        "NEURAL_BRIDGE_EXTERNAL_ROOT",
        str(Path("/Volumes") / "onn. Drive" / "Neural Bridge"),
    )
)
PHASE4_EXT = EXT_ROOT / "outputs/again_dense_2hz_phase4_pca_bridge_20260625_full"
DENSE_ROOT = ROOT / ".cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz"
SANITIZE_REPLACEMENTS = (
    (str(ROOT), "$NEURAL_BRIDGE_REPO"),
    (str(EXT_ROOT), "$NEURAL_BRIDGE_EXTERNAL_ROOT"),
)

HARD_EXCLUDE_PARTS = {
    ".git",
    ".cache",
    "per_video",
    "features",
    "pca_components",
    "score_parts",
    "cache",
    "checkpoints",
    "__pycache__",
    "node_modules",
}
HARD_EXCLUDE_EXTS = {
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wav",
}
LIGHTWEIGHT_PATTERNS = ("*.md", "*.csv", "*.json", "*.jsonl", "*.parquet", "*.txt")
APPROVED_OVERSIZE = {"again_dense_root_metadata/labels_aligned_2hz.parquet"}
TEXT_EXTS_FOR_SANITIZE = {".csv", ".json", ".jsonl", ".md", ".txt"}
COMPRESS_OVERSIZE_EXTS = {".csv"}
SKIP_SOURCE_NAME_PATTERNS = (
    "veatic_neuro_benchmark_124video_cortical_*.json",
    "veatic_neuro_benchmark_124video_cortical_*.summary.md",
)


included: list[dict[str, object]] = []
compressed_large: list[dict[str, object]] = []
omitted_large: list[dict[str, object]] = []
omitted_hard: list[dict[str, object]] = []
seen: set[Path] = set()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: Path) -> int:
    return int(path.stat().st_size)


def sanitize_text(value: str) -> str:
    sanitized = value
    for needle, replacement in SANITIZE_REPLACEMENTS:
        sanitized = sanitized.replace(needle, replacement)
    return sanitized


def sanitize_path(value: Path | str) -> str:
    return sanitize_text(str(value))


def sanitize_bundle_file(path: Path) -> None:
    if path.suffix.lower() not in TEXT_EXTS_FOR_SANITIZE:
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    # Rewriting also normalizes copied CSV/JSON/Markdown evidence to LF so
    # git's whitespace gate does not flag CRLF bytes from upstream exports.
    path.write_text(sanitize_text(text), encoding="utf-8")


def gzip_sanitized_text(src: Path, dest: Path) -> None:
    text = src.read_text(encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as handle:
            handle.write(sanitize_text(text).encode("utf-8"))


def hard_excluded_dest(dest_rel: Path) -> str | None:
    parts = set(dest_rel.parts)
    if parts & HARD_EXCLUDE_PARTS:
        return "hard_exclude_path_component"
    if dest_rel.suffix.lower() in HARD_EXCLUDE_EXTS:
        return "hard_exclude_extension"
    return None


def default_dest(src: Path) -> Path:
    resolved = src.resolve()
    try:
        return resolved.relative_to(ROOT.resolve())
    except ValueError:
        try:
            return Path("external_root") / resolved.relative_to(EXT_ROOT.resolve())
        except ValueError:
            return Path("external_absolute") / resolved.name


def add_file(src: Path, reason: str, *, dest_rel: Path | None = None, approve_oversize: bool = False) -> None:
    src = src.resolve()
    if not src.exists() or not src.is_file():
        return
    dest_rel = dest_rel or default_dest(src)
    if src in seen and dest_rel != default_dest(src):
        # Allow a deliberate second copy only if it has a different canonical destination.
        pass
    elif src in seen:
        return
    seen.add(src)
    dest_rel = Path(str(dest_rel))
    size = file_size(src)
    digest = sha256(src)
    hard_reason = hard_excluded_dest(dest_rel)
    if hard_reason:
        omitted_hard.append(
            {
                "source_path": sanitize_path(src),
                "bundle_path": str(dest_rel),
                "size_bytes": size,
                "sha256": digest,
                "reason": hard_reason,
            }
        )
        return
    if size > MAX_SIZE and not (approve_oversize or str(dest_rel) in APPROVED_OVERSIZE):
        if src.suffix.lower() in COMPRESS_OVERSIZE_EXTS:
            compressed_rel = Path("large_evidence_compressed") / Path(f"{dest_rel}.gz")
            compressed_dest = BUNDLE / compressed_rel
            gzip_sanitized_text(src, compressed_dest)
            compressed_row = {
                "source_path": sanitize_path(src),
                "original_bundle_path": str(dest_rel),
                "bundle_path": str(compressed_rel),
                "original_size_bytes": size,
                "original_sha256": digest,
                "size_bytes": file_size(compressed_dest),
                "sha256": sha256(compressed_dest),
                "compression": "gzip",
                "reason": f"compressed_oversize_{reason}",
                "oversize_approved": True,
            }
            included.append(compressed_row)
            compressed_large.append(compressed_row)
            return
        omitted_large.append(
            {
                "source_path": sanitize_path(src),
                "bundle_path": str(dest_rel),
                "size_bytes": size,
                "sha256": digest,
                "reason": f"exceeds_70MB_cap_for_{reason}",
            }
        )
        return
    dest = BUNDLE / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    sanitize_bundle_file(dest)
    size = file_size(dest)
    digest = sha256(dest)
    included.append(
        {
            "source_path": sanitize_path(src),
            "bundle_path": str(dest_rel),
            "size_bytes": size,
            "sha256": digest,
            "reason": reason,
            "oversize_approved": bool(size > MAX_SIZE),
        }
    )


def add_tree(base: Path, reason: str, patterns: tuple[str, ...] = LIGHTWEIGHT_PATTERNS) -> None:
    if not base.exists():
        return
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in SKIP_SOURCE_NAME_PATTERNS):
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            add_file(path, reason)


def add_dense_file(rel: str, reason: str, *, approve_oversize: bool = False) -> None:
    add_file(
        DENSE_ROOT / rel,
        reason,
        dest_rel=Path("again_dense_root_metadata") / rel,
        approve_oversize=approve_oversize,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_metadata() -> dict[str, object]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run(["git", "status", "--short"])
    return {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_commit_short": run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(status),
        "git_status_short": status or "",
    }


def build_bundle() -> None:
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    for name in ["README.md", "ROADMAP.md", "REQUIREMENTS.md", "AGENTS.md", ".codexignore"]:
        add_file(ROOT / name, "top_level_project_doc")

    for path in [
        "docs/PROJECT_MEMORY.md",
        "docs/current_project_state.md",
        "docs/again_dense_h100_cache.md",
        "docs/external_assets_manifest.md",
        "docs/veatic_v2_evidence_summary.md",
        "docs/veatic_v2_evidence_freeze.md",
        "docs/veatic_raw_representation_audit.md",
        "docs/README.md",
    ]:
        add_file(ROOT / path, "project_doc")

    add_tree(ROOT / "benchmarks/veatic", "veatic_benchmark_lightweight_evidence")
    for folder in [
        "outputs/veatic_124_raw_representation_audit_primary_20260620_152411",
        "outputs/veatic_124_raw_representation_tensor_export_v1",
        "outputs/veatic_124_frozen_tensor_ridge_only_20260620",
        "outputs/veatic_124_frozen_tensor_trained_heads_mps_20260620_full",
        "outputs/veatic_124_temporal_fairness_20260616_1509",
        "outputs/veatic_124_temporal_context_v2_20260616_1557",
    ]:
        add_tree(ROOT / folder, "veatic_output_lightweight_evidence")

    for folder in [
        "outputs/again_cleaned_inventory_audit_20260621_123531",
        "outputs/again_alignment_offset_diagnosis_20260621_131041",
        "outputs/again_boundary_aligned_1hz_manifest_20260621_205412",
    ]:
        add_tree(ROOT / folder, "again_phase0_audit_handoff")
    for report in [
        "reports/again_dense_h100_local_audit_20260625.md",
        "reports/again_video_audio_stream_inventory_20260622.md",
    ]:
        add_file(ROOT / report, "again_phase0_report")

    if DENSE_ROOT.exists():
        for rel in [
            "BASELINE_READINESS.md",
            "README_OUTPUT_SCHEMA.md",
            "global_run_metadata.json",
            "output_schema.json",
            "summary_report.json",
            "video_metadata.csv",
            "row_index.csv",
            "row_index.parquet",
            "labels_aligned_2hz.README.md",
            "labels_aligned_2hz_summary.json",
            "splits_by_video.json",
            "splits_duration_balanced.json",
            "splits_quality_filtered.json",
            "_run/global_run_metadata.json",
            "_run/output_schema.json",
            "_run/summary_report.json",
            "_run/cache_repair_20260625_stale_success_tracebacks.json",
            "_derived/raw_cortical_block_summary_b256.json",
            "_derived/temporal_diagnostics_summary_features.json",
        ]:
            add_dense_file(rel, "again_dense_root_lightweight_metadata")
        add_dense_file("labels_aligned_2hz.parquet", "again_phase1_approved_label_parquet", approve_oversize=True)

    add_file(ROOT / "reports/again_labels_aligned_2hz_20260625_091209.md", "again_phase1_label_alignment_report")
    add_file(ROOT / "reports/again_dense_2hz_ar_baseline_20260625_093722.md", "again_phase2_ar_report")
    add_tree(ROOT / "outputs/again_dense_2hz_ar_baseline_20260625_093714", "again_phase2_ar_output")
    add_file(ROOT / "reports/again_dense_2hz_raw_cortical_vs_ar_20260625_094242.md", "again_phase3_raw_cortical_report")
    add_tree(ROOT / "outputs/again_dense_2hz_raw_cortical_benchmark_20260625_093733", "again_phase3_raw_cortical_output")

    for report in [
        "reports/again_dense_2hz_phase4_pca_feature_build_20260625_153419.md",
        "reports/again_dense_2hz_phase4_pca_bridge_benchmark_20260625_153419.md",
        "reports/again_dense_2hz_phase4_pca_promotion_summary_20260625_153419.md",
    ]:
        add_file(ROOT / report, "again_phase4_report")
    phase4_root = PHASE4_EXT if PHASE4_EXT.exists() else ROOT / "outputs/again_dense_2hz_phase4_pca_bridge_20260625_full"
    for rel in ["summary.json", "run_manifest.json", "pca_feature_manifest.csv", "pca_feature_manifest.json"]:
        add_file(phase4_root / rel, "again_phase4_manifest")
    for subdir in ["metrics", "promotion", "diagnostics"]:
        add_tree(phase4_root / subdir, f"again_phase4_{subdir}")

    for report in [
        "reports/again_dense_2hz_phase5_feature_inputs_182423.md",
        "reports/again_dense_2hz_phase5_learned_heads_benchmark_182423.md",
        "reports/again_dense_2hz_phase5_promotion_summary_182423.md",
    ]:
        add_file(ROOT / report, "again_phase5_report")
    for folder in [
        "outputs/again_dense_2hz_phase5_learned_heads_20260625_182423",
        "outputs/again_dense_2hz_phase5_learned_heads_20260625_185338",
    ]:
        base = ROOT / folder
        for rel in ["summary.json", "run_manifest.json", "feature_input_manifest.json", "model_config_manifest.json"]:
            add_file(base / rel, "again_phase5_manifest")
        for subdir in ["metrics", "promotion", "diagnostics", "training_curves"]:
            add_tree(base / subdir, f"again_phase5_{subdir}")

    for script in [
        "backend/scripts/again_dense_2hz_benchmark.py",
        "backend/scripts/audit_again_dense_h100_cache.py",
        "backend/scripts/build_again_labels_aligned_2hz.py",
        "backend/scripts/run_again_dense_2hz_ar_baseline.py",
        "backend/scripts/run_again_dense_2hz_raw_cortical_benchmark.py",
        "backend/scripts/again_dense_2hz_phase4_pca_bridge.py",
        "backend/scripts/build_again_dense_2hz_train_only_pca_features.py",
        "backend/scripts/run_again_dense_2hz_pca_bridge_benchmark.py",
        "backend/scripts/summarize_again_dense_2hz_phase4_pca_bridge.py",
        "backend/scripts/run_again_dense_2hz_phase5_learned_heads.py",
        "backend/scripts/run_veatic_strict_benchmark.py",
        "backend/scripts/run_veatic_neuro_benchmark.py",
        "backend/scripts/run_veatic_event_spike_retest.py",
        "backend/scripts/run_veatic_event_conditioned_retest.py",
        "backend/scripts/run_veatic_raw_representation_audit.py",
        "backend/scripts/run_veatic_frozen_tensor_incremental_benchmark.py",
        "backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py",
        "backend/scripts/veatic_frozen_tensor_adapter.py",
        "backend/scripts/veatic_frozen_tensor_trained_heads.py",
        "tools/export_veatic_raw_representation_metadata_bundle.py",
        "tools/export_veatic_raw_representation_tensors.py",
    ]:
        add_file(ROOT / script, "result_producing_script")

    for path in sorted((ROOT / "tests").glob("test_*")):
        if path.is_file() and any(token in path.name for token in ["again_dense", "veatic", "grouped", "mlx_vjepa21"]):
            add_file(path, "benchmark_validation_test")

    write_bundle_readme()
    write_manifests()


def write_bundle_readme() -> None:
    readme = f"""# Neural Bridge Evidence Bundle: Phase 0-5 (2026-06-25)

This lightweight bundle packages review evidence for Neural Bridge from VEATIC through dense AGAIN Phase 0-5. It is intentionally not a runnable cache bundle: heavy tensors, videos, checkpoints, model weights, PCA arrays, and per-video feature caches are excluded.

## Phase Map

- VEATIC evidence: strict VEATIC-124 v2 benchmarks, timing/fairness checks, raw-vs-compressed representation audit, frozen tensor exports, and trained-head summaries.
- AGAIN Phase 0: dataset/H100/cache readiness audits, cleaned inventory, alignment/boundary handoff evidence, and dense-root metadata.
- AGAIN Phase 1: true 2Hz label alignment and target construction evidence.
- AGAIN Phase 2: AR-only 2Hz baseline evidence.
- AGAIN Phase 3: raw cortical/TRIBE-v2 versus AR benchmark evidence.
- AGAIN Phase 4: fold-safe train-only PCA bridge benchmark evidence.
- AGAIN Phase 5: MLX learned-head benchmark evidence plus compact label-permutation sanity run.

## Canonical Evidence Locations

- Top-level docs: `README.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `AGENTS.md`, `.codexignore`.
- Project docs: `docs/PROJECT_MEMORY.md`, `docs/current_project_state.md`, `docs/again_dense_h100_cache.md`, `docs/external_assets_manifest.md`, and VEATIC evidence docs.
- VEATIC: `benchmarks/veatic/` and selected `outputs/veatic_*` metadata/report folders.
- AGAIN Phase 0: `outputs/again_cleaned_inventory_audit_20260621_123531/`, `outputs/again_alignment_offset_diagnosis_20260621_131041/`, `reports/again_dense_h100_local_audit_20260625.md`, and `again_dense_root_metadata/`.
- AGAIN Phase 1: `reports/again_labels_aligned_2hz_20260625_091209.md`, `again_dense_root_metadata/labels_aligned_2hz.parquet`, and label summaries.
- AGAIN Phase 2: `reports/again_dense_2hz_ar_baseline_20260625_093722.md` and `outputs/again_dense_2hz_ar_baseline_20260625_093714/`.
- AGAIN Phase 3: `reports/again_dense_2hz_raw_cortical_vs_ar_20260625_094242.md` and `outputs/again_dense_2hz_raw_cortical_benchmark_20260625_093733/`.
- AGAIN Phase 4: phase 4 reports plus `external_root/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/` lightweight metrics, promotion gates, diagnostics, and manifests.
- AGAIN Phase 5: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/` and `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`, excluding checkpoints.
- Compressed large evidence: `large_evidence_compressed/` contains gzip-compressed CSVs that were too large to keep as raw CSV in git.

## Headline Results

- VEATIC best spike PR-AUC: `0.2536` vs AR `0.1969`.
- AGAIN Phase 3 AR+raw PR-AUC: `0.17030` vs AR `0.14725`.
- AGAIN Phase 4 best PR-AUC: `0.17165` vs AR `0.14725`.
- AGAIN Phase 5 main learned-head PR-AUC: `0.21913` vs Phase 4 `0.17165` and AR `0.14725`.
- AGAIN Phase 5 label-permutation follow-up best real grouped PR-AUC: `0.22458`; label permutation PR-AUC `0.10428`; blocked temporal real PR-AUC `0.20712`.

## Intentional Omissions

Hard-excluded payloads include `per_video/`, `features/`, `pca_components/`, `score_parts/`, `cache/`, `checkpoints/`, broad `.cache/` payload paths, NumPy/model binaries, videos, and audio. Critical evidence files under 70MB are allowed. The approved explicit exception is `again_dense_root_metadata/labels_aligned_2hz.parquet`, which preserves Phase 1 labels for review. Oversize CSV evidence is stored as deterministic gzip under `large_evidence_compressed/`. Larger non-CSV files are otherwise omitted. See `omitted_large_files_manifest.json`.

## Compressed Large Evidence

The following raw CSVs exceeded the normal evidence cap and are included as `.csv.gz` inside `large_evidence_compressed/`:

- `again_dense_root_metadata/row_index.csv`
- `outputs/again_boundary_aligned_1hz_manifest_20260621_205412/again_boundary_aligned_1hz_manifest.csv`
- `outputs/again_cleaned_inventory_audit_20260621_123531/again_manifest_proposal.csv`

To inspect one:

```bash
gzip -cd large_evidence_compressed/again_dense_root_metadata/row_index.csv.gz | head
```

## Verify Checksums

```bash
python3 - <<'PY'
import hashlib, json, pathlib
bundle = pathlib.Path('evidence_bundle_phase0_to_phase5_20260625')
rows = json.loads((bundle / 'checksum_manifest.json').read_text())['files']
for row in rows:
    path = bundle / row['bundle_path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row['sha256']:
        raise SystemExit(f"checksum mismatch: {{path}}")
print(f"verified {{len(rows)}} bundled files")
PY
```

Generated at `{datetime.now(timezone.utc).isoformat()}`.
"""
    path = BUNDLE / "BUNDLE_README.md"
    path.write_text(readme, encoding="utf-8")
    included.append(
        {
            "source_path": "<generated>",
            "bundle_path": "BUNDLE_README.md",
            "size_bytes": file_size(path),
            "sha256": sha256(path),
            "reason": "generated_bundle_readme",
            "oversize_approved": False,
        }
    )


def write_manifests() -> None:
    files = sorted(included, key=lambda row: str(row["bundle_path"]))
    generated_manifest_names = [
        "bundle_manifest.json",
        "bundle_file_inventory.csv",
        "omitted_large_files_manifest.json",
        "checksum_manifest.json",
    ]
    manifest = {
        "bundle_name": BUNDLE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_individual_file_size_bytes": MAX_SIZE,
        "approved_oversize_files": sorted(APPROVED_OVERSIZE),
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files),
        "headline_results": {
            "veatic_best_spike_pr_auc": 0.2536,
            "veatic_ar_pr_auc": 0.1969,
            "again_phase3_ar_plus_raw_pr_auc": 0.17030,
            "again_phase3_ar_pr_auc": 0.14725,
            "again_phase4_best_pr_auc": 0.17165,
            "again_phase5_main_best_pr_auc": 0.21913,
            "again_phase5_followup_best_real_grouped_pr_auc": 0.22458,
            "again_phase5_followup_label_permutation_pr_auc": 0.10428,
            "again_phase5_followup_blocked_temporal_real_pr_auc": 0.20712,
        },
        "git": git_metadata(),
        "hard_exclude_parts": sorted(HARD_EXCLUDE_PARTS),
        "hard_exclude_extensions": sorted(HARD_EXCLUDE_EXTS),
        "compressed_large_files": sorted(compressed_large, key=lambda row: str(row["bundle_path"])),
        "generated_manifest_files": generated_manifest_names,
        "files": files,
    }
    (BUNDLE / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(BUNDLE / "bundle_file_inventory.csv", files)
    (BUNDLE / "omitted_large_files_manifest.json").write_text(
        json.dumps(
            {
                "files": sorted(omitted_large, key=lambda row: str(row["source_path"])),
                "compressed_large_files": sorted(compressed_large, key=lambda row: str(row["bundle_path"])),
                "hard_excluded_seen_count": len(omitted_hard),
                "hard_excluded_examples": sorted(omitted_hard, key=lambda row: str(row["source_path"]))[:50],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checksum_files = files[:]
    for name in ["bundle_manifest.json", "bundle_file_inventory.csv", "omitted_large_files_manifest.json"]:
        path = BUNDLE / name
        checksum_files.append(
            {
                "source_path": "<generated>",
                "bundle_path": name,
                "size_bytes": file_size(path),
                "sha256": sha256(path),
                "reason": "generated_bundle_manifest",
                "oversize_approved": False,
            }
        )
    (BUNDLE / "checksum_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "bundle_path": row["bundle_path"],
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in sorted(checksum_files, key=lambda row: str(row["bundle_path"]))
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_bundle()
    print(
        json.dumps(
            {
                "bundle": str(BUNDLE),
                "compressed_large_files": len(compressed_large),
                "included_files": len(included),
                "total_size_mb": round(sum(int(row["size_bytes"]) for row in included) / 1024 / 1024, 2),
                "large_omissions": len(omitted_large),
                "hard_excluded_seen": len(omitted_hard),
            },
            indent=2,
            sort_keys=True,
        )
    )
