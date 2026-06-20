#!/usr/bin/env python3
"""Export a metadata-only VEATIC raw representation review bundle.

This is a recovery/export helper, not a benchmark runner. It slices existing
audit reports, reads checkpoint/cache metadata, and reconstructs top-k vertex
index lists from the existing raw cortical cache without rescoring candidates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.cortical_roi_mapper import (  # noqa: E402
    CorticalRoiMapper,
    FSAVERAGE5_VERTICES_PER_HEMI,
    TRIBE_CORTICAL_VERTICES,
)
from backend.scripts import run_veatic_raw_representation_audit as audit  # noqa: E402


SOURCE_AUDIT_NAME = "veatic_124_raw_representation_audit_primary_20260620_152411"
EXTERNAL_ROOT_VALUE = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT")
EXTERNAL_ROOT = Path(EXTERNAL_ROOT_VALUE).expanduser() if EXTERNAL_ROOT_VALUE else ROOT / ".missing_external_root"
SOURCE_AUDIT_DIR = Path(
    os.environ.get(
        "VEATIC_RAW_REPRESENTATION_AUDIT_DIR",
        str(EXTERNAL_ROOT / "outputs" / SOURCE_AUDIT_NAME),
    )
).expanduser()
TRACKED_AUDIT_DIR = ROOT / "outputs" / SOURCE_AUDIT_DIR.name
BUNDLE_NAME = "veatic_raw_representation_review_bundle_20260620_v2"
BUNDLE_DIR = ROOT / BUNDLE_NAME
ZIP_PATH = ROOT / f"{BUNDLE_NAME}.zip"
CHECKPOINT_STATE = SOURCE_AUDIT_DIR / "_checkpoint" / "state.json"
FIT_CACHE_DIR = SOURCE_AUDIT_DIR / "_checkpoint" / "fit_cache"
DEFAULT_REPORT = ROOT / "benchmarks" / "veatic" / "veatic_manifest_124_complete_20260616.report.json"

GLOBAL_REPORTS = [
    "representation_audit_report.md",
    "raw_cache_inventory_report.md",
    "raw_cache_inventory_summary.json",
    "raw_cache_inventory.csv",
    "raw_vs_compressed_leaderboard.csv",
    "grouped_video_results.csv",
    "fixed_split_results.csv",
    "representation_results_primary.csv",
    "representation_results_all.csv",
    "control_results.csv",
    "matched_row_context_results.csv",
    "supervised_feature_selection_stability.csv",
]

TOP_LEVEL_COPIES = [
    "run_manifest.json",
    "leakage_audit.json",
    "candidate_promotion_summary.json",
    "representation_candidates.json",
]

FORBIDDEN_PAYLOAD_MARKERS = (
    "tribe_raw_output.npz",
    "huggingface",
    ".cache/huggingface",
    "/models/",
    "\\models\\",
)
FORBIDDEN_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_json_maybe(value: Any) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class BundleWriter:
    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        self.inventory: list[dict[str, Any]] = []

    def _record(self, path: Path, original: Path | None, reason: str, candidate: str | None) -> None:
        rel = path.relative_to(self.bundle_dir).as_posix()
        self.inventory.append(
            {
                "original_absolute_path": str(original.resolve()) if original else None,
                "bundle_relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "reason_included": reason,
                "candidate_name": candidate,
            }
        )

    def copy(self, src: Path, rel: str, reason: str, candidate: str | None = None) -> None:
        if not src.exists():
            return
        dest = self.bundle_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self._record(dest, src, reason, candidate)

    def json(self, rel: str, payload: Any, reason: str, candidate: str | None = None) -> None:
        dest = self.bundle_dir / rel
        write_json(dest, payload)
        self._record(dest, None, reason, candidate)

    def text(self, rel: str, text: str, reason: str, candidate: str | None = None) -> None:
        dest = self.bundle_dir / rel
        write_text(dest, text)
        self._record(dest, None, reason, candidate)

    def csv(self, rel: str, rows: list[dict[str, Any]], reason: str, candidate: str | None = None) -> None:
        dest = self.bundle_dir / rel
        write_csv(dest, rows)
        self._record(dest, None, reason, candidate)


def load_state() -> dict[str, Any]:
    return json.loads(CHECKPOINT_STATE.read_text(encoding="utf-8"))


def candidate_names(state: dict[str, Any]) -> list[str]:
    configured = state.get("config", {}).get("candidate_names") or []
    if configured:
        return [str(item) for item in configured]
    payload = json.loads((SOURCE_AUDIT_DIR / "representation_candidates.json").read_text(encoding="utf-8"))
    return [str(item["name"]) for item in payload]


def split_candidate_rows(rows: list[dict[str, str]], candidate: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("candidate") == candidate]


def matched_candidate_rows(rows: list[dict[str, str]], candidate: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("candidate") == candidate or row.get("matched_context_candidate") == candidate
    ]


def hyperparameter_rows(state: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for collection in ("result_rows", "matched_rows"):
        for row in state.get(collection, []):
            if row.get("candidate") != candidate and row.get("matched_context_candidate") != candidate:
                continue
            raw = row.get("selected_hyperparameters")
            if not raw:
                continue
            parsed = parse_json_maybe(raw)
            entry = {
                "source_collection": collection,
                "scope": row.get("scope"),
                "split": row.get("split"),
                "held_out_video_ids": row.get("held_out_video_ids"),
                "target": row.get("target"),
                "threshold": row.get("threshold"),
                "task_type": row.get("task_type"),
                "target_tier": row.get("target_tier"),
                "head": row.get("head"),
                "model": row.get("model"),
                "selected_hyperparameters": parsed,
            }
            key = json.dumps(entry, sort_keys=True)
            if key not in seen:
                selected.append(entry)
                seen.add(key)
    return selected


def build_pca_fit_cache_manifest() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for path in sorted(FIT_CACHE_DIR.rglob("*.npz")):
        rel = path.relative_to(FIT_CACHE_DIR).as_posix()
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {}
            for key in bundle.files:
                arr = bundle[key]
                arrays[key] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "nbytes": int(arr.nbytes),
                }
            meta = json.loads(str(np.asarray(bundle["pca_meta_json"]).item()))
        scope_split = path.parent.name
        components = int(meta.get("actual_components") or meta.get("requested_components") or 0)
        related = []
        if components == 64:
            related.extend(["cortical_pca_64", "cortical_pca64_delta"])
        if components == 128:
            related.extend(
                [
                    "pca_current_128",
                    "pca_delta_128",
                    "pca_sequence_128_causal_past_2s_mean",
                    "pca_sequence_128_causal_past_2s_mean_std_last_slope",
                ]
            )
        if components == 256:
            related.extend(["pca_current_256", "pca_delta_256", "pca_sequence_256_causal_past_2s_mean"])
        entry = {
            "original_absolute_path": str(path.resolve()),
            "relative_to_fit_cache": rel,
            "scope_split": scope_split,
            "file_size_bytes": path.stat().st_size,
            "arrays": arrays,
            "pca_metadata": meta,
            "related_candidates_inferred_from_component_count": related,
            "payload_copied": False,
        }
        manifest.append(entry)
        csv_rows.append(
            {
                "relative_to_fit_cache": rel,
                "scope_split": scope_split,
                "file_size_bytes": path.stat().st_size,
                "projected_all_shape": json.dumps(arrays.get("projected_all", {}).get("shape")),
                "projected_all_dtype": arrays.get("projected_all", {}).get("dtype"),
                "requested_components": meta.get("requested_components"),
                "actual_components": meta.get("actual_components"),
                "train_rows": meta.get("train_rows"),
                "backend": meta.get("backend"),
                "explained_variance_ratio_sum": meta.get("explained_variance_ratio_sum"),
                "fit_cache_key": meta.get("fit_cache_key"),
                "related_candidates": json.dumps(related),
                "payload_copied": False,
            }
        )
    return manifest, csv_rows


def build_roi_metadata() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    atlas = CorticalRoiMapper().load_destrieux_atlas()
    if not atlas:
        raise RuntimeError("Destrieux atlas unavailable through CorticalRoiMapper")
    left = np.asarray(atlas["left"])
    right = np.asarray(atlas["right"])
    labels = np.concatenate([left, right])
    if labels.shape[0] != TRIBE_CORTICAL_VERTICES:
        raise RuntimeError(f"Atlas label count {labels.shape[0]} != {TRIBE_CORTICAL_VERTICES}")
    label_names = [str(item) for item in atlas.get("labels", [])]
    parcels = [int(label) for label in sorted(set(labels.tolist())) if int(label) >= 0]
    parcel_rows = []
    parcel_sizes: dict[str, int] = {}
    for label in parcels:
        left_count = int(np.sum(left == label))
        right_count = int(np.sum(right == label))
        total = left_count + right_count
        parcel_sizes[str(label)] = total
        parcel_rows.append(
            {
                "label_id": label,
                "label_name": label_names[label] if 0 <= label < len(label_names) else None,
                "left_vertex_count": left_count,
                "right_vertex_count": right_count,
                "total_vertex_count": total,
            }
        )
    metadata = {
        "source": "CorticalRoiMapper().load_destrieux_atlas()",
        "atlas": "destrieux_surface",
        "fsaverage5_vertices_per_hemi": FSAVERAGE5_VERTICES_PER_HEMI,
        "tribe_cortical_vertices": TRIBE_CORTICAL_VERTICES,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "combined_label_shape": list(labels.shape),
        "label_name_count": len(label_names),
        "parcel_count": len(parcel_rows),
        "parcel_sizes": parcel_sizes,
        "labels": label_names,
        "model_scoring_performed": False,
        "raw_tensor_built": False,
    }
    return metadata, parcel_rows


def topk_selected_vertices(raw: np.ndarray, train_idx: np.ndarray, y_train: np.ndarray, k: int) -> tuple[np.ndarray, str, str]:
    train_x = raw[train_idx]
    y = np.asarray(y_train, dtype=np.float64)
    y_std = float(np.std(y))
    if y_std < 1e-12:
        scores = np.zeros(train_x.shape[1], dtype=np.float64)
        score_method = "constant_target"
    else:
        x_mean = train_x.mean(axis=0)
        x_centered = train_x - x_mean
        x_std = train_x.std(axis=0)
        y_centered = y - float(np.mean(y))
        denom = np.maximum(x_std * y_std * max(train_x.shape[0] - 1, 1), 1e-12)
        scores = np.abs((x_centered * y_centered[:, None]).sum(axis=0) / denom)
        score_method = "absolute_pearson_or_point_biserial"
    actual_k = min(int(k), raw.shape[1])
    selected = np.argsort(-scores, kind="mergesort")[:actual_k].astype(np.int64, copy=False)
    digest = hashlib.blake2b(selected.astype(np.int64).tobytes(), digest_size=12).hexdigest()
    return selected, digest, score_method


def build_topk_reconstruction(state: dict[str, Any], bundle: BundleWriter) -> dict[str, Any]:
    stability_rows = [
        row for row in state.get("stability_rows", []) if row.get("candidate") == "topk_vertices_512"
    ]
    if not stability_rows:
        return {"status": "not_applicable", "rows_checked": 0, "failures": []}

    run_manifest = json.loads((SOURCE_AUDIT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_path = Path(run_manifest["manifest"]).expanduser()
    report_path = Path(run_manifest.get("report") or DEFAULT_REPORT).expanduser()
    cache_dir = Path(run_manifest["cache_dir"]).expanduser()
    config = state.get("config", {})
    mode = str(config.get("mode") or run_manifest.get("mode") or "primary-audit")
    grouped_folds = int(config.get("grouped_folds") or 5)
    target_specs = {
        (item["target"], None if item.get("threshold") is None else float(item["threshold"]), item["task_type"]): item
        for item in config.get("targets", [])
    }

    ctx = audit.retest.RetestContext(manifest_path, report_path, cache_dir)
    raw = audit.build_base_feature_sets(ctx)["cortical_raw"]
    scope_rows_by_name: dict[str, list[dict[str, Any]]] = {"all_videos": ctx.accepted_rows}
    if any(row.get("scope") == "exclude_video_83" for row in stability_rows):
        scope_rows_by_name["exclude_video_83"] = [
            row for row in ctx.accepted_rows if str(row["video_id"]) != "83"
        ]
    split_rows: dict[tuple[str, str], tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for scope, rows in scope_rows_by_name.items():
        for split, held, train_rows, test_rows in audit.split_specs(rows, grouped_folds, mode):
            split_rows[(scope, split)] = (held, train_rows, test_rows)

    json_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in stability_rows:
        scope = str(row["scope"])
        split = str(row["split"])
        split_key = (scope, split)
        if split_key not in split_rows:
            failures.append({"row": row, "reason": "split rows not reconstructed"})
            continue
        target_key = (
            row["target"],
            None if row.get("threshold") is None else float(row["threshold"]),
            row["task_type"],
        )
        target_spec = target_specs.get(target_key)
        if not target_spec:
            failures.append({"row": row, "reason": "target spec not found in checkpoint config"})
            continue
        _held, train_rows, _test_rows = split_rows[split_key]
        train_selected, train_y = audit.target_rows(
            ctx,
            train_rows,
            str(target_spec["target"]),
            target_spec.get("horizon"),
            target_spec.get("threshold"),
            str(target_spec["task_type"]),
        )
        train_idx = audit.bench.row_indices(ctx.accepted_rows, train_selected)
        selected, digest, score_method = topk_selected_vertices(
            raw,
            train_idx,
            train_y,
            int(row.get("selected_vertex_count") or 512),
        )
        expected_digest = str(row.get("selected_vertices_digest"))
        expected_count = int(row.get("selected_vertex_count") or 0)
        passed = digest == expected_digest and int(selected.size) == expected_count
        result = {
            "scope": scope,
            "split": split,
            "held_out_video_ids": row.get("held_out_video_ids"),
            "target": row.get("target"),
            "threshold": row.get("threshold"),
            "task_type": row.get("task_type"),
            "target_tier": row.get("target_tier"),
            "head": row.get("head"),
            "train_row_count_after_target_filter": len(train_selected),
            "selected_vertex_count": int(selected.size),
            "expected_selected_vertex_count": expected_count,
            "selected_vertices_digest": digest,
            "expected_selected_vertices_digest": expected_digest,
            "digest_verified": passed,
            "score_method": score_method,
            "expected_score_method": row.get("score_method"),
            "selected_vertices": selected.astype(int).tolist(),
        }
        json_rows.append(result)
        csv_rows.append(
            {
                key: json.dumps(value) if key == "selected_vertices" else value
                for key, value in result.items()
            }
        )
        if not passed:
            failures.append(
                {
                    "scope": scope,
                    "split": split,
                    "target": row.get("target"),
                    "threshold": row.get("threshold"),
                    "expected_digest": expected_digest,
                    "actual_digest": digest,
                    "expected_count": expected_count,
                    "actual_count": int(selected.size),
                }
            )

    status = "pass" if not failures else "fail"
    payload = {
        "status": status,
        "rows_checked": len(json_rows),
        "failures": failures,
        "reconstruction_method": (
            "Recomputed train-only absolute Pearson/point-biserial top-k vertex "
            "selection from existing cortical_raw cache and checkpoint split/target rows."
        ),
        "model_scoring_performed": False,
        "raw_tensor_exported": False,
        "rows": json_rows,
    }
    if status != "pass":
        bundle.json(
            "candidate_artifacts/topk_vertices_512/TOPK_DIGEST_FAILURE.json",
            payload,
            "top-k deterministic reconstruction failure report",
            "topk_vertices_512",
        )
        bundle.text(
            "candidate_artifacts/topk_vertices_512/TOPK_DIGEST_FAILURE.md",
            "# topk_vertices_512 Digest Verification Failed\n\n"
            "The deterministic reconstruction did not match stored digest/count rows. "
            "No selected vertex lists should be used from this recovery attempt.\n",
            "top-k deterministic reconstruction failure report",
            "topk_vertices_512",
        )
        raise RuntimeError("topk_vertices_512 digest verification failed")

    bundle.json(
        "candidate_artifacts/topk_vertices_512/selected_vertices_by_split_target.json",
        payload,
        "digest-verified reconstructed top-k selected vertex lists",
        "topk_vertices_512",
    )
    bundle.csv(
        "candidate_artifacts/topk_vertices_512/selected_vertices_by_split_target.csv",
        csv_rows,
        "digest-verified reconstructed top-k selected vertex lists",
        "topk_vertices_512",
    )
    return payload


def build_tensor_cache_inventory(
    state: dict[str, Any],
    pca_manifest: list[dict[str, Any]],
    topk_status: dict[str, Any],
) -> dict[str, Any]:
    run_manifest = json.loads((SOURCE_AUDIT_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    cache_dir = Path(run_manifest["cache_dir"])
    raw_files = sorted(cache_dir.glob("*/tribe_raw_output.npz"))
    raw_total = sum(path.stat().st_size for path in raw_files if path.exists())
    return {
        "source_audit_dir": str(SOURCE_AUDIT_DIR),
        "tracked_lightweight_dir": str(TRACKED_AUDIT_DIR),
        "checkpoint_state": str(CHECKPOINT_STATE),
        "result_rows": len(state.get("result_rows", [])),
        "matched_rows": len(state.get("matched_rows", [])),
        "stability_rows": len(state.get("stability_rows", [])),
        "pca_fit_cache_file_count": len(pca_manifest),
        "pca_fit_cache_total_size_bytes": sum(int(item["file_size_bytes"]) for item in pca_manifest),
        "pca_fit_cache_payloads_copied": False,
        "raw_tribe_cache_dir": str(cache_dir),
        "raw_tribe_output_file_count_seen": len(raw_files),
        "raw_tribe_output_total_size_bytes_seen": raw_total,
        "raw_tribe_outputs_copied": False,
        "raw_videos_copied": False,
        "model_weights_copied": False,
        "huggingface_cache_copied": False,
        "full_candidate_tensors_rebuilt": False,
        "model_scoring_performed": False,
        "topk_digest_verification_status": topk_status.get("status"),
    }


def forbidden_payloads(bundle_dir: Path) -> list[str]:
    hits = []
    for path in bundle_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        lower = rel.lower()
        if path.suffix.lower() in FORBIDDEN_VIDEO_EXTENSIONS:
            hits.append(rel)
            continue
        if any(marker in lower for marker in FORBIDDEN_PAYLOAD_MARKERS):
            hits.append(rel)
            continue
        if path.suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".safetensors", ".bin"}:
            hits.append(rel)
    return hits


def zip_bundle(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir.parent))


def validate_zip(zip_path: Path) -> bool:
    result = subprocess.run(["unzip", "-tq", str(zip_path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return False
    return True


def main() -> int:
    start = time.monotonic()
    if not EXTERNAL_ROOT_VALUE and "VEATIC_RAW_REPRESENTATION_AUDIT_DIR" not in os.environ:
        raise RuntimeError(
            "Set NEURAL_BRIDGE_EXTERNAL_ROOT or VEATIC_RAW_REPRESENTATION_AUDIT_DIR before exporting the metadata bundle."
        )
    if not SOURCE_AUDIT_DIR.exists():
        raise FileNotFoundError(SOURCE_AUDIT_DIR)
    if not CHECKPOINT_STATE.exists():
        raise FileNotFoundError(CHECKPOINT_STATE)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True)
    bundle = BundleWriter(BUNDLE_DIR)

    state = load_state()
    candidates = candidate_names(state)
    candidate_descriptions = {
        item["name"]: item
        for item in json.loads((SOURCE_AUDIT_DIR / "representation_candidates.json").read_text(encoding="utf-8"))
    }

    for filename in TOP_LEVEL_COPIES:
        bundle.copy(SOURCE_AUDIT_DIR / filename, filename, "existing global audit metadata")
    bundle.copy(CHECKPOINT_STATE, "state.json", "checkpoint state top-level convenience copy")
    bundle.copy(CHECKPOINT_STATE, "checkpoint_state/state.json", "checkpoint state for exact run recovery")
    for filename in GLOBAL_REPORTS:
        bundle.copy(SOURCE_AUDIT_DIR / filename, f"reports/{filename}", "existing global audit report")

    csv_sources = {
        "representation_results_all": read_csv(SOURCE_AUDIT_DIR / "representation_results_all.csv"),
        "grouped_video_results": read_csv(SOURCE_AUDIT_DIR / "grouped_video_results.csv"),
        "fixed_split_results": read_csv(SOURCE_AUDIT_DIR / "fixed_split_results.csv"),
        "control_results": read_csv(SOURCE_AUDIT_DIR / "control_results.csv"),
        "matched_row_context_results": read_csv(SOURCE_AUDIT_DIR / "matched_row_context_results.csv"),
        "supervised_feature_selection_stability": read_csv(SOURCE_AUDIT_DIR / "supervised_feature_selection_stability.csv"),
    }

    for candidate in candidates:
        base = f"candidate_artifacts/{candidate}"
        all_rows = split_candidate_rows(csv_sources["representation_results_all"], candidate)
        grouped_fold_rows = [row for row in all_rows if str(row.get("split", "")).startswith("grouped_")]
        fixed_rows = split_candidate_rows(csv_sources["fixed_split_results"], candidate)
        grouped_rows = split_candidate_rows(csv_sources["grouped_video_results"], candidate)
        control_rows = split_candidate_rows(csv_sources["control_results"], candidate)
        matched_rows = matched_candidate_rows(csv_sources["matched_row_context_results"], candidate)
        stability_rows = split_candidate_rows(csv_sources["supervised_feature_selection_stability"], candidate)
        bundle.json(
            f"{base}/candidate_metadata.json",
            candidate_descriptions.get(candidate, {"name": candidate}),
            "candidate descriptor from representation_candidates.json",
            candidate,
        )
        bundle.csv(
            f"{base}/representation_results_all.csv",
            all_rows,
            "candidate slice from representation_results_all.csv",
            candidate,
        )
        bundle.csv(
            f"{base}/per_fold_scores.csv",
            grouped_fold_rows,
            "candidate grouped-fold score rows sliced from representation_results_all.csv",
            candidate,
        )
        bundle.csv(
            f"{base}/fixed_split_scores.csv",
            fixed_rows,
            "candidate fixed-split score rows sliced from fixed_split_results.csv",
            candidate,
        )
        bundle.csv(
            f"{base}/grouped_video_summary.csv",
            grouped_rows,
            "candidate grouped-video summary sliced from grouped_video_results.csv",
            candidate,
        )
        bundle.csv(
            f"{base}/control_results.csv",
            control_rows,
            "candidate controls sliced from control_results.csv",
            candidate,
        )
        if matched_rows:
            bundle.csv(
                f"{base}/matched_row_context_results.csv",
                matched_rows,
                "candidate matched-row context results sliced from global CSV",
                candidate,
            )
        if stability_rows:
            bundle.csv(
                f"{base}/supervised_feature_selection_stability.csv",
                stability_rows,
                "candidate supervised feature-selection stability rows sliced from global CSV",
                candidate,
            )
        bundle.json(
            f"{base}/hyperparameter_selection.json",
            {
                "candidate": candidate,
                "source": "checkpoint state result_rows and matched_rows",
                "rows": hyperparameter_rows(state, candidate),
            },
            "candidate hyperparameter-selection metadata extracted from checkpoint state",
            candidate,
        )

    pca_manifest, pca_csv_rows = build_pca_fit_cache_manifest()
    bundle.json(
        "checkpoint_state/fit_cache/pca_fit_cache_manifest.json",
        {
            "metadata_only": True,
            "payload_npz_files_copied": False,
            "files": pca_manifest,
        },
        "PCA fit-cache metadata and array shapes without copying NPZ payloads",
    )
    bundle.csv(
        "checkpoint_state/fit_cache/pca_fit_cache_manifest.csv",
        pca_csv_rows,
        "PCA fit-cache metadata and array shapes without copying NPZ payloads",
    )
    bundle.text(
        "checkpoint_state/fit_cache/README.md",
        "# PCA Fit Cache Metadata\n\n"
        "This metadata-only bundle does not include the original `.npz` PCA projection caches. "
        "The manifest in this directory records source paths, shapes, dtypes, sizes, and "
        "PCA metadata read from the existing checkpoint cache.\n",
        "checkpoint fit-cache metadata explanation",
    )

    roi_metadata, roi_rows = build_roi_metadata()
    bundle.json(
        "candidate_artifacts/roi_parcel_features/roi_atlas_mapping_metadata.json",
        roi_metadata,
        "ROI atlas/parcel mapping metadata from existing CorticalRoiMapper loader",
        "roi_parcel_features",
    )
    bundle.csv(
        "candidate_artifacts/roi_parcel_features/roi_parcel_sizes.csv",
        roi_rows,
        "ROI parcel size table from existing CorticalRoiMapper loader",
        "roi_parcel_features",
    )

    topk_status = build_topk_reconstruction(state, bundle)
    tensor_cache_inventory = build_tensor_cache_inventory(state, pca_manifest, topk_status)
    bundle.json(
        "tensor_cache_inventory.json",
        tensor_cache_inventory,
        "metadata-only tensor/cache inventory",
    )

    missing = [
        "Original candidate-named standalone artifacts were not found in source outputs; this bundle contains recovered slices/manifests instead.",
        "Full raw candidate tensors were not rebuilt or included by request.",
        "PCA fit-cache .npz payloads were not copied in this metadata-only v2 bundle.",
        "Raw TRIBE tribe_raw_output.npz files were read only for top-k reconstruction and were not copied.",
    ]
    readme = f"""# VEATIC-124 Raw Representation Audit Metadata Recovery Bundle v2

This bundle is for external review of the VEATIC-124 raw representation audit.
It is a metadata-only recovery/export pass built from the existing audit output,
checkpoint state, global CSV/JSON reports, PCA fit-cache metadata, ROI atlas
loader metadata, and digest-verified top-k vertex reconstruction.

- No full VEATIC representation audit was rerun.
- No video re-encoding was performed.
- No model scoring or all-candidate rescoring was performed.
- No raw videos, raw TRIBE output NPZ files, model weights, or Hugging Face
  caches were copied.
- Frozen baseline: `cortical_pca64_delta`.
- Main promoted next candidate: `pca_sequence_128_causal_past_2s_mean`.
- Important side candidate: `roi_parcel_features`.
- Useful but cautionary candidate: `topk_vertices_512`.
- Suspicious/resampled video: `83`.
- Leakage audit status: pass.
- Candidate-specific checkpoint artifacts: original standalone files were not
  present; this v2 bundle adds recovered per-candidate slices, hyperparameter
  metadata, PCA cache manifests, ROI parcel metadata, and top-k selected vertex
  lists reconstructed from the existing raw cortical cache.

Top-k digest verification status: `{topk_status.get("status")}` with
{topk_status.get("rows_checked", 0)} rows checked.

## Still Missing Or Intentionally Not Included

""" + "\n".join(f"- {item}" for item in missing) + "\n"
    bundle.text("README.md", readme, "bundle README")

    summary = {
        "bundle_name": BUNDLE_NAME,
        "source_audit_dir": str(SOURCE_AUDIT_DIR),
        "elapsed_seconds_before_zip": time.monotonic() - start,
        "candidate_count": len(candidates),
        "pca_fit_cache_entries": len(pca_manifest),
        "topk_digest_status": topk_status.get("status"),
        "topk_rows_checked": topk_status.get("rows_checked"),
        "missing_or_intentionally_not_included": missing,
        "metadata_only": True,
    }
    bundle.json("recovery_export_summary.json", summary, "recovery/export summary")

    inventory_payload = {
        "schema_version": "veatic_raw_representation_review_bundle_file_inventory_v2",
        "note": "file_inventory.json excludes its own self-referential hash; all other bundled files are listed.",
        "created_from_source_audit_dir": str(SOURCE_AUDIT_DIR),
        "files": sorted(bundle.inventory, key=lambda item: item["bundle_relative_path"]),
    }
    write_json(BUNDLE_DIR / "file_inventory.json", inventory_payload)

    forbidden = forbidden_payloads(BUNDLE_DIR)
    if forbidden:
        raise RuntimeError("Forbidden payload files found in bundle: " + json.dumps(forbidden, indent=2))

    zip_bundle(BUNDLE_DIR, ZIP_PATH)
    zip_valid = validate_zip(ZIP_PATH)
    elapsed = time.monotonic() - start
    final = {
        "elapsed_seconds": elapsed,
        "bundle_dir": str(BUNDLE_DIR),
        "zip_path": str(ZIP_PATH),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_validated_with_unzip_tq": zip_valid,
        "files_added": len(bundle.inventory) + 1,
        "file_inventory_entries_excluding_self": len(bundle.inventory),
        "files_still_missing": missing,
        "forbidden_payload_files_included": forbidden,
        "raw_videos_or_tribe_outputs_or_model_or_hf_cache_included": bool(forbidden),
        "topk_digest_verification_status": topk_status.get("status"),
        "topk_rows_checked": topk_status.get("rows_checked"),
    }
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if zip_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
