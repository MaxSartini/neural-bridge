"""No-reencode raw cortical representation audit for VEATIC-124.

This script reads the existing TRIBE v2 cache, builds train-only
representations from ``tribe_raw_output.npz`` predictions, and compares them
against the frozen ``cortical_pca64_delta`` baseline without changing the
strict v2 benchmark artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts import run_veatic_event_conditioned_retest as conditioned  # noqa: E402
from backend.scripts import run_veatic_event_spike_retest as retest  # noqa: E402
from backend.scripts import run_veatic_strict_benchmark as strict  # noqa: E402
from backend.scripts import run_veatic_temporal_fairness_benchmark as fairness  # noqa: E402
from backend.scripts import veatic_representation_builders as reps  # noqa: E402


bench = retest.bench

RIDGE_ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LOGISTIC_C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
PRIMARY_TARGETS = (
    ("arousal__future_spike_1_3s", None, 0.05, "binary", "primary"),
    ("arousal__future_spike_1_3s", None, 0.075, "binary", "primary"),
    ("arousal__future_change_p3s_movement", 3, 0.05, "binary", "primary"),
)
SECONDARY_TARGETS = (
    ("arousal__future_change_p2s_movement", 2, 0.05, "binary", "secondary"),
)
CONTINUOUS_TARGETS = (
    ("arousal__future_change_p2s_continuous", 2, None, "continuous", "diagnostic"),
    ("arousal__future_change_p3s_continuous", 3, None, "continuous", "diagnostic"),
)
CONTROL_MODELS = (
    "ar",
    "shuffled",
    "random",
    "timestamp",
    "video_time",
    "majority",
    "label_shuffle_across_videos",
    "label_shuffle_within_video",
    "feature_shuffle_across_videos",
    "feature_shuffle_within_video",
)
FROZEN_BASELINE_CSV = ROOT / "benchmarks" / "veatic" / "veatic_124_alignment_ar_behavior_audit.csv"
CHECKPOINT_SUBDIR = "_checkpoint"
CHECKPOINT_STATE_FILE = "state.json"


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
    root = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT") or local_env_value("NEURAL_BRIDGE_EXTERNAL_ROOT")
    return Path(root).expanduser() if root else ROOT / "external_assets"


def default_cache_dir() -> Path:
    return external_root() / "benchmarks" / "veatic" / "tribe_cache"


def default_external_output_dir() -> Path:
    return external_root() / "outputs" / f"veatic_124_raw_representation_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (ROOT / path).resolve()


def json_safe(value: Any) -> Any:
    return bench.json_safe(value)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def idx_digest(idx: np.ndarray) -> str:
    arr = np.ascontiguousarray(idx, dtype=np.int64)
    return hashlib.blake2b(arr.view(np.uint8), digest_size=12).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_state_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_SUBDIR / CHECKPOINT_STATE_FILE


def safe_cache_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def split_fit_cache_dir(output_dir: Path, scope: str, split: str) -> Path:
    return output_dir / CHECKPOINT_SUBDIR / "fit_cache" / f"{safe_cache_label(scope)}__{safe_cache_label(split)}"


def job_key(
    *,
    scope: str,
    split: str,
    candidate: str,
    head: str,
    target: str,
    threshold: float | None,
    task_type: str,
) -> str:
    return json.dumps(
        [scope, split, candidate, head, target, threshold, task_type],
        separators=(",", ":"),
    )


def job_seed(base_seed: int, key: str) -> int:
    digest = hashlib.blake2b(f"{int(base_seed)}:{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def checkpoint_config(args: argparse.Namespace, mode: str, candidate_names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "veatic_raw_representation_checkpoint_config_v1",
        "mode": mode,
        "seed": args.seed,
        "pca_backend": args.pca_backend,
        "ridge_backend": args.ridge_backend,
        "heads": list(args.heads),
        "candidate_names": list(candidate_names),
        "targets": [
            {
                "target": target,
                "horizon": horizon,
                "threshold": threshold,
                "task_type": task_type,
                "target_tier": target_tier,
            }
            for target, horizon, threshold, task_type, target_tier in selected_targets(mode)
        ],
        "grouped_folds": args.grouped_folds,
        "skip_sensitivity": bool(args.skip_sensitivity),
    }


def empty_checkpoint_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "veatic_raw_representation_checkpoint_state_v1",
        "config": config,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": None,
        "finalized": False,
        "job_status": {},
        "result_rows": [],
        "matched_rows": [],
        "stability_rows": [],
        "skipped_rows": [],
    }


def load_checkpoint_state(output_dir: Path) -> dict[str, Any] | None:
    path = checkpoint_state_path(output_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read checkpoint state at {path}: {exc}") from exc


def save_checkpoint_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(checkpoint_state_path(output_dir), state)


def validate_checkpoint_config(state: dict[str, Any], config: dict[str, Any]) -> None:
    previous = state.get("config", {})
    if previous != config:
        raise RuntimeError(
            "Checkpoint config does not match this run. Use the same mode, heads, "
            "fold count, seed, PCA/ridge backend, and sensitivity setting, or choose "
            "a fresh --output-dir."
        )


def pending_job_specs(
    *,
    completed_jobs: set[str],
    scope: str,
    split: str,
    candidate: str,
    heads: list[str],
    mode: str,
) -> list[tuple[str, int | None, float | None, str, str, str]]:
    pending = []
    for target, horizon, threshold, task_type, target_tier in selected_targets(mode):
        for head in heads:
            if task_type == "continuous" and head != "ridge_score":
                continue
            key = job_key(
                scope=scope,
                split=split,
                candidate=candidate,
                head=head,
                target=target,
                threshold=threshold,
                task_type=task_type,
            )
            if key not in completed_jobs:
                pending.append((target, horizon, threshold, task_type, target_tier, head))
    return pending


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = retest.rows_by_video(rows)
    return {str(key): value for key, value in grouped.items()}


def np_array_summary(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    return json.dumps(arr.astype(int).tolist() if arr.dtype.kind in {"b", "i", "u"} else arr.tolist())


def inventory_cache(
    manifest_rows: list[dict[str, Any]],
    report_path: Path,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    grouped = rows_by_video(manifest_rows)
    report = read_json_file(report_path)
    complete_video_ids = [str(item) for item in report.get("complete_video_ids", sorted(grouped, key=int))]
    rows: list[dict[str, Any]] = []
    alignment_counts: Counter[str] = Counter()
    missing_raw = 0
    prediction_missing = 0
    nonfinite_total = 0
    suspicious_videos: list[str] = []
    modality_counts: Counter[str] = Counter()

    for video_id in complete_video_ids:
        video_rows = grouped.get(video_id, [])
        expected_rows = len(video_rows)
        video_dir = cache_dir / video_id
        raw_path = video_dir / "tribe_raw_output.npz"
        status = read_json_file(video_dir / "cache_status.json")
        summary = read_json_file(video_dir / "tribe_summary.json")
        record: dict[str, Any] = {
            "video_id": video_id,
            "expected_manifest_rows": expected_rows,
            "raw_path": str(raw_path),
            "raw_exists": raw_path.exists(),
            "predictions_exists": False,
            "prediction_rows": None,
            "prediction_width": None,
            "prediction_dtype": None,
            "finite_count": None,
            "nonfinite_count": None,
            "row_alignment": "missing_raw",
            "requires_resampling": None,
            "suspicious_or_resampled": False,
            "is_video_83": video_id == "83",
            "modality_missing_flags": None,
            "feature_modality_present_flags": None,
            "present_modalities": None,
            "missing_modalities": None,
            "cache_status_complete": status.get("complete"),
            "cache_status_error": status.get("error"),
            "event_quality": json.dumps(summary.get("event_quality", {}), sort_keys=True),
            "segment_quality": json.dumps(summary.get("segment_quality", {}), sort_keys=True),
        }
        if not raw_path.exists():
            missing_raw += 1
            rows.append(record)
            suspicious_videos.append(video_id)
            continue
        try:
            with np.load(raw_path) as bundle:
                files = set(bundle.files)
                record["npz_keys"] = ",".join(sorted(files))
                if "predictions" not in files:
                    prediction_missing += 1
                    record["row_alignment"] = "missing_predictions"
                    rows.append(record)
                    suspicious_videos.append(video_id)
                    continue
                predictions = np.asarray(bundle["predictions"])
                finite_mask = np.isfinite(predictions)
                nonfinite = int(predictions.size - int(np.sum(finite_mask)))
                nonfinite_total += nonfinite
                record.update(
                    {
                        "predictions_exists": True,
                        "prediction_rows": int(predictions.shape[0]),
                        "prediction_width": int(predictions.shape[1]) if predictions.ndim >= 2 else None,
                        "prediction_dtype": str(predictions.dtype),
                        "finite_count": int(np.sum(finite_mask)),
                        "nonfinite_count": nonfinite,
                    }
                )
                if "modality_missing_flags" in files:
                    record["modality_missing_flags"] = np_array_summary(bundle["modality_missing_flags"])
                if "feature_modality_present_flags" in files:
                    record["feature_modality_present_flags"] = np_array_summary(bundle["feature_modality_present_flags"])
        except Exception as exc:  # noqa: BLE001 - inventory should report corrupt cache rows.
            record["row_alignment"] = "load_error"
            record["load_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(record)
            suspicious_videos.append(video_id)
            continue

        flags = strict.read_modality_flags(raw_path)
        quality = strict.read_event_quality(video_dir / "tribe_summary.json")
        if flags is None and quality:
            flags = (
                bool(quality.get("missing_text", True)),
                bool(quality.get("missing_audio", True)),
                bool(quality.get("missing_video", True)),
            )
        if flags is not None:
            present = [name for name, missing in zip(strict.MODALITY_ORDER, flags) if not missing]
            missing = [name for name, is_missing in zip(strict.MODALITY_ORDER, flags) if is_missing]
            record["present_modalities"] = "+".join(present) if present else "none"
            record["missing_modalities"] = "+".join(missing) if missing else "none"
            modality_counts[f"{record['present_modalities']}|missing={record['missing_modalities']}"] += 1

        if record["prediction_rows"] == expected_rows:
            alignment = "exact"
            requires_resampling = False
        else:
            alignment = "linear_resampled"
            requires_resampling = True
        record["row_alignment"] = alignment
        record["requires_resampling"] = requires_resampling
        record["suspicious_or_resampled"] = bool(requires_resampling or nonfinite or video_id == "83")
        alignment_counts[alignment] += 1
        if record["suspicious_or_resampled"]:
            suspicious_videos.append(video_id)
        rows.append(record)

    summary = {
        "schema_version": "veatic_raw_cache_inventory_v1",
        "cache_dir": str(cache_dir),
        "manifest_rows": len(manifest_rows),
        "video_count": len(complete_video_ids),
        "raw_exists_count": sum(1 for row in rows if row.get("raw_exists")),
        "missing_raw_count": missing_raw,
        "missing_predictions_count": prediction_missing,
        "alignment_counts": dict(alignment_counts),
        "nonfinite_total": nonfinite_total,
        "suspicious_video_ids": sorted(set(suspicious_videos), key=lambda item: int(item) if item.isdigit() else item),
        "modality_counts": dict(modality_counts),
        "sensitivity_policy": "summaries include all_videos; exclude_video_83 is opt-in with --with-sensitivity",
    }
    report_lines = [
        "# VEATIC-124 Raw Cache Inventory",
        "",
        f"- Cache dir: `{cache_dir}`",
        f"- Videos audited: {summary['video_count']}",
        f"- Raw outputs present: {summary['raw_exists_count']}",
        f"- Missing raw outputs: {summary['missing_raw_count']}",
        f"- Alignment counts: `{summary['alignment_counts']}`",
        f"- Nonfinite values: {summary['nonfinite_total']}",
        f"- Suspicious/resampled videos: `{','.join(summary['suspicious_video_ids']) or 'none'}`",
        "",
        "Video `83` is sensitivity-flagged in the inventory; exclude-video reruns are opt-in.",
    ]
    return rows, summary, "\n".join(report_lines) + "\n"


def load_frozen_reference_baselines(path: Path = FROZEN_BASELINE_CSV) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("feature_mode") != "cortical_pca64_delta":
                continue
            if row.get("target") not in {"arousal__future_spike_1_3s", "arousal__future_change_p3s_movement"}:
                continue
            rows.append(
                {
                    "source_file": str(path),
                    "feature_mode": row.get("feature_mode"),
                    "split": row.get("split"),
                    "target": row.get("target"),
                    "threshold": float(row["threshold"]) if row.get("threshold") not in {None, ""} else None,
                    "real_pr_auc": float(row["real_pr_auc"]) if row.get("real_pr_auc") else None,
                    "ar_pr_auc": float(row["ar_pr_auc"]) if row.get("ar_pr_auc") else None,
                    "event_count": float(row["event_count"]) if row.get("event_count") else None,
                    "n": int(float(row["n"])) if row.get("n") else None,
                    "reference_role": "frozen_v2_baseline_do_not_mutate",
                }
            )
    return rows


def build_base_feature_sets(ctx: retest.RetestContext) -> dict[str, np.ndarray]:
    blocks: dict[str, list[np.ndarray]] = defaultdict(list)
    for video_id in ctx.video_ids:
        blocks["cortical_raw"].append(ctx.features_by_video[video_id]["cortical_raw"])
        blocks["cortical_global"].append(ctx.features_by_video[video_id]["cortical_global"])
    return {key: np.concatenate(values, axis=0).astype(np.float32, copy=False) for key, values in blocks.items()}


def selected_targets(mode: str) -> tuple[tuple[str, int | None, float | None, str, str], ...]:
    if mode == "smoke":
        return (PRIMARY_TARGETS[0],)
    if mode == "primary-audit":
        return PRIMARY_TARGETS
    return PRIMARY_TARGETS + SECONDARY_TARGETS + CONTINUOUS_TARGETS


def split_specs(
    rows: list[dict[str, Any]],
    grouped_folds: int,
    mode: str,
) -> list[tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]]:
    specs: list[tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]] = []
    train_rows, test_rows, _gap = retest.fixed_rows(rows, "blocked_temporal_gap")
    specs.append(("blocked", [], train_rows, test_rows))
    if mode == "smoke":
        return specs
    train_rows, test_rows, _gap = retest.fixed_rows(rows, "official_70_30")
    specs.append(("official", [], train_rows, test_rows))
    specs.extend(fairness.grouped_video_folds(rows, grouped_folds))
    return specs


def target_rows(
    ctx: retest.RetestContext,
    rows: list[dict[str, Any]],
    target: str,
    horizon: int | None,
    threshold: float | None,
    task_type: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if task_type == "continuous":
        if horizon is None:
            raise ValueError(f"horizon required for {target}")
        selected, values = retest.future_change_rows(ctx, rows, int(horizon))
        return selected, values.astype(np.float64)
    if threshold is None:
        raise ValueError(f"threshold required for {target}")
    selected, y, _ = strict.target_rows(ctx, rows, target, horizon, float(threshold))
    return selected, y.astype(np.float64)


def inner_validation_indices(rows: list[dict[str, Any]], seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped = rows_by_video(rows)
    video_ids = sorted(grouped, key=lambda item: int(item) if item.isdigit() else item)
    if len(video_ids) >= 2:
        val_ids = [video_id for offset, video_id in enumerate(video_ids) if offset % 5 == seed % 5]
        if not val_ids:
            val_ids = [video_ids[seed % len(video_ids)]]
        val_set = set(val_ids)
        train_idx = []
        val_idx = []
        for index, row in enumerate(rows):
            (val_idx if str(row["video_id"]) in val_set else train_idx).append(index)
        if train_idx and val_idx:
            return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64), val_ids
    n = len(rows)
    if n < 4:
        return np.arange(n, dtype=np.int64), np.asarray([], dtype=np.int64), []
    cut = max(1, int(n * 0.8))
    return np.arange(cut, dtype=np.int64), np.arange(cut, n, dtype=np.int64), []


def standardize_apply(train_x: np.ndarray, apply_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_x = np.asarray(train_x, dtype=np.float32)
    apply_x = np.asarray(apply_x, dtype=np.float32)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    zero_std = std < 1e-6
    std[zero_std] = 1.0
    return (train_x - mean) / std, (apply_x - mean) / std, {"zero_std_columns": int(np.sum(zero_std))}


def ridge_scores(train_x: np.ndarray, train_y: np.ndarray, apply_x: np.ndarray, alpha: float) -> np.ndarray:
    scores, _ = bench.ridge_fit_predict(
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.float64),
        np.asarray(apply_x, dtype=np.float32),
        alpha=float(alpha),
    )
    return scores.astype(np.float64)


def concat32(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype=np.float32) for part in parts], axis=1).astype(np.float32, copy=False)


def pr_auc_or_none(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.int64)
    if y.size == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return None
    return retest.pr_auc(y, scores)


def select_ridge_alpha(train_x: np.ndarray, train_y: np.ndarray, train_rows: list[dict[str, Any]], seed: int) -> tuple[float, dict[str, Any]]:
    inner_train, inner_val, val_ids = inner_validation_indices(train_rows, seed)
    if inner_val.size < 4 or np.sum(train_y[inner_val] == 1) == 0 or np.sum(train_y[inner_val] == 0) == 0:
        return 1.0, {"selected": False, "alpha": 1.0, "reason": "inner_validation_not_feasible"}
    best_alpha = 1.0
    best_score = -math.inf
    scores_by_alpha = {}
    for alpha in RIDGE_ALPHA_GRID:
        try:
            val_scores = ridge_scores(train_x[inner_train], train_y[inner_train], train_x[inner_val], alpha)
            score = pr_auc_or_none(train_y[inner_val], val_scores)
        except Exception as exc:  # noqa: BLE001
            score = None
            scores_by_alpha[str(alpha)] = f"{type(exc).__name__}: {exc}"
            continue
        scores_by_alpha[str(alpha)] = score
        if score is not None and score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, {
        "selected": best_score > -math.inf,
        "alpha": best_alpha,
        "inner_validation_pr_auc": None if best_score == -math.inf else best_score,
        "inner_validation_video_ids": val_ids,
        "scores_by_alpha": scores_by_alpha,
    }


def logistic_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    apply_x: np.ndarray,
    c_value: float,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    train_z, apply_z, _ = standardize_apply(train_x, apply_x)
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=17,
    )
    model.fit(train_z, np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(apply_z)[:, 1].astype(np.float64)


def select_logistic_c(train_x: np.ndarray, train_y: np.ndarray, train_rows: list[dict[str, Any]], seed: int) -> tuple[float, dict[str, Any]]:
    inner_train, inner_val, val_ids = inner_validation_indices(train_rows, seed)
    if (
        inner_val.size < 4
        or len(set(np.asarray(train_y[inner_train], dtype=int).tolist())) < 2
        or len(set(np.asarray(train_y[inner_val], dtype=int).tolist())) < 2
    ):
        return 1.0, {"selected": False, "C": 1.0, "reason": "inner_validation_not_feasible"}
    best_c = 1.0
    best_score = -math.inf
    scores_by_c = {}
    for c_value in LOGISTIC_C_GRID:
        try:
            val_scores = logistic_scores(train_x[inner_train], train_y[inner_train], train_x[inner_val], c_value)
            score = pr_auc_or_none(train_y[inner_val], val_scores)
        except Exception as exc:  # noqa: BLE001
            score = None
            scores_by_c[str(c_value)] = f"{type(exc).__name__}: {exc}"
            continue
        scores_by_c[str(c_value)] = score
        if score is not None and score > best_score:
            best_score = score
            best_c = float(c_value)
    return best_c, {
        "selected": best_score > -math.inf,
        "C": best_c,
        "inner_validation_pr_auc": None if best_score == -math.inf else best_score,
        "inner_validation_video_ids": val_ids,
        "scores_by_C": scores_by_c,
    }


def fit_head_scores(
    head: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_rows: list[dict[str, Any]],
    *,
    task_type: str,
    seed: int,
    force_ridge_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if head == "ridge_score" or task_type == "continuous":
        if force_ridge_alpha is not None:
            alpha, meta = float(force_ridge_alpha), {
                "selected": False,
                "alpha": float(force_ridge_alpha),
                "reason": "frozen_reference_original_alpha",
            }
        else:
            alpha, meta = select_ridge_alpha(train_x, train_y, train_rows, seed) if task_type == "binary" else (1.0, {"selected": False, "alpha": 1.0, "reason": "continuous_default"})
        return (
            ridge_scores(train_x, train_y, train_x, alpha),
            ridge_scores(train_x, train_y, test_x, alpha),
            {"head": "ridge_score", **meta},
        )
    if head == "logistic_l2":
        if len(set(np.asarray(train_y, dtype=int).tolist())) < 2:
            raise RuntimeError("logistic_l2 requires both train classes")
        c_value, meta = select_logistic_c(train_x, train_y, train_rows, seed)
        return (
            logistic_scores(train_x, train_y, train_x, c_value),
            logistic_scores(train_x, train_y, test_x, c_value),
            {"head": "logistic_l2", **meta},
        )
    raise ValueError(f"Unsupported head: {head}")


def continuous_metrics(train_y: np.ndarray, train_scores: np.ndarray, test_y: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    metrics = retest.regression_metrics(test_y.astype(np.float64), test_scores.astype(np.float64))
    zero = np.zeros_like(test_y, dtype=np.float64)
    zero_metrics = retest.regression_metrics(test_y.astype(np.float64), zero)
    metrics.update(
        {
            "zero_mae": zero_metrics["mae"],
            "real_minus_zero_mae": None if metrics["mae"] is None or zero_metrics["mae"] is None else float(zero_metrics["mae"] - metrics["mae"]),
            "event_count": None,
            "positive_class_rate": None,
        }
    )
    return metrics


def evaluate_scores(
    task_type: str,
    train_y: np.ndarray,
    train_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, Any]:
    if task_type == "continuous":
        return continuous_metrics(train_y, train_scores, test_y, test_scores)
    return retest.event_metrics(train_y.astype(np.int64), train_scores, test_y.astype(np.int64), test_scores)


def video_time_matrix(ctx: retest.RetestContext, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return conditioned.video_time_matrix(ctx, train_rows, test_rows)


def metrics_to_row(
    base: dict[str, Any],
    model: str,
    metrics: dict[str, Any],
    head_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = dict(base)
    row["model"] = model
    if head_meta:
        row["selected_hyperparameters"] = json.dumps(json_safe(head_meta), sort_keys=True)
    for key, value in metrics.items():
        row[key] = finite(value) if isinstance(value, (int, float, np.floating)) or value is None else value
    return row


def score_candidate(
    *,
    ctx: retest.RetestContext,
    candidate_name: str,
    fitted: reps.FittedRepresentation,
    head: str,
    scope: str,
    split_label: str,
    held: list[str],
    target: str,
    horizon: int | None,
    threshold: float | None,
    task_type: str,
    target_tier: str,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    seed: int,
    rng: np.random.Generator,
    context_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_idx = bench.row_indices(ctx.accepted_rows, train_selected)
    test_idx = bench.row_indices(ctx.accepted_rows, test_selected)
    train_rep = fitted.transform(train_selected, train_idx)
    test_rep = fitted.transform(test_selected, test_idx)
    if train_rep.values.shape[0] < 8 or test_rep.values.shape[0] < 4:
        return [], [], []
    y_train = train_y[train_rep.keep_mask]
    y_test = test_y[test_rep.keep_mask]
    if task_type == "binary" and (np.sum(y_train == 1) == 0 or np.sum(y_test == 1) == 0):
        return [], [], []
    train_rows = train_rep.rows
    test_rows = test_rep.rows
    context_key = (idx_digest(train_rep.idx), idx_digest(test_rep.idx))
    cached_context = context_cache.get(context_key)
    if cached_context is None:
        train_ar = bench.autoregressive_features(ctx.accepted_rows, train_rows, "arousal", include_current=True)
        test_ar = bench.autoregressive_features(ctx.accepted_rows, test_rows, "arousal", include_current=True)
        train_time = bench.time_features(train_rows)
        test_time = bench.time_features(test_rows)
        train_video_time, test_video_time = video_time_matrix(ctx, train_rows, test_rows)
        context_cache[context_key] = (
            train_ar,
            test_ar,
            train_time,
            test_time,
            train_video_time,
            test_video_time,
        )
    else:
        train_ar, test_ar, train_time, test_time, train_video_time, test_video_time = cached_context

    model_designs: dict[str, tuple[np.ndarray | None, np.ndarray | None, np.ndarray]] = {}
    model_designs["ar"] = (train_ar, test_ar, y_train)
    model_designs["real"] = (concat32([train_ar, train_rep.values]), concat32([test_ar, test_rep.values]), y_train)
    shuffled_train = rng.permutation(train_rep.values)
    shuffled_test = rng.permutation(test_rep.values)
    model_designs["shuffled"] = (concat32([train_ar, shuffled_train]), concat32([test_ar, shuffled_test]), y_train)
    random_train = rng.normal(size=train_rep.values.shape).astype(np.float32)
    random_test = rng.normal(size=test_rep.values.shape).astype(np.float32)
    model_designs["random"] = (concat32([train_ar, random_train]), concat32([test_ar, random_test]), y_train)
    model_designs["timestamp"] = (train_time, test_time, y_train)
    model_designs["video_time"] = (train_video_time, test_video_time, y_train)
    model_designs["label_shuffle_across_videos"] = (model_designs["real"][0], model_designs["real"][1], rng.permutation(y_train))
    model_designs["label_shuffle_within_video"] = (model_designs["real"][0], model_designs["real"][1], retest.shuffle_by_video(train_rows, y_train, rng))
    feature_shuffle_train = retest.shuffle_by_video(train_rows, train_rep.values, rng)
    feature_shuffle_test = retest.shuffle_by_video(test_rows, test_rep.values, rng)
    model_designs["feature_shuffle_within_video"] = (
        concat32([train_ar, feature_shuffle_train]),
        concat32([test_ar, feature_shuffle_test]),
        y_train,
    )
    model_designs["feature_shuffle_across_videos"] = (
        concat32([train_ar, rng.permutation(train_rep.values)]),
        concat32([test_ar, rng.permutation(test_rep.values)]),
        y_train,
    )

    base = {
        "scope": scope,
        "candidate": candidate_name,
        "family": fitted.metadata().get("family"),
        "head": head,
        "split": split_label,
        "held_out_video_ids": ",".join(held),
        "target": target,
        "threshold": threshold,
        "task_type": task_type,
        "target_tier": target_tier,
        "n_train": int(y_train.size),
        "n_test": int(y_test.size),
        "feature_width": int(train_rep.values.shape[1]),
        "train_videos": int(len({str(row["video_id"]) for row in train_rows})),
        "test_videos": int(len({str(row["video_id"]) for row in test_rows})),
    }
    rows: list[dict[str, Any]] = []
    model_metrics: dict[str, dict[str, Any]] = {}
    frozen_reference = fitted.metadata().get("family") == "frozen_reference"
    for model, (train_x, test_x, model_y_train) in model_designs.items():
        print(
            f"[INFO] scoring scope={scope} split={split_label} candidate={candidate_name} "
            f"target={target}@{threshold} head={head} model={model} width={train_x.shape[1]}",
            flush=True,
        )
        try:
            train_scores, test_scores, head_meta = fit_head_scores(
                head,
                train_x,  # type: ignore[arg-type]
                model_y_train,
                test_x,  # type: ignore[arg-type]
                train_rows,
                task_type=task_type,
                seed=seed,
                force_ridge_alpha=1.0 if frozen_reference and head == "ridge_score" else None,
            )
            eval_train_y = model_y_train if model.startswith("label_shuffle") else y_train
            metrics = evaluate_scores(task_type, eval_train_y, train_scores, y_test, test_scores)
        except Exception as exc:  # noqa: BLE001
            metrics = {"error": f"{type(exc).__name__}: {exc}"}
            head_meta = {"head": head, "failed": True}
        model_metrics[model] = metrics
        rows.append(metrics_to_row(base, model, metrics, head_meta))

    majority = retest.majority_metrics(y_train.astype(np.int64), y_test.astype(np.int64)) if task_type == "binary" else continuous_metrics(y_train, np.zeros_like(y_train), y_test, np.full_like(y_test, np.mean(y_train)))
    model_metrics["majority"] = majority
    rows.append(metrics_to_row(base, "majority", majority, {"head": "majority"}))

    real = model_metrics.get("real", {})
    for row in rows:
        if row["model"] == "real":
            for control in ("ar", "shuffled", "random", "timestamp", "video_time"):
                control_metrics = model_metrics.get(control, {})
                for metric in ("pr_auc", "f1", "balanced_accuracy", "precision", "recall", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall"):
                    left = real.get(metric)
                    right = control_metrics.get(metric)
                    row[f"real_minus_{control}_{metric}"] = None if left is None or right is None else float(left) - float(right)
            if task_type == "continuous":
                for control in ("ar", "shuffled", "random", "timestamp", "video_time"):
                    left = real.get("mae")
                    right = model_metrics.get(control, {}).get("mae")
                    row[f"real_minus_{control}_mae"] = None if left is None or right is None else float(right) - float(left)

    stability_rows = []
    meta = fitted.metadata()
    if meta.get("compression_type") == "train_only_topk_vertices":
        stability_rows.append(
            {
                **base,
                "selected_vertices_digest": meta.get("selected_vertices_digest"),
                "selected_vertex_count": meta.get("selected_vertex_count"),
                "score_method": meta.get("score_method"),
            }
        )

    matched_rows = matched_context_rows(
        ctx=ctx,
        candidate_name=candidate_name,
        fitted=fitted,
        base=base,
        head=head,
        task_type=task_type,
        y_train=y_train,
        y_test=y_test,
        train_rows=train_rows,
        test_rows=test_rows,
        train_ar=train_ar,
        test_ar=test_ar,
        train_idx=train_rep.idx,
        test_idx=test_rep.idx,
        context_metrics=real,
        seed=seed,
    )
    return rows, matched_rows, stability_rows


def matched_context_rows(
    *,
    ctx: retest.RetestContext,
    candidate_name: str,
    fitted: reps.FittedRepresentation,
    base: dict[str, Any],
    head: str,
    task_type: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_ar: np.ndarray,
    test_ar: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    context_metrics: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    if not isinstance(fitted, reps.MatrixBackedRepresentation):
        return []
    current = fitted.auxiliary_matrices.get("matched_current")
    if current is None:
        return []
    current_train = current[train_idx]
    current_test = current[test_idx]
    try:
        train_scores, test_scores, head_meta = fit_head_scores(
            head,
            concat32([train_ar, current_train]),
            y_train,
            concat32([test_ar, current_test]),
            train_rows,
            task_type=task_type,
            seed=seed,
            force_ridge_alpha=1.0 if fitted.metadata().get("family") == "frozen_reference" and head == "ridge_score" else None,
        )
        current_metrics = evaluate_scores(task_type, y_train, train_scores, y_test, test_scores)
    except Exception as exc:  # noqa: BLE001
        current_metrics = {"error": f"{type(exc).__name__}: {exc}"}
        head_meta = {"head": head, "failed": True}
    metric = "pr_auc" if task_type == "binary" else "mae"
    context_value = context_metrics.get(metric)
    current_value = current_metrics.get(metric)
    if task_type == "binary":
        gain = None if context_value is None or current_value is None else float(context_value) - float(current_value)
    else:
        gain = None if context_value is None or current_value is None else float(current_value) - float(context_value)
    row = {
        **base,
        "matched_context_candidate": candidate_name,
        "matched_current_feature_width": int(current_train.shape[1]),
        "matched_metric": metric,
        "context_value": finite(context_value),
        "current_only_value": finite(current_value),
        "matched_real_gain": gain,
        "selected_hyperparameters": json.dumps(json_safe(head_meta), sort_keys=True),
    }
    return [row]


def summarize_grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not str(row.get("split", "")).startswith("grouped_"):
            continue
        grouped[(row.get("scope"), row.get("candidate"), row.get("head"), row.get("target"), row.get("threshold"), row.get("model"))].append(row)
    out = []
    for key, values in grouped.items():
        record = {
            "scope": key[0],
            "candidate": key[1],
            "head": key[2],
            "target": key[3],
            "threshold": key[4],
            "model": key[5],
            "fold_count": len(values),
            "total_event_count": sum(float(row.get("event_count") or 0.0) for row in values),
            "total_n_test": sum(int(row.get("n_test") or 0) for row in values),
        }
        for metric in ("pr_auc", "f1", "balanced_accuracy", "precision", "recall", "top_1pct_recall", "top_5pct_recall", "top_10pct_recall"):
            arr = np.asarray([row.get(metric) for row in values if row.get(metric) is not None], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None
            record[f"{metric}_std"] = float(np.std(arr)) if arr.size else None
        for control in ("ar", "shuffled", "random", "timestamp", "video_time"):
            arr = np.asarray([row.get(f"real_minus_{control}_pr_auc") for row in values if row.get(f"real_minus_{control}_pr_auc") is not None], dtype=np.float64)
            record[f"real_minus_{control}_pr_auc_mean"] = float(np.mean(arr)) if arr.size else None
            record[f"real_minus_{control}_pr_auc_positive_folds"] = int(np.sum(arr > 0)) if arr.size else 0
        out.append(record)
    return sorted(out, key=lambda row: (str(row["scope"]), str(row["target"]), float(row["threshold"] or 0.0), str(row["candidate"]), str(row["model"])))


def leaderboard(rows: list[dict[str, Any]], grouped_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = [
        row for row in rows
        if row.get("model") == "real" and row.get("scope") == "all_videos" and row.get("task_type") == "binary"
    ]
    grouped_real = [
        row for row in grouped_rows
        if row.get("model") == "real" and row.get("scope") == "all_videos"
    ]
    out = []
    for row in fixed:
        out.append(
            {
                "scope": row.get("scope"),
                "candidate": row.get("candidate"),
                "head": row.get("head"),
                "target": row.get("target"),
                "threshold": row.get("threshold"),
                "split": row.get("split"),
                "pr_auc": row.get("pr_auc"),
                "f1": row.get("f1"),
                "real_minus_ar_pr_auc": row.get("real_minus_ar_pr_auc"),
                "real_minus_shuffled_pr_auc": row.get("real_minus_shuffled_pr_auc"),
                "real_minus_random_pr_auc": row.get("real_minus_random_pr_auc"),
                "real_minus_timestamp_pr_auc": row.get("real_minus_timestamp_pr_auc"),
                "real_minus_video_time_pr_auc": row.get("real_minus_video_time_pr_auc"),
                "feature_width": row.get("feature_width"),
            }
        )
    for row in grouped_real:
        out.append(
            {
                "scope": row.get("scope"),
                "candidate": row.get("candidate"),
                "head": row.get("head"),
                "target": row.get("target"),
                "threshold": row.get("threshold"),
                "split": "grouped_video_mean",
                "pr_auc": row.get("pr_auc_mean"),
                "f1": row.get("f1_mean"),
                "real_minus_ar_pr_auc": row.get("real_minus_ar_pr_auc_mean"),
                "real_minus_shuffled_pr_auc": row.get("real_minus_shuffled_pr_auc_mean"),
                "real_minus_random_pr_auc": row.get("real_minus_random_pr_auc_mean"),
                "real_minus_timestamp_pr_auc": row.get("real_minus_timestamp_pr_auc_mean"),
                "real_minus_video_time_pr_auc": row.get("real_minus_video_time_pr_auc_mean"),
                "feature_width": None,
            }
        )
    return sorted(out, key=lambda row: (row["pr_auc"] is None, -(row["pr_auc"] or -1.0), str(row["candidate"])))


def promotion_summary(grouped_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_target: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in grouped_rows:
        if row.get("scope") == "all_videos" and row.get("candidate") == "cortical_pca64_delta" and row.get("model") == "real":
            baseline_by_target[(row.get("head"), row.get("target"), row.get("threshold"))] = row
    candidates: list[dict[str, Any]] = []
    for row in grouped_rows:
        if row.get("scope") != "all_videos" or row.get("model") != "real":
            continue
        candidate = row.get("candidate")
        key = (row.get("head"), row.get("target"), row.get("threshold"))
        baseline = baseline_by_target.get(key)
        baseline_pr = baseline.get("pr_auc_mean") if baseline else None
        gain = None if baseline_pr is None or row.get("pr_auc_mean") is None else float(row["pr_auc_mean"]) - float(baseline_pr)
        control_pass = all(
            (row.get(f"real_minus_{control}_pr_auc_mean") is not None and row.get(f"real_minus_{control}_pr_auc_mean") > 0)
            for control in ("ar", "shuffled", "random", "timestamp", "video_time")
        )
        stable_folds = int(row.get("real_minus_ar_pr_auc_positive_folds") or 0)
        if candidate == "cortical_pca64_delta":
            category = "frozen_baseline"
        elif gain is not None and gain >= 0.01 and control_pass and stable_folds >= 3:
            category = "promoted_candidate"
        elif gain is not None and gain > 0 and stable_folds >= 2:
            category = "promising_exploratory"
        elif control_pass:
            category = "valid_but_weaker"
        else:
            category = "overfit_or_nuisance_warning"
        candidates.append(
            {
                "candidate": candidate,
                "head": row.get("head"),
                "target": row.get("target"),
                "threshold": row.get("threshold"),
                "grouped_pr_auc_mean": row.get("pr_auc_mean"),
                "frozen_pca64_delta_pr_auc_mean": baseline_pr,
                "gain_vs_frozen_pca64_delta": gain,
                "control_pass": control_pass,
                "stable_positive_folds_vs_ar": stable_folds,
                "category": category,
                "promotion_rule": "requires >=0.01 grouped PR-AUC or F1/top-k gain plus controls and stability",
            }
        )
    return {
        "schema_version": "veatic_raw_representation_promotion_v1",
        "baseline": "cortical_pca64_delta",
        "baseline_policy": "Grouped promotion compares against same-run cortical_pca64_delta; frozen fixed-split reference values are reported separately.",
        "candidates": sorted(candidates, key=lambda item: (item["category"] != "promoted_candidate", item["gain_vs_frozen_pca64_delta"] is None, -(item["gain_vs_frozen_pca64_delta"] or -999.0))),
    }


def leakage_audit(candidate_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in candidate_rows:
        if row.get("uses_future_features"):
            failures.append({"candidate": row["name"], "failure": "uses_future_features"})
    grouped_failures = []
    for row in result_rows:
        split = str(row.get("split", ""))
        if not split.startswith("grouped_"):
            continue
        held = {item for item in str(row.get("held_out_video_ids", "")).split(",") if item}
        if not held:
            grouped_failures.append({"candidate": row.get("candidate"), "split": split, "failure": "missing_held_out_video_ids"})
    return {
        "schema_version": "veatic_raw_representation_leakage_audit_v1",
        "fit_scope": "train_rows_only",
        "test_label_tuning": False,
        "future_feature_rows_in_primary": False,
        "positive_only_pr_auc_headline": False,
        "failures": failures + grouped_failures,
        "status": "pass" if not failures and not grouped_failures else "fail",
    }


def write_report(
    path: Path,
    *,
    mode: str,
    inventory_summary: dict[str, Any],
    candidates: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
    promotion: dict[str, Any],
    leakage: dict[str, Any],
    frozen_references: list[dict[str, Any]],
    outputs: dict[str, Path],
) -> None:
    top_rows = leaderboard_rows[:12]
    promoted = [row for row in promotion.get("candidates", []) if row.get("category") == "promoted_candidate"]
    lines = [
        "# VEATIC-124 Raw Representation Audit",
        "",
        "## Scope",
        "",
        "- No video re-encoding was performed.",
        "- All representations use cached `tribe_raw_output.npz` cortical predictions.",
        "- Frozen `cortical_pca64_delta` is retained as the comparison baseline.",
        "- Current 0s alignment remains primary; causal windows are row-matched when they drop history rows.",
        "",
        "## Run Summary",
        "",
        f"- Mode: `{mode}`",
        f"- Cache videos audited: {inventory_summary.get('video_count')}",
        f"- Raw outputs present: {inventory_summary.get('raw_exists_count')}",
        f"- Suspicious/resampled videos: `{','.join(inventory_summary.get('suspicious_video_ids', [])) or 'none'}`",
        f"- Candidate count: {len(candidates)}",
        f"- Leakage audit status: `{leakage.get('status')}`",
        "",
        "## Frozen Reference Baseline",
        "",
    ]
    if frozen_references:
        lines.extend(
            [
                "| Split | Target | Thr | Frozen PR-AUC | AR PR-AUC | N | Events |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in frozen_references[:8]:
            lines.append(
                f"| `{row.get('split')}` | `{row.get('target')}` | {fmt(row.get('threshold'))} | "
                f"{fmt(row.get('real_pr_auc'))} | {fmt(row.get('ar_pr_auc'))} | "
                f"{row.get('n') or 'NA'} | {fmt(row.get('event_count'))} |"
            )
    else:
        lines.append("- Frozen reference CSV was not found; same-run `cortical_pca64_delta` rows are still included.")
    lines.extend(
        [
            "",
        "## Promotion Verdict",
        "",
        ]
    )
    if promoted:
        for row in promoted:
            lines.append(
                f"- `{row['candidate']}` on `{row['target']}` @ `{row['threshold']}`: "
                f"grouped PR-AUC gain `{row['gain_vs_frozen_pca64_delta']:.4f}`."
            )
    else:
        lines.append("- No candidate met the strict promotion rule in this run.")
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Candidate | Split | Target | Thr | Head | PR-AUC | F1 | vs AR | vs shuffled | vs random |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| `{row.get('candidate')}` | `{row.get('split')}` | `{row.get('target')}` | "
            f"{fmt(row.get('threshold'))} | `{row.get('head')}` | {fmt(row.get('pr_auc'))} | "
            f"{fmt(row.get('f1'))} | {fmt(row.get('real_minus_ar_pr_auc'))} | "
            f"{fmt(row.get('real_minus_shuffled_pr_auc'))} | {fmt(row.get('real_minus_random_pr_auc'))} |"
        )
    lines.extend(["", "## Output Files", ""])
    for label, output in outputs.items():
        lines.append(f"- {label}: `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "NA"
    return f"{val:.4f}"


def copy_lightweight_outputs(external_dir: Path, tracked_dir: Path, outputs: dict[str, Path]) -> dict[str, Path]:
    tracked_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for label in ("report_md", "leaderboard_csv", "promotion_json", "leakage_json"):
        src = outputs[label]
        dst = tracked_dir / src.name
        shutil.copy2(src, dst)
        copied[label] = dst
    return copied


def audit_plan(args: argparse.Namespace) -> dict[str, Any]:
    mode = selected_mode(args)
    candidates = reps.describe_candidates(reps.candidate_names_for_mode(mode), seed=args.seed)
    return {
        "schema_version": "veatic_raw_representation_audit_plan_v1",
        "mode": mode,
        "no_reencode": True,
        "cache_dir": str(Path(args.cache_dir).expanduser()),
        "output_dir": str(Path(args.output_dir).expanduser()) if args.output_dir else str(default_external_output_dir()),
        "tracked_output_dir": str(Path(args.tracked_output_dir).expanduser()) if args.tracked_output_dir else "outputs/<run-name>",
        "candidates": candidates,
        "targets": [
            {
                "target": target,
                "horizon": horizon,
                "threshold": threshold,
                "task_type": task_type,
                "tier": tier,
            }
            for target, horizon, threshold, task_type, tier in selected_targets(mode)
        ],
        "splits": ["blocked_temporal_gap"] if mode == "smoke" else ["blocked_temporal_gap", "official_70_30", f"grouped_video_{args.grouped_folds}_fold"],
        "heads": args.heads,
        "controls": list(CONTROL_MODELS),
        "sensitivity": ["all_videos"] + ([] if args.skip_sensitivity else ["exclude_video_83"]),
        "leakage_rules": [
            "fit representations on train rows only",
            "use inner train-video validation for hyperparameter selection",
            "keep grouped train/test videos disjoint",
            "do not use future cortical rows for causal primary representations",
            "do not tune thresholds on test or filtered test subsets",
        ],
    }


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_preflight(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    candidate_names = reps.candidate_names_for_mode(mode)
    required_modules = {
        "numpy": "array and matrix operations",
        "scipy": "PCA eigensolve fallback and metrics support",
        "sklearn": "PLS/logistic heads and dependency contract",
    }
    if "roi_parcel_features" in candidate_names:
        required_modules["nibabel"] = "surface atlas IO for roi_parcel_features"
        required_modules["nilearn"] = "Destrieux atlas loading for roi_parcel_features"
    missing = [
        {"module": module, "reason": reason}
        for module, reason in required_modules.items()
        if not module_available(module)
    ]
    checks: dict[str, Any] = {
        "schema_version": "veatic_raw_representation_preflight_v1",
        "mode": mode,
        "python_executable": sys.executable,
        "required_modules": required_modules,
        "missing_modules": missing,
        "pca_backend": args.pca_backend,
        "ridge_backend": args.ridge_backend,
        "roi_atlas_loaded": None,
        "mps_available": None,
        "status": "pass",
    }
    if args.pca_backend in {"mps_gram", "mps_power"}:
        try:
            import torch

            checks["mps_available"] = bool(torch.backends.mps.is_available())
        except Exception as exc:  # noqa: BLE001
            checks["mps_available"] = False
            missing.append({"module": "torch", "reason": f"MPS PCA backend requested but torch import failed: {type(exc).__name__}: {exc}"})
        if checks["mps_available"] is False:
            missing.append({"module": "torch.backends.mps", "reason": f"{args.pca_backend} requires Apple MPS; use --pca-backend cpu_svd or auto if unavailable"})
    if "roi_parcel_features" in candidate_names and not missing:
        try:
            from backend.app.services.cortical_roi_mapper import CorticalRoiMapper

            checks["roi_atlas_loaded"] = bool(CorticalRoiMapper().load_destrieux_atlas())
            if not checks["roi_atlas_loaded"]:
                missing.append({"module": "nilearn.datasets.fetch_atlas_surf_destrieux", "reason": "ROI candidate selected but Destrieux atlas did not load"})
        except Exception as exc:  # noqa: BLE001
            checks["roi_atlas_loaded"] = False
            missing.append({"module": "nilearn/nibabel", "reason": f"ROI atlas load failed: {type(exc).__name__}: {exc}"})
    if missing:
        checks["status"] = "fail"
        checks["missing_modules"] = missing
    return checks


def selected_mode(args: argparse.Namespace) -> str:
    if args.full_audit:
        return "full-audit"
    if args.primary_audit:
        return "primary-audit"
    return "smoke"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    start = time.monotonic()
    mode = selected_mode(args)
    preflight = dependency_preflight(args, mode)
    if preflight["status"] != "pass":
        raise RuntimeError(
            "Audit preflight failed before heavy scoring: "
            + json.dumps(preflight["missing_modules"], indent=2)
        )
    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend
    manifest_path = resolve_repo_path(args.manifest)
    report_path = resolve_repo_path(args.report)
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_external_output_dir()
    tracked_dir = Path(args.tracked_output_dir).expanduser().resolve() if args.tracked_output_dir else ROOT / "outputs" / output_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = bench.load_manifest(manifest_path)
    inventory_rows, inventory_summary, inventory_report = inventory_cache(manifest_rows, report_path, cache_dir)
    frozen_references = load_frozen_reference_baselines()
    ctx = retest.RetestContext(manifest_path, report_path, cache_dir)
    base_feature_sets = build_base_feature_sets(ctx)
    candidate_names = reps.candidate_names_for_mode(mode)
    candidate_descriptions = reps.describe_candidates(candidate_names, seed=args.seed)
    config = checkpoint_config(args, mode, candidate_names)
    state = load_checkpoint_state(output_dir)
    if state is not None:
        validate_checkpoint_config(state, config)
        if not args.resume and not state.get("finalized"):
            raise RuntimeError(
                f"Checkpoint already exists in {output_dir}. Pass --resume to continue it "
                "or choose a fresh --output-dir."
            )
    else:
        state = empty_checkpoint_state(config)
        save_checkpoint_state(output_dir, state)

    scopes = [("all_videos", ctx.accepted_rows)]
    if not args.skip_sensitivity:
        scopes.append(("exclude_video_83", [row for row in ctx.accepted_rows if str(row["video_id"]) != "83"]))

    result_rows: list[dict[str, Any]] = list(state.get("result_rows", []))
    matched_rows: list[dict[str, Any]] = list(state.get("matched_rows", []))
    stability_rows: list[dict[str, Any]] = list(state.get("stability_rows", []))
    skipped_rows: list[dict[str, Any]] = list(state.get("skipped_rows", []))
    job_status: dict[str, str] = dict(state.get("job_status", {}))
    completed_jobs = {key for key, status in job_status.items() if status in {"complete", "skipped"}}

    for scope_label, scope_rows in scopes:
        splits = split_specs(scope_rows, args.grouped_folds, mode)
        for split_label, held, train_rows, test_rows in splits:
            split_train_idx = bench.row_indices(ctx.accepted_rows, train_rows)
            unsupervised_cache: dict[str, reps.FittedRepresentation] = {}
            split_fit_cache: dict[str, Any] = {}
            fit_cache_context = {
                "fit_cache": split_fit_cache,
                "fit_cache_dir": split_fit_cache_dir(output_dir, scope_label, split_label),
            }
            context_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
            for candidate_name in candidate_names:
                builder = reps.builder_from_name(candidate_name, seed=args.seed)
                candidate_pending = pending_job_specs(
                    completed_jobs=completed_jobs,
                    scope=scope_label,
                    split=split_label,
                    candidate=candidate_name,
                    heads=list(args.heads),
                    mode=mode,
                )
                if not candidate_pending:
                    continue
                if not builder.uses_labels_for_fit:
                    try:
                        unsupervised_cache[candidate_name] = builder.fit(
                            train_rows,
                            split_train_idx,
                            ctx.accepted_rows,
                            base_feature_sets,
                            inner_validation=fit_cache_context,
                        )
                    except Exception as exc:  # noqa: BLE001
                        for target, _horizon, threshold, task_type, _target_tier, head in candidate_pending:
                            key = job_key(
                                scope=scope_label,
                                split=split_label,
                                candidate=candidate_name,
                                head=head,
                                target=target,
                                threshold=threshold,
                                task_type=task_type,
                            )
                            skipped_rows.append({
                                "scope": scope_label,
                                "candidate": candidate_name,
                                "head": head,
                                "split": split_label,
                                "target": target,
                                "threshold": threshold,
                                "status": "skipped",
                                "reason": f"{type(exc).__name__}: {exc}",
                            })
                            job_status[key] = "skipped"
                            completed_jobs.add(key)
                        state["skipped_rows"] = skipped_rows
                        state["job_status"] = job_status
                        save_checkpoint_state(output_dir, state)
            for target, horizon, threshold, task_type, target_tier in selected_targets(mode):
                train_selected, train_y = target_rows(ctx, train_rows, target, horizon, threshold, task_type)
                test_selected, test_y = target_rows(ctx, test_rows, target, horizon, threshold, task_type)
                if not train_selected or not test_selected:
                    continue
                for candidate_name in candidate_names:
                    builder = reps.builder_from_name(candidate_name, seed=args.seed)
                    pending_heads = [
                        head for head in args.heads
                        if not (task_type == "continuous" and head != "ridge_score")
                        and job_key(
                            scope=scope_label,
                            split=split_label,
                            candidate=candidate_name,
                            head=head,
                            target=target,
                            threshold=threshold,
                            task_type=task_type,
                        )
                        not in completed_jobs
                    ]
                    if not pending_heads:
                        continue
                    if builder.uses_labels_for_fit:
                        train_idx = bench.row_indices(ctx.accepted_rows, train_selected)
                        try:
                            fitted = builder.fit(
                                train_selected,
                                train_idx,
                                ctx.accepted_rows,
                                base_feature_sets,
                                y_train=train_y,
                                inner_validation=fit_cache_context,
                            )
                        except Exception as exc:  # noqa: BLE001
                            for head in pending_heads:
                                key = job_key(
                                    scope=scope_label,
                                    split=split_label,
                                    candidate=candidate_name,
                                    head=head,
                                    target=target,
                                    threshold=threshold,
                                    task_type=task_type,
                                )
                                skipped_rows.append({
                                    "scope": scope_label,
                                    "candidate": candidate_name,
                                    "head": head,
                                    "split": split_label,
                                    "target": target,
                                    "threshold": threshold,
                                    "status": "skipped",
                                    "reason": f"{type(exc).__name__}: {exc}",
                                })
                                job_status[key] = "skipped"
                                completed_jobs.add(key)
                            state["skipped_rows"] = skipped_rows
                            state["job_status"] = job_status
                            save_checkpoint_state(output_dir, state)
                            continue
                    else:
                        fitted = unsupervised_cache.get(candidate_name)
                        if fitted is None:
                            continue
                    for head in pending_heads:
                        key = job_key(
                            scope=scope_label,
                            split=split_label,
                            candidate=candidate_name,
                            head=head,
                            target=target,
                            threshold=threshold,
                            task_type=task_type,
                        )
                        local_seed = job_seed(args.seed, key)
                        rows, matched, stability = score_candidate(
                            ctx=ctx,
                            candidate_name=candidate_name,
                            fitted=fitted,
                            head=head,
                            scope=scope_label,
                            split_label=split_label,
                            held=held,
                            target=target,
                            horizon=horizon,
                            threshold=threshold,
                            task_type=task_type,
                            target_tier=target_tier,
                            train_selected=train_selected,
                            train_y=train_y,
                            test_selected=test_selected,
                            test_y=test_y,
                            seed=local_seed,
                            rng=np.random.default_rng(local_seed),
                            context_cache=context_cache,
                        )
                        result_rows.extend(rows)
                        matched_rows.extend(matched)
                        stability_rows.extend(stability)
                        job_status[key] = "complete"
                        completed_jobs.add(key)
                        state["result_rows"] = result_rows
                        state["matched_rows"] = matched_rows
                        state["stability_rows"] = stability_rows
                        state["job_status"] = job_status
                        save_checkpoint_state(output_dir, state)
            print(f"[INFO] finished scope={scope_label} split={split_label}", flush=True)

    grouped_rows = summarize_grouped(result_rows)
    fixed_rows = [row for row in result_rows if row.get("split") in {"blocked", "official"}]
    control_rows = [row for row in result_rows if row.get("model") != "real"]
    leaderboard_rows = leaderboard(result_rows, grouped_rows)
    promotion = promotion_summary(grouped_rows)
    promotion["frozen_fixed_split_references"] = frozen_references
    leakage = leakage_audit(candidate_descriptions, result_rows)
    run_manifest = {
        "schema_version": "veatic_raw_representation_audit_run_v1",
        "mode": mode,
        "elapsed_seconds": time.monotonic() - start,
        "manifest": str(manifest_path),
        "report": str(report_path),
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "tracked_output_dir": str(tracked_dir),
        "seed": args.seed,
        "pca_backend": args.pca_backend,
        "ridge_backend": args.ridge_backend,
        "heads": list(args.heads),
        "candidate_count": len(candidate_descriptions),
        "result_rows": len(result_rows),
        "skipped_rows": skipped_rows,
        "no_reencode": True,
        "skip_sensitivity": bool(args.skip_sensitivity),
        "checkpoint_state": str(checkpoint_state_path(output_dir)),
        "resume_supported": True,
        "frozen_fixed_split_references": frozen_references,
        "preflight": preflight,
    }
    outputs = {
        "inventory_csv": output_dir / "raw_cache_inventory.csv",
        "inventory_summary_json": output_dir / "raw_cache_inventory_summary.json",
        "inventory_report_md": output_dir / "raw_cache_inventory_report.md",
        "candidates_json": output_dir / "representation_candidates.json",
        "all_results_csv": output_dir / "representation_results_all.csv",
        "primary_results_csv": output_dir / "representation_results_primary.csv",
        "grouped_results_csv": output_dir / "grouped_video_results.csv",
        "fixed_results_csv": output_dir / "fixed_split_results.csv",
        "control_results_csv": output_dir / "control_results.csv",
        "matched_context_csv": output_dir / "matched_row_context_results.csv",
        "stability_csv": output_dir / "supervised_feature_selection_stability.csv",
        "leaderboard_csv": output_dir / "raw_vs_compressed_leaderboard.csv",
        "promotion_json": output_dir / "candidate_promotion_summary.json",
        "leakage_json": output_dir / "leakage_audit.json",
        "run_manifest_json": output_dir / "run_manifest.json",
        "report_md": output_dir / "representation_audit_report.md",
    }
    primary_rows = [
        row for row in result_rows
        if row.get("target_tier") == "primary" and row.get("model") == "real"
    ]
    write_csv(outputs["inventory_csv"], inventory_rows)
    write_json(outputs["inventory_summary_json"], inventory_summary)
    outputs["inventory_report_md"].write_text(inventory_report, encoding="utf-8")
    write_json(outputs["candidates_json"], candidate_descriptions)
    write_csv(outputs["all_results_csv"], result_rows + skipped_rows)
    write_csv(outputs["primary_results_csv"], primary_rows)
    write_csv(outputs["grouped_results_csv"], grouped_rows)
    write_csv(outputs["fixed_results_csv"], fixed_rows)
    write_csv(outputs["control_results_csv"], control_rows)
    write_csv(outputs["matched_context_csv"], matched_rows)
    write_csv(outputs["stability_csv"], stability_rows)
    write_csv(outputs["leaderboard_csv"], leaderboard_rows)
    write_json(outputs["promotion_json"], promotion)
    write_json(outputs["leakage_json"], leakage)
    write_json(outputs["run_manifest_json"], run_manifest)
    write_report(
        outputs["report_md"],
        mode=mode,
        inventory_summary=inventory_summary,
        candidates=candidate_descriptions,
        leaderboard_rows=leaderboard_rows,
        promotion=promotion,
        leakage=leakage,
        frozen_references=frozen_references,
        outputs=outputs,
    )
    lightweight = copy_lightweight_outputs(output_dir, tracked_dir, outputs)
    run_manifest["lightweight_tracked_outputs"] = {key: str(value) for key, value in lightweight.items()}
    write_json(outputs["run_manifest_json"], run_manifest)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2), flush=True)
    state["finalized"] = True
    state["run_manifest"] = run_manifest
    save_checkpoint_state(output_dir, state)
    return {"manifest": run_manifest, "outputs": outputs, "tracked_outputs": lightweight}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a no-reencode VEATIC raw cortical representation audit.")
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json")
    parser.add_argument("--cache-dir", default=str(default_cache_dir()))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tracked-output-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from an existing _checkpoint/state.json in --output-dir.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Run a tiny candidate/target/split subset.")
    mode.add_argument("--primary-audit", action="store_true", help="Run the focused primary audit candidate set.")
    mode.add_argument("--full-audit", action="store_true", help="Run all implemented candidate families.")
    parser.add_argument("--grouped-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="auto", choices=("auto", "mps_solve", "cpu_pinv"))
    parser.add_argument("--heads", nargs="+", default=["ridge_score"], choices=("ridge_score", "logistic_l2"))
    parser.add_argument("--skip-sensitivity", dest="skip_sensitivity", action="store_true", help="Run only the full all_videos 124/124 scope. This is the default.")
    parser.add_argument("--with-sensitivity", dest="skip_sensitivity", action="store_false", help="Also run the exclude_video_83 sensitivity scope after the full 124/124 scope.")
    parser.set_defaults(skip_sensitivity=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the audit contract without loading cache files.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate runtime dependencies without loading cache files.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(audit_plan(args), indent=2))
        return
    if args.preflight_only:
        print(json.dumps(dependency_preflight(args, selected_mode(args)), indent=2))
        return
    run_audit(args)


if __name__ == "__main__":
    main()
