"""Full-AGAIN AR-only context baseline.

This module computes an annotation-history AR baseline on the boundary-audited
1Hz AGAIN manifest. It intentionally does not run TRIBE, scout encoders, or any
VEATIC pipeline, and it does not try to compare sparse TRIBE features against
rows where those features do not exist.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from backend.scripts.again_scout_sparse_pipeline import (
    assert_again_only_output_path,
    default_boundary_manifest_root,
    safe_float,
)
from backend.scripts.again_sparse_tribe_teacher_500 import threshold_from_train, top_recall


SCHEMA_VERSION = "again_full_ar_context_v1"
BENCHMARK_MODE = "again_full_995_ar_context_only"
DEFAULT_MANIFEST_PATH = default_boundary_manifest_root() / "again_boundary_aligned_1hz_manifest.csv"
PRIMARY_TARGETS = (
    ("future_spike_1_3s_ge_0.05", 0.05),
    ("future_spike_1_3s_ge_0.075", 0.075),
)


@dataclass(frozen=True)
class FullArContextConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    report_dir: Path = Path("reports")
    report_date: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_seed: int = 20260622
    n_splits: int = 5
    ridge_alpha: float = 1.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: clean_csv(row.get(key, "")) for key in fields} for row in rows])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def clean_csv(value: Any) -> Any:
    value = clean_json(value)
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def parse_bool(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    return None


def group_rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["video_id"]), []).append(row)
    for video_rows in grouped.values():
        video_rows.sort(key=lambda item: safe_float(item.get("time_start_seconds"), 0.0))
    return grouped


def build_ar_rows(manifest_rows: list[dict[str, str]], target_column: str) -> list[dict[str, Any]]:
    grouped = group_rows_by_video(manifest_rows)
    out: list[dict[str, Any]] = []
    for video_id, rows in grouped.items():
        arousal_by_time = {
            float(safe_float(row.get("time_start_seconds"), 0.0)): float(safe_float(row.get("arousal"), 0.0))
            for row in rows
        }
        for row in rows:
            label = parse_bool(row.get(target_column))
            if label is None:
                continue
            if str(row.get("target_feasible_future_spike_1_3s", "")).strip().lower() != "true":
                continue
            t = float(safe_float(row.get("time_start_seconds"), 0.0))
            current = arousal_by_time.get(t, float(safe_float(row.get("arousal"), 0.0)))
            lag1 = arousal_by_time.get(t - 1.0, current)
            lag2 = arousal_by_time.get(t - 2.0, lag1)
            out.append(
                {
                    "dataset_name": "AGAIN_cleaned",
                    "benchmark_scope": "full_again_995_boundary_aligned_1hz",
                    "video_id": video_id,
                    "time_start_seconds": t,
                    "target_column": target_column,
                    "spike_label": label,
                    "arousal_current": current,
                    "arousal_lag1": lag1,
                    "arousal_lag2": lag2,
                    "arousal_delta1": current - lag1,
                    "arousal_delta2": lag1 - lag2,
                    "game": row.get("game", ""),
                    "genre": row.get("genre", ""),
                    "alignment_policy": row.get("alignment_policy", ""),
                    "row_has_sparse_tribe_features": False,
                }
            )
    return out


def finite_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["arousal_current"],
                row["arousal_lag1"],
                row["arousal_lag2"],
                row["arousal_delta1"],
                row["arousal_delta2"],
            ]
            for row in rows
        ],
        dtype=np.float32,
    )


def mlx_primal_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Low-dimensional MLX primal ridge for full-AGAIN AR rows."""
    xtr = mx.array(np.nan_to_num(train_x, nan=0.0, posinf=0.0, neginf=0.0), dtype=mx.float32)
    xte = mx.array(np.nan_to_num(test_x, nan=0.0, posinf=0.0, neginf=0.0), dtype=mx.float32)
    ytr = mx.array(train_y.astype(np.float32), dtype=mx.float32)
    mean = mx.mean(xtr, axis=0)
    std = mx.sqrt(mx.var(xtr, axis=0) + 1e-6)
    xtr = (xtr - mean) / std
    xte = (xte - mean) / std
    y_mean = mx.mean(ytr)
    y_centered = ytr - y_mean

    n_features = train_x.shape[1]
    gram = mx.transpose(xtr) @ xtr
    rhs = mx.transpose(xtr) @ y_centered
    system = gram + float(alpha) * mx.eye(n_features, dtype=mx.float32)
    weights = mx.zeros((n_features,), dtype=mx.float32)
    residual = rhs - system @ weights
    direction = residual
    rs_old = mx.sum(residual * residual)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        ap = system @ direction
        step = rs_old / (mx.sum(direction * ap) + 1e-8)
        weights = weights + step * direction
        residual = residual - step * ap
        rs_new = mx.sum(residual * residual)
        if float(np.asarray(rs_new)) < tol:
            rs_old = rs_new
            break
        direction = residual + (rs_new / (rs_old + 1e-8)) * direction
        rs_old = rs_new

    train_scores = xtr @ weights + y_mean
    test_scores = xte @ weights + y_mean
    mx.eval(train_scores, test_scores)
    return (
        np.asarray(train_scores, dtype=np.float32),
        np.asarray(test_scores, dtype=np.float32),
        {
            "ridge_backend": "mlx_primal_conjugate_gradient_low_dimensional",
            "ridge_iterations": iterations,
            "feature_width": int(n_features),
        },
    )


def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, scores) if len(np.unique(y_true)) > 1 else math.nan,
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else math.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if len(np.unique(y_true)) > 1 else math.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "top_1pct_recall": top_recall(y_true, scores, 0.01),
        "top_5pct_recall": top_recall(y_true, scores, 0.05),
        "top_10pct_recall": top_recall(y_true, scores, 0.10),
        "predicted_positive_rate": float(np.mean(pred)) if len(pred) else math.nan,
    }


def evaluate_target(rows: list[dict[str, Any]], *, target_column: str, alpha: float, n_splits: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.asarray([int(row["spike_label"]) for row in rows], dtype=int)
    groups = np.asarray([str(row["video_id"]) for row in rows])
    x = finite_matrix(rows)
    videos = sorted(set(groups))
    splits = min(n_splits, len(videos))
    fold_rows: list[dict[str, Any]] = []
    if splits < 2 or len(np.unique(y)) < 2:
        return fold_rows, {
            "target_column": target_column,
            "folds": 0,
            "mean_pr_auc": math.nan,
            "notes": "not enough groups or label classes",
        }

    gkf = GroupKFold(n_splits=splits)
    for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        train_scores, test_scores, info = mlx_primal_ridge_predict(
            x[train_idx],
            y_train,
            x[test_idx],
            alpha=alpha,
        )
        threshold = threshold_from_train(y_train, train_scores)
        fold_rows.append(
            {
                "benchmark_mode": BENCHMARK_MODE,
                "benchmark_scope": "full_again_995_boundary_aligned_1hz",
                "target_column": target_column,
                "model_lane": "AR_only",
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_videos": int(len(set(groups[train_idx]))),
                "test_videos": int(len(set(groups[test_idx]))),
                "train_event_count": int(np.sum(y_train)),
                "test_event_count": int(np.sum(y_test)),
                "train_positive_rate": float(np.mean(y_train)),
                "test_positive_rate": float(np.mean(y_test)),
                "decision_threshold_train_only": threshold,
                "row_population": "full_again_target_feasible_rows",
                "row_has_sparse_tribe_features": False,
                "comparable_to_sparse_pca128": False,
                "comparison_role": "full_dataset_ar_context_only",
                **info,
                **metric_row(y_test, test_scores, threshold),
            }
        )

    if not fold_rows:
        summary = {
            "target_column": target_column,
            "folds": 0,
            "mean_pr_auc": math.nan,
            "notes": "no valid grouped folds",
        }
    else:
        summary = {
            "benchmark_mode": BENCHMARK_MODE,
            "benchmark_scope": "full_again_995_boundary_aligned_1hz",
            "target_column": target_column,
            "model_lane": "AR_only",
            "folds": len(fold_rows),
            "videos": int(len(videos)),
            "rows": int(len(rows)),
            "event_count": int(np.sum(y)),
            "positive_rate": float(np.mean(y)),
            "mean_pr_auc": float(np.nanmean([row["pr_auc"] for row in fold_rows])),
            "mean_roc_auc": float(np.nanmean([row["roc_auc"] for row in fold_rows])),
            "mean_f1": float(np.nanmean([row["f1"] for row in fold_rows])),
            "mean_balanced_accuracy": float(np.nanmean([row["balanced_accuracy"] for row in fold_rows])),
            "mean_top_5pct_recall": float(np.nanmean([row["top_5pct_recall"] for row in fold_rows])),
            "mean_top_10pct_recall": float(np.nanmean([row["top_10pct_recall"] for row in fold_rows])),
            "ridge_backend": "mlx_primal_conjugate_gradient_low_dimensional",
            "row_has_sparse_tribe_features": False,
            "comparable_to_sparse_pca128": False,
            "notes": "Context baseline only. Sparse PCA128 cannot be evaluated on full 995 until sparse features exist for a full-scope queue.",
        }
    return fold_rows, summary


def build_report(summary: dict[str, Any], lane_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# AGAIN Full AR Context Baseline",
        "",
        "## Scope",
        "",
        "- This is a full-AGAIN annotation-history AR context baseline.",
        "- It uses the boundary-audited 1Hz manifest only.",
        "- It does not run TRIBE, V-JEPA, or sparse PCA128.",
        "- It must not be treated as a direct row-matched comparison against the 50-video sparse teacher pilot.",
        "",
        "## Results",
        "",
    ]
    for row in lane_rows:
        lines.append(
            f"- `{row['target_column']}`: videos `{row['videos']}`, rows `{row['rows']}`, "
            f"events `{row['event_count']}`, PR-AUC `{100 * row['mean_pr_auc']:.2f}%`, "
            f"ROC-AUC `{100 * row['mean_roc_auc']:.2f}%`"
        )
    lines.extend(
        [
            "",
            "## Fix Applied",
            "",
            "The sparse 500-window pilot remains a 50-video sparse-row pilot. This full AR context file is the correct",
            "full-dataset denominator for AR-only, but it is not a replacement for a row-matched AR + sparse PCA128 test.",
            "To test AR + PCA128 on all AGAIN videos, sparse PCA128 features must first be generated for a full-scope queue",
            "covering the 995 videos under the same row contract.",
            "",
            "## Guardrails",
            "",
            f"- tribe_encoding_run=`{summary['tribe_encoding_run']}`",
            f"- models_trained=`{summary['models_trained']}`",
            f"- veatic_outputs_modified=`{summary['veatic_outputs_modified']}`",
            f"- direct_sparse_pca128_comparison_made=`{summary['direct_sparse_pca128_comparison_made']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_full_ar_context(*, output_root: Path, config: FullArContextConfig) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root exists and is non-empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv_rows(config.manifest_path)
    fold_rows: list[dict[str, Any]] = []
    lane_rows: list[dict[str, Any]] = []
    target_row_counts: dict[str, int] = {}
    for target_column, _threshold in PRIMARY_TARGETS:
        target_rows = build_ar_rows(manifest_rows, target_column)
        target_row_counts[target_column] = len(target_rows)
        target_fold_rows, target_summary = evaluate_target(
            target_rows,
            target_column=target_column,
            alpha=config.ridge_alpha,
            n_splits=config.n_splits,
        )
        fold_rows.extend(target_fold_rows)
        lane_rows.append(target_summary)

    unique_videos = len({row.get("video_id", "") for row in manifest_rows if row.get("video_id")})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_mode": BENCHMARK_MODE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "manifest_path": str(config.manifest_path),
        "manifest_rows": len(manifest_rows),
        "manifest_videos": unique_videos,
        "target_row_counts": target_row_counts,
        "targets": [target for target, _ in PRIMARY_TARGETS],
        "ridge_backend": "mlx_primal_conjugate_gradient_low_dimensional",
        "row_population": "full_again_target_feasible_rows",
        "tribe_encoding_run": False,
        "vjepa_encoding_run": False,
        "sparse_pca128_features_used": False,
        "direct_sparse_pca128_comparison_made": False,
        "models_trained": True,
        "training_scope": "full_again_ar_context_grouped_video_cv",
        "veatic_outputs_modified": False,
        "notes": "Full AR-only context. Direct AR+PCA128 requires matching sparse feature rows across the target scope.",
    }
    write_csv(output_root / "again_full_ar_context_lane_results.csv", lane_rows)
    write_csv(output_root / "again_full_ar_context_fold_results.csv", fold_rows)
    write_json(output_root / "again_full_ar_context_summary.json", summary)
    report = build_report(summary, lane_rows)
    report_path = config.report_dir / f"again_full_ar_context_{config.report_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    run_manifest = {
        **summary,
        "report_path": str(report_path),
        "files_written": [
            str(output_root / "again_full_ar_context_lane_results.csv"),
            str(output_root / "again_full_ar_context_fold_results.csv"),
            str(output_root / "again_full_ar_context_summary.json"),
            str(output_root / "run_manifest.json"),
            str(report_path),
        ],
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    return run_manifest
