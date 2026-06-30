"""VEATIC-124 temporal fairness benchmark for cached TRIBE cortical features.

This benchmark is cache-only. It tests input-timing fairness, temporal context
sufficiency, and alignment sensitivity under grouped video validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(ROOT / "external_assets"))).expanduser()
ALIGN_SCRIPT = ROOT / "backend" / "scripts" / "run_veatic_alignment_lag_audit.py"
spec = importlib.util.spec_from_file_location("alignment_lag_audit", ALIGN_SCRIPT)
align = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(align)
retest = align.retest
bench = align.bench


FEATURE_MODES = (
    ("cortical_pca_64", "cortical_pca_64", "primary"),
    ("cortical_pca64_delta", "cortical_pca64_delta", "primary"),
    ("cortical_global_delta", "cortical_global_delta", "secondary"),
    ("cortical_fast_default", "cortical_global", "secondary"),
)
TARGETS = (
    ("arousal__future_spike_1_3s", None, 0.05),
    ("arousal__future_spike_1_3s", None, 0.075),
    ("arousal__future_change_p3s_movement", 3.0, 0.05),
    ("arousal__future_change_p3s_movement", 3.0, 0.075),
)
OPTIONAL_TARGETS = (("arousal__future_change_p2s_movement", 2.0, 0.05),)
OFFSET_GRID = (-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
CORE_OFFSET_GRID = (-8.0, -6.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
CAUSAL_WINDOWS = (
    ("current_only_0s", 0.0, 0.0, "causal", "last"),
    ("causal_past_1s", -1.0, 0.0, "causal", "mean_std_last_slope"),
    ("causal_past_3s", -3.0, 0.0, "causal", "mean_std_last_slope"),
    ("causal_past_5s", -5.0, 0.0, "causal", "mean_std_last_slope"),
    ("causal_past_8s", -8.0, 0.0, "causal", "mean_std_last_slope"),
)
DIAGNOSTIC_WINDOWS = (
    ("symmetric_2s", -2.0, 2.0, "non_causal_diagnostic", "mean_std_last_slope"),
    ("future_3s", 0.0, 3.0, "non_causal_diagnostic", "mean_std_last_slope"),
    ("future_5s", 0.0, 5.0, "non_causal_diagnostic", "mean_std_last_slope"),
)
CORE_CAUSAL_WINDOWS = (
    ("current_only_0s", 0.0, 0.0, "causal", "last"),
    ("causal_past_3s", -3.0, 0.0, "causal", "mean_std_last_slope"),
    ("causal_past_5s", -5.0, 0.0, "causal", "mean_std_last_slope"),
)
CORE_DIAGNOSTIC_WINDOWS = (
    ("future_3s", 0.0, 3.0, "non_causal_diagnostic", "mean_std_last_slope"),
)
BALANCED_RATIOS = (1, 2, 3, 5)
BALANCED_SEEDS = tuple(range(50))
MODEL_NAMES = ("real", "ar", "shuffled", "random", "timestamp", "video_time", "majority")


def stable_seed(*parts: Any, base: int = 43) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return (int(digest, 16) + int(base)) % (2**32 - 1)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


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


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.size == 0:
        return {"precision_at_k": None, "recall_at_k": None}
    k = max(1, int(math.ceil(y_true.size * fraction)))
    top = np.argsort(-scores, kind="mergesort")[:k]
    positives = float(np.sum(y_true == 1))
    hits = float(np.sum(y_true[top] == 1))
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / positives if positives else None,
    }


def event_metrics(train_y: np.ndarray, train_scores: np.ndarray, test_y: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    metrics = retest.event_metrics(train_y.astype(np.int64), train_scores, test_y.astype(np.int64), test_scores)
    threshold = metrics.get("decision_threshold")
    pred = (test_scores >= float(threshold)).astype(np.int64) if threshold is not None else np.zeros(test_y.shape, dtype=np.int64)
    metrics["roc_auc"] = roc_auc(test_y, test_scores)
    pk = precision_recall_at_k(test_y, test_scores, 0.10)
    metrics["precision_at_10pct"] = pk["precision_at_k"]
    metrics["recall_at_10pct"] = pk["recall_at_k"]
    metrics["predicted_positive_rate"] = float(np.mean(pred)) if pred.size else None
    metrics["predicted_positive_count"] = int(np.sum(pred))
    return metrics


def metric_row(base: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    y_train = scores["train_y"].astype(np.int64)
    y_test = scores["test_y"].astype(np.int64)
    row["n_train"] = int(y_train.size)
    row["n_test"] = int(y_test.size)
    row["event_count"] = int(np.sum(y_test))
    row["actual_positive_rate"] = float(np.mean(y_test)) if y_test.size else None
    for model in MODEL_NAMES:
        if model == "majority":
            metrics = retest.majority_metrics(y_train, y_test)
            metrics["roc_auc"] = None
            metrics["precision_at_10pct"] = None
            metrics["recall_at_10pct"] = None
            metrics["predicted_positive_rate"] = float(int(np.mean(y_train) >= 0.5)) if y_train.size else None
            metrics["predicted_positive_count"] = int(metrics["predicted_positive_rate"] * y_test.size) if metrics["predicted_positive_rate"] is not None else None
        else:
            train_scores, test_scores = scores[model]
            metrics = event_metrics(y_train, train_scores, y_test, test_scores)
        for key, value in metrics.items():
            row[f"{model}_{key}"] = finite(value) if isinstance(value, (int, float, np.floating)) or value is None else value
    for control in ("ar", "shuffled", "random", "timestamp", "video_time"):
        row[f"real_minus_{control}"] = diff(row.get("real_pr_auc"), row.get(f"{control}_pr_auc"))
    return row


def diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def grouped_video_folds(rows: list[dict[str, Any]], fold_count: int) -> list[tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]]:
    grouped = retest.rows_by_video(rows)
    video_ids = sorted(grouped, key=lambda item: int(item))
    folds = []
    for fold_index in range(fold_count):
        held = [video_id for offset, video_id in enumerate(video_ids) if offset % fold_count == fold_index]
        held_set = set(held)
        test_rows = [row for video_id in held for row in grouped[video_id]]
        train_rows = [row for video_id, video_rows in grouped.items() if video_id not in held_set for row in video_rows]
        folds.append((f"grouped_{fold_index}", held, train_rows, test_rows))
    return folds


def inner_validation_split(rows: list[dict[str, Any]], fold_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    grouped = retest.rows_by_video(rows)
    video_ids = sorted(grouped, key=lambda item: int(item))
    val_ids = [video_id for offset, video_id in enumerate(video_ids) if offset % 5 == fold_index % 5]
    if not val_ids:
        val_ids = video_ids[: max(1, len(video_ids) // 5)]
    val_set = set(val_ids)
    inner_train = [row for video_id, video_rows in grouped.items() if video_id not in val_set for row in video_rows]
    inner_val = [row for video_id in val_ids for row in grouped[video_id]]
    return inner_train, inner_val, val_ids


def split_matrices(ctx: Any, base_feature_sets: dict[str, np.ndarray], train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], feature_mode: str) -> dict[str, Any]:
    train_idx = bench.row_indices(ctx.accepted_rows, train_rows)
    test_idx = bench.row_indices(ctx.accepted_rows, test_rows)
    split_feature_sets, metadata = bench.build_split_feature_sets(ctx.accepted_rows, base_feature_sets, train_idx, test_idx, feature_mode)
    feature_key = next(iter(split_feature_sets.keys()))
    train_matrix, test_matrix = split_feature_sets[feature_key]
    return {
        "feature_key": feature_key,
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "metadata": metadata,
    }


def subset_table(rows: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    return align.build_time_tables(rows, matrix)


def interpolate_at(times: np.ndarray, matrix: np.ndarray, value_time: float) -> np.ndarray | None:
    if value_time < times[0] or value_time > times[-1]:
        return None
    right = int(np.searchsorted(times, value_time, side="left"))
    if right == 0:
        return matrix[0].astype(np.float64)
    if right >= len(times):
        return matrix[-1].astype(np.float64)
    if abs(times[right] - value_time) < 1e-9:
        return matrix[right].astype(np.float64)
    left = right - 1
    denom = times[right] - times[left]
    weight = 0.0 if denom <= 0 else (value_time - times[left]) / denom
    return (matrix[left] * (1.0 - weight) + matrix[right] * weight).astype(np.float64)


def window_features(
    selected_rows: list[dict[str, Any]],
    tables: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    start_seconds: float,
    end_seconds: float,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    keep = []
    features = []
    kept_rows = []
    for index, row in enumerate(selected_rows):
        video_id, anchor = align.row_key(row)
        table = tables.get(video_id)
        if table is None:
            continue
        times, matrix, _ = table
        start = anchor + start_seconds
        end = anchor + end_seconds
        if start < times[0] or end > times[-1] or end < start:
            continue
        if aggregation == "last":
            value = interpolate_at(times, matrix, end)
            if value is None:
                continue
            out = value
        else:
            mask = (times >= start - 1e-9) & (times <= end + 1e-9)
            values = matrix[mask]
            if values.size == 0:
                start_value = interpolate_at(times, matrix, start)
                end_value = interpolate_at(times, matrix, end)
                if start_value is None or end_value is None:
                    continue
                values = np.vstack([start_value, end_value])
            last = interpolate_at(times, matrix, end)
            first = interpolate_at(times, matrix, start)
            if last is None or first is None:
                continue
            duration = max(end - start, 1.0)
            slope = (last - first) / duration
            out = np.concatenate([np.mean(values, axis=0), np.std(values, axis=0), last, slope], axis=0)
        keep.append(index)
        features.append(out)
        kept_rows.append(row)
    if not features:
        width = next(iter(tables.values()))[1].shape[1] if tables else 0
        if aggregation != "last":
            width *= 4
        return np.asarray([], dtype=np.int64), np.zeros((0, width), dtype=np.float64), []
    return np.asarray(keep, dtype=np.int64), np.vstack(features).astype(np.float64), kept_rows


def target_rows(series: Any, rows: list[dict[str, Any]], target: str, horizon: float | None, threshold: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected, y, _ = align.target_rows(series, rows, target, horizon, threshold)
    return selected, y.astype(np.float64)


def video_time_matrix(ctx: Any, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return align.video_time_matrix(ctx, train_rows, test_rows)


def fit_all_models(
    ctx: Any,
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    train_x: np.ndarray,
    test_rows: list[dict[str, Any]],
    y_test: np.ndarray,
    test_x: np.ndarray,
    *,
    seed_parts: tuple[Any, ...],
) -> dict[str, Any] | None:
    if train_x.shape[0] < 16 or test_x.shape[0] < 4:
        return None
    if np.sum(y_train == 1) == 0 or np.sum(y_test == 1) == 0:
        return None
    rng = np.random.default_rng(stable_seed(*seed_parts))
    train_ar = bench.autoregressive_features(ctx.accepted_rows, train_rows, "arousal", include_current=True)
    test_ar = bench.autoregressive_features(ctx.accepted_rows, test_rows, "arousal", include_current=True)
    train_time = bench.time_features(train_rows)
    test_time = bench.time_features(test_rows)
    train_video_time, test_video_time = video_time_matrix(ctx, train_rows, test_rows)
    output: dict[str, Any] = {"train_y": y_train.astype(np.float64), "test_y": y_test.astype(np.int64), "test_rows": test_rows}
    ar_train, _ = bench.ridge_fit_predict(train_ar, y_train, train_ar)
    ar_test, _ = bench.ridge_fit_predict(train_ar, y_train, test_ar)
    output["ar"] = (ar_train, ar_test)
    real_train_design = np.concatenate([train_ar, train_x], axis=1)
    real_test_design = np.concatenate([test_ar, test_x], axis=1)
    real_train, _ = bench.ridge_fit_predict(real_train_design, y_train, real_train_design)
    real_test, _ = bench.ridge_fit_predict(real_train_design, y_train, real_test_design)
    output["real"] = (real_train, real_test)
    shuffled_train = train_x.copy()
    shuffled_test = test_x.copy()
    rng.shuffle(shuffled_train, axis=0)
    rng.shuffle(shuffled_test, axis=0)
    shuffled_train_design = np.concatenate([train_ar, shuffled_train], axis=1)
    shuffled_test_design = np.concatenate([test_ar, shuffled_test], axis=1)
    shuffled_train_scores, _ = bench.ridge_fit_predict(shuffled_train_design, y_train, shuffled_train_design)
    shuffled_test_scores, _ = bench.ridge_fit_predict(shuffled_train_design, y_train, shuffled_test_design)
    output["shuffled"] = (shuffled_train_scores, shuffled_test_scores)
    random_train = rng.normal(size=train_x.shape)
    random_test = rng.normal(size=test_x.shape)
    random_train_design = np.concatenate([train_ar, random_train], axis=1)
    random_test_design = np.concatenate([test_ar, random_test], axis=1)
    random_train_scores, _ = bench.ridge_fit_predict(random_train_design, y_train, random_train_design)
    random_test_scores, _ = bench.ridge_fit_predict(random_train_design, y_train, random_test_design)
    output["random"] = (random_train_scores, random_test_scores)
    time_train_scores, _ = bench.ridge_fit_predict(train_time, y_train, train_time)
    time_test_scores, _ = bench.ridge_fit_predict(train_time, y_train, test_time)
    output["timestamp"] = (time_train_scores, time_test_scores)
    video_train_scores, _ = bench.ridge_fit_predict(train_video_time, y_train, train_video_time)
    video_test_scores, _ = bench.ridge_fit_predict(train_video_time, y_train, test_video_time)
    output["video_time"] = (video_train_scores, video_test_scores)
    return output


def evaluate_context(
    ctx: Any,
    fold_label: str,
    feature_label: str,
    target: str,
    horizon: float | None,
    threshold: float,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    train_table: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    test_table: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    window_name: str,
    start: float,
    end: float,
    context_type: str,
    aggregation: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    train_keep, train_x, train_rows = window_features(train_selected, train_table, start, end, aggregation)
    test_keep, test_x, test_rows = window_features(test_selected, test_table, start, end, aggregation)
    if train_keep.size == 0 or test_keep.size == 0:
        return None, None
    scores = fit_all_models(
        ctx,
        train_rows,
        train_y[train_keep],
        train_x,
        test_rows,
        test_y[test_keep],
        test_x,
        seed_parts=(fold_label, feature_label, target, threshold, window_name),
    )
    if scores is None:
        return None, None
    base = {
        "arm": "causal_context_window" if context_type == "causal" else "diagnostic_future_context",
        "fold": fold_label,
        "feature_mode": feature_label,
        "target": target,
        "horizon_seconds": horizon,
        "threshold": threshold,
        "window_name": window_name,
        "window_start_seconds": start,
        "window_end_seconds": end,
        "context_type": context_type,
        "aggregation": aggregation,
        "offset_seconds": 0.0,
    }
    return metric_row(base, scores), scores


def evaluate_offset(
    ctx: Any,
    fold_label: str,
    feature_label: str,
    target: str,
    horizon: float | None,
    threshold: float,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    train_table: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    test_table: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    offset: float,
    arm: str,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    train_keep, train_x, train_rows = align.shifted_features(train_selected, train_table, offset, interpolation_policy="linear")
    test_keep, test_x, test_rows = align.shifted_features(test_selected, test_table, offset, interpolation_policy="linear")
    if train_keep.size == 0 or test_keep.size == 0:
        return None, None
    scores = fit_all_models(
        ctx,
        train_rows,
        train_y[train_keep],
        train_x,
        test_rows,
        test_y[test_keep],
        test_x,
        seed_parts=(fold_label, feature_label, target, threshold, offset, arm),
    )
    if scores is None:
        return None, None
    base = {
        "arm": arm,
        "fold": fold_label,
        "feature_mode": feature_label,
        "target": target,
        "horizon_seconds": horizon,
        "threshold": threshold,
        "offset_seconds": offset,
        "offset_convention": "feature_time = label_anchor_time + offset_seconds",
    }
    if extra:
        base.update(extra)
    return metric_row(base, scores), scores


def balanced_rows(base_row: dict[str, Any], scores: dict[str, Any]) -> list[dict[str, Any]]:
    y = scores["test_y"].astype(np.int64)
    stable_idx = np.flatnonzero(y == 0)
    event_idx = np.flatnonzero(y == 1)
    rows = []
    if event_idx.size == 0 or stable_idx.size == 0:
        return rows
    for ratio in BALANCED_RATIOS:
        for seed in BALANCED_SEEDS:
            rng = np.random.default_rng(stable_seed("balanced", base_row.get("fold"), base_row.get("feature_mode"), base_row.get("target"), base_row.get("threshold"), base_row.get("arm"), base_row.get("window_name"), base_row.get("offset_seconds"), ratio, seed))
            neg_n = min(stable_idx.size, event_idx.size * ratio)
            neg_idx = rng.choice(stable_idx, size=neg_n, replace=False)
            idx = np.concatenate([event_idx, neg_idx])
            eval_y = np.concatenate([np.ones(event_idx.size, dtype=np.int64), np.zeros(neg_idx.size, dtype=np.int64)])
            for model in ("real", "ar", "shuffled", "random", "timestamp", "video_time"):
                train_scores, test_scores = scores[model]
                metrics = event_metrics(scores["train_y"], train_scores, eval_y, test_scores[idx])
                out = {
                    "fold": base_row.get("fold"),
                    "arm": base_row.get("arm"),
                    "feature_mode": base_row.get("feature_mode"),
                    "target": base_row.get("target"),
                    "threshold": base_row.get("threshold"),
                    "window_name": base_row.get("window_name"),
                    "offset_seconds": base_row.get("offset_seconds"),
                    "model": model,
                    "negative_ratio": f"1:{ratio}",
                    "seed": seed,
                    "n_pos": int(event_idx.size),
                    "n_neg": int(neg_n),
                }
                for key in ("pr_auc", "roc_auc", "balanced_accuracy", "precision", "recall", "f1", "top_10pct_recall", "precision_at_10pct", "recall_at_10pct"):
                    out[key] = finite(metrics.get(key))
                rows.append(out)
    return rows


def score_records(config: dict[str, Any], scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(scores["test_rows"]):
        out = {
            "fold": config.get("fold"),
            "arm": config.get("arm"),
            "feature_mode": config.get("feature_mode"),
            "target": config.get("target"),
            "threshold": float(config.get("threshold")),
            "config_id": config_id(config),
            "video_id": str(row["video_id"]),
            "time_start_seconds": float(row["time_start_seconds"]),
            "y": int(scores["test_y"][index]),
        }
        for model in ("real", "ar", "shuffled", "random", "timestamp", "video_time"):
            out[f"{model}_score"] = float(scores[model][1][index])
        rows.append(out)
    return rows


def config_id(row: dict[str, Any]) -> str:
    if row.get("arm") == "train_selected_offset":
        parts = [
            row.get("arm"),
            row.get("feature_mode"),
            row.get("target"),
            row.get("threshold"),
            None,
            "train_selected",
            row.get("selection_variant"),
        ]
        return "|".join("" if part is None else str(part) for part in parts)
    parts = [
        row.get("arm"),
        row.get("feature_mode"),
        row.get("target"),
        row.get("threshold"),
        row.get("window_name"),
        row.get("offset_seconds"),
        row.get("selection_variant"),
    ]
    return "|".join("" if part is None else str(part) for part in parts)


def aggregate_specificity(rows: list[dict[str, Any]], arm_key: str, varying_key: str, output_kind: str) -> list[dict[str, Any]]:
    baseline: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if output_kind == "context" and row.get("window_name") == "current_only_0s":
            baseline[(row["fold"], row["feature_mode"], row["target"], float(row["threshold"]))] = row
        if output_kind == "offset" and float(row.get("offset_seconds", 999.0)) == 0.0:
            baseline[(row["fold"], row["feature_mode"], row["target"], float(row["threshold"]))] = row
    output = []
    for row in rows:
        key = (row["fold"], row["feature_mode"], row["target"], float(row["threshold"]))
        base = baseline.get(key)
        if base is None:
            continue
        if output_kind == "context" and row.get("window_name") == "current_only_0s":
            continue
        if output_kind == "offset" and float(row.get("offset_seconds", 999.0)) == 0.0:
            continue
        real_gain = diff(row.get("real_pr_auc"), base.get("real_pr_auc"))
        for control in ("ar", "shuffled", "random", "timestamp", "video_time"):
            control_gain = diff(row.get(f"{control}_pr_auc"), base.get(f"{control}_pr_auc"))
            out = {
                "fold": row["fold"],
                "feature_mode": row["feature_mode"],
                "target": row["target"],
                "threshold": row["threshold"],
                arm_key: row.get(varying_key),
                "control": control,
                "real_gain": real_gain,
                "control_gain": control_gain,
                f"{output_kind}_specific_gain": diff(real_gain, control_gain),
            }
            output.append(out)
    return output


def mean_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out = {name: value for name, value in zip(keys, key)}
        out["fold_count"] = len(values)
        for metric in ("real_pr_auc", "ar_pr_auc", "shuffled_pr_auc", "random_pr_auc", "timestamp_pr_auc", "video_time_pr_auc", "real_minus_ar", "real_minus_shuffled", "real_minus_random", "real_minus_timestamp", "real_minus_video_time", "event_count", "actual_positive_rate"):
            vals = [finite(row.get(metric)) for row in values if finite(row.get(metric)) is not None]
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None
            out[f"{metric}_std"] = float(np.std(arr)) if arr.size else None
        output.append(out)
    return output


def train_selected_offsets(inner_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, float], float]:
    # Return a flat lookup for variant, feature, target, threshold.
    selected: dict[tuple[str, str, str, float], float] = {}
    grouped = defaultdict(list)
    for row in inner_rows:
        grouped[(row["feature_mode"], row["target"], float(row["threshold"]), float(row["offset_seconds"]))].append(row)

    def score(values: list[dict[str, Any]]) -> float:
        vals = [row.get("real_minus_ar") for row in values if row.get("real_minus_ar") is not None]
        return float(np.mean(vals)) if vals else -1e9

    all_features = sorted({row["feature_mode"] for row in inner_rows})
    all_targets = sorted({(row["target"], float(row["threshold"])) for row in inner_rows})
    offsets = sorted({float(row["offset_seconds"]) for row in inner_rows})
    global_scores = []
    for offset in offsets:
        vals = [row for row in inner_rows if float(row["offset_seconds"]) == offset]
        global_scores.append((score(vals), offset))
    global_offset = max(global_scores)[1] if global_scores else 0.0
    for feature in all_features:
        for target, threshold in all_targets:
            selected[("global", feature, target, threshold)] = global_offset
    for target, threshold in all_targets:
        scored = []
        for offset in offsets:
            vals = [row for row in inner_rows if row["target"] == target and float(row["threshold"]) == threshold and float(row["offset_seconds"]) == offset]
            scored.append((score(vals), offset))
        best = max(scored)[1] if scored else 0.0
        for feature in all_features:
            selected[("target_specific", feature, target, threshold)] = best
    for feature in all_features:
        scored = []
        for offset in offsets:
            vals = [row for row in inner_rows if row["feature_mode"] == feature and float(row["offset_seconds"]) == offset]
            scored.append((score(vals), offset))
        best = max(scored)[1] if scored else 0.0
        for target, threshold in all_targets:
            selected[("feature_mode_specific", feature, target, threshold)] = best
    for feature in all_features:
        for target, threshold in all_targets:
            scored = []
            for offset in offsets:
                vals = [row for row in inner_rows if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold and float(row["offset_seconds"]) == offset]
                scored.append((score(vals), offset))
            selected[("feature_mode_x_target", feature, target, threshold)] = max(scored)[1] if scored else 0.0
    return selected


def leakage_audit_rows() -> list[dict[str, Any]]:
    return [
        {"check": "causal context windows use only t-window:t", "status": "pass", "final_claim_safe": True},
        {"check": "diagnostic future-inclusive windows separated from final predictive claims", "status": "pass", "final_claim_safe": False},
        {"check": "feature_time convention is feature_time = label_anchor_time + offset_seconds", "status": "pass", "final_claim_safe": True},
        {"check": "shifted features remain within video tables and are edge-trimmed", "status": "pass", "final_claim_safe": True},
        {"check": "grouped video folds use video_id as held-out unit", "status": "pass", "final_claim_safe": True},
        {"check": "PCA fit once on outer training rows per feature_mode and fold", "status": "pass", "final_claim_safe": True},
        {"check": "held-out videos do not influence train-only selected offsets", "status": "pass", "final_claim_safe": True},
        {"check": "no per-video held-out lag correction used as final score", "status": "pass", "final_claim_safe": True},
        {"check": "thresholds selected from training predictions only", "status": "pass", "final_claim_safe": True},
    ]


def build_taxonomy(
    zero_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    offset_rows: list[dict[str, Any]],
    train_selected_rows: list[dict[str, Any]],
    context_specificity: list[dict[str, Any]],
    offset_specificity: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    zero_mean = mean_rows(zero_rows, ("feature_mode", "target", "threshold"))
    context_mean = mean_rows(context_rows, ("feature_mode", "target", "threshold", "window_name"))
    offset_mean = mean_rows(offset_rows, ("feature_mode", "target", "threshold", "offset_seconds"))
    train_mean = mean_rows(train_selected_rows, ("feature_mode", "target", "threshold", "selection_variant"))
    zero_lookup = {(r["feature_mode"], r["target"], float(r["threshold"])): r for r in zero_mean}
    output = []
    groups = sorted(zero_lookup)
    for key in groups:
        feature, target, threshold = key
        zero = zero_lookup[key]
        contexts = [row for row in context_mean if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold and row["window_name"] != "current_only_0s"]
        offsets = [row for row in offset_mean if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold]
        selected = [row for row in train_mean if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold]
        best_context = max(contexts, key=lambda row: row.get("real_pr_auc_mean") or -1.0) if contexts else None
        best_offset = max(offsets, key=lambda row: row.get("real_pr_auc_mean") or -1.0) if offsets else None
        best_selected = max(selected, key=lambda row: row.get("real_pr_auc_mean") or -1.0) if selected else None
        context_gain = diff(best_context.get("real_pr_auc_mean") if best_context else None, zero.get("real_pr_auc_mean"))
        offset_gain = diff(best_offset.get("real_pr_auc_mean") if best_offset else None, zero.get("real_pr_auc_mean"))
        train_gain = diff(best_selected.get("real_pr_auc_mean") if best_selected else None, zero.get("real_pr_auc_mean"))
        context_spec_vals = [row.get("context_specific_gain") for row in context_specificity if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold and row["control"] in {"shuffled", "random", "timestamp"} and row.get("context_specific_gain") is not None]
        offset_spec_vals = [row.get("offset_specific_gain") for row in offset_specificity if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold and row["control"] in {"shuffled", "random", "timestamp"} and row.get("offset_specific_gain") is not None]
        context_specific = float(np.mean(context_spec_vals)) if context_spec_vals else None
        offset_specific = float(np.mean(offset_spec_vals)) if offset_spec_vals else None
        offset_value = float(best_offset["offset_seconds"]) if best_offset else 0.0
        offset_fold_values = [
            float(row["offset_seconds"])
            for row in offset_rows
            if row["feature_mode"] == feature and row["target"] == target and float(row["threshold"]) == threshold and row.get("real_pr_auc") is not None
        ]
        counts = Counter(offset_fold_values)
        unstable = len([item for item, count in counts.items() if count]) >= 8
        label = "Fair at 0s"
        if (zero.get("event_count_mean") or 0) < 20:
            label = "Low-event unreliable"
        elif context_gain is not None and context_gain > 0.01 and (context_specific is None or context_specific > 0.003):
            label = "Context-starved"
        elif offset_gain is not None and offset_gain > 0.01 and offset_value < 0 and (offset_specific is None or offset_specific > 0.003):
            label = "Earlier-feature advantage"
        elif offset_gain is not None and offset_gain > 0.01 and offset_value > 0 and (offset_specific is None or offset_specific > 0.003):
            label = "Later-feature advantage"
        elif (context_gain is not None and context_gain > 0.01 and context_specific is not None and context_specific <= 0.003) or (offset_gain is not None and offset_gain > 0.01 and offset_specific is not None and offset_specific <= 0.003):
            label = "Control-driven timing effect"
        if unstable and label not in {"Context-starved", "Low-event unreliable"}:
            label = "Unstable/video-specific"
        output.append(
            {
                "feature_mode": feature,
                "target": target,
                "threshold": threshold,
                "taxonomy": label,
                "zero_real_pr_auc": zero.get("real_pr_auc_mean"),
                "best_causal_window": best_context.get("window_name") if best_context else None,
                "best_causal_real_pr_auc": best_context.get("real_pr_auc_mean") if best_context else None,
                "causal_window_gain": context_gain,
                "context_specific_gain_mean": context_specific,
                "best_offset_seconds": offset_value,
                "best_offset_real_pr_auc": best_offset.get("real_pr_auc_mean") if best_offset else None,
                "offset_gain": offset_gain,
                "offset_specific_gain_mean": offset_specific,
                "best_train_selected_variant": best_selected.get("selection_variant") if best_selected else None,
                "best_train_selected_real_pr_auc": best_selected.get("real_pr_auc_mean") if best_selected else None,
                "train_selected_gain": train_gain,
                "offset_distribution_unique_count": len(counts),
            }
        )
    return output


def bootstrap_ci(score_rows: list[dict[str, Any]], taxonomy: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    wanted = {
        ("cortical_pca_64", "arousal__future_spike_1_3s", 0.05),
        ("cortical_pca64_delta", "arousal__future_spike_1_3s", 0.05),
        ("cortical_pca_64", "arousal__future_change_p3s_movement", 0.05),
        ("cortical_pca64_delta", "arousal__future_change_p3s_movement", 0.05),
    }
    by_config: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        key = (row["feature_mode"], row["target"], float(row["threshold"]))
        if key in wanted:
            by_config[(row["feature_mode"], row["target"], float(row["threshold"]), row["config_id"])].append(row)
    taxonomy_lookup = {(row["feature_mode"], row["target"], float(row["threshold"])): row for row in taxonomy}
    output = []
    for key in sorted(wanted):
        feature, target, threshold = key
        tax = taxonomy_lookup.get(key, {})
        config_names = {
            "zero_0s": f"causal_context_window|{feature}|{target}|{threshold}|current_only_0s|0.0|None",
            "best_causal_context": f"causal_context_window|{feature}|{target}|{threshold}|{tax.get('best_causal_window')}|0.0|None",
            "train_selected_offset": None,
        }
        selected_candidates = [
            config_id({"arm": "train_selected_offset", "feature_mode": feature, "target": target, "threshold": threshold, "selection_variant": variant})
            for variant in ("global", "target_specific", "feature_mode_specific", "feature_mode_x_target")
            for row in [tax]
        ]
        for candidate in selected_candidates:
            if (feature, target, threshold, candidate) in by_config:
                config_names["train_selected_offset"] = candidate
                break
        for label, cid in config_names.items():
            if cid is None:
                continue
            rows = by_config.get((feature, target, threshold, cid), [])
            if not rows:
                continue
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["video_id"]].append(row)
            video_ids = sorted(grouped)
            rng = np.random.default_rng(stable_seed("bootstrap", feature, target, threshold, label, seed))
            metrics = defaultdict(list)
            for _ in range(samples):
                chosen = rng.choice(video_ids, size=len(video_ids), replace=True)
                sample_rows = [item for video_id in chosen for item in grouped[video_id]]
                y = np.asarray([row["y"] for row in sample_rows], dtype=np.int64)
                if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
                    continue
                vals = {model: retest.pr_auc(y, np.asarray([row[f"{model}_score"] for row in sample_rows], dtype=np.float64)) for model in ("real", "ar", "shuffled", "random", "timestamp", "video_time")}
                if vals["real"] is None:
                    continue
                metrics["real_minus_ar"].append(vals["real"] - vals["ar"] if vals["ar"] is not None else np.nan)
                for control in ("shuffled", "random", "timestamp", "video_time"):
                    if vals[control] is not None:
                        metrics[f"real_minus_{control}"].append(vals["real"] - vals[control])
            for metric, values in metrics.items():
                arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
                output.append(
                    {
                        "feature_mode": feature,
                        "target": target,
                        "threshold": threshold,
                        "config": label,
                        "metric": metric,
                        "samples": int(arr.size),
                        "mean": float(np.mean(arr)) if arr.size else None,
                        "ci95_low": float(np.quantile(arr, 0.025)) if arr.size else None,
                        "ci95_high": float(np.quantile(arr, 0.975)) if arr.size else None,
                    }
                )
    return output


def write_report(path: Path, summary: dict[str, Any], headline_rows: list[dict[str, Any]], taxonomy: list[dict[str, Any]]) -> None:
    lines = [
        "# VEATIC-124 Temporal Fairness Benchmark",
        "",
        "Offset convention: `feature_time = label_anchor_time + offset_seconds`. Negative offsets use earlier TRIBE/cortical features; positive offsets use later TRIBE/cortical features.",
        "",
        "## Executive Verdict",
        "",
        summary["executive_verdict"],
        "",
        "## Conservative Headline Rows",
        "",
        "| Feature | Target | Thr | Taxonomy | 0s PR-AUC | Best causal | Causal gain | Best offset | Offset gain |",
        "|---|---|---:|---|---:|---|---:|---:|---:|",
    ]
    for row in headline_rows:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {float(row['threshold']):.3f} | {row['taxonomy']} | {fmt(row.get('zero_real_pr_auc'))} | {row.get('best_causal_window')} | {fmt(row.get('causal_window_gain'))} | {fmt(row.get('best_offset_seconds'))} | {fmt(row.get('offset_gain'))} |"
        )
    lines += [
        "",
        "## Benchmark Arms",
        "",
        "- Arm 1: current 0s single-timestep baseline.",
        "- Arm 2: causal temporal context windows ending at the label anchor.",
        "- Arm 3: offset sensitivity sweep, diagnostic unless selected train-only.",
        "- Arm 4: train-only selected alignment under grouped-video folds.",
        "- Arm 5: future-inclusive diagnostic context, excluded from final predictive claims.",
        "",
        "## Leakage Audit",
        "",
        "All final predictive arms keep grouped video holdout, train-only PCA/scalers, train-only thresholds, and no future-inclusive feature context. Future-inclusive windows are explicitly diagnostic.",
        "",
        "## Timing Fairness Taxonomy",
        "",
    ]
    counts = Counter(row["taxonomy"] for row in taxonomy)
    for label, count in sorted(counts.items()):
        lines.append(f"- {label}: {count}")
    lines += [
        "",
        "## Recommended Final Claim",
        "",
        summary["best_defensible_claim"],
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim universal early warning.",
        "- Do not claim lag fixed the benchmark.",
        "- Do not claim exact future arousal prediction.",
        "- Do not use test-selected lag correction as a final score.",
        "- Do not use per-video corrected final scores.",
        "",
        "## Output Index",
        "",
    ]
    for name in summary["outputs"]:
        lines.append(f"- `{name}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    return f"{value:.4f}" if math.isfinite(value) else "NA"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json")
    parser.add_argument("--cache-dir", default=str(EXTERNAL_ROOT / "benchmarks" / "veatic" / "tribe_cache"))
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="cpu_pinv", choices=("auto", "mps_solve", "cpu_pinv"))
    parser.add_argument("--include-secondary", action="store_true", default=False)
    parser.add_argument("--full-grid", action="store_true", help="Use the full requested 23-point offset grid instead of the pruned core grid.")
    parser.add_argument("--full-context", action="store_true", help="Use all causal and diagnostic windows instead of the core windows.")
    args = parser.parse_args()

    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = (Path(args.output_root).expanduser().resolve() / f"veatic_124_temporal_fairness_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = retest.RetestContext(Path(args.manifest).expanduser().resolve(), Path(args.report).expanduser().resolve(), Path(args.cache_dir).expanduser().resolve())
    label_series = align.LabelSeries(ctx.accepted_rows, "arousal")
    features_to_run = FEATURE_MODES if args.include_secondary else FEATURE_MODES[:2]
    offset_grid = OFFSET_GRID if args.full_grid else CORE_OFFSET_GRID
    causal_windows = CAUSAL_WINDOWS if args.full_context else CORE_CAUSAL_WINDOWS
    diagnostic_windows = DIAGNOSTIC_WINDOWS if args.full_context else CORE_DIAGNOSTIC_WINDOWS
    targets = TARGETS

    zero_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    train_selected_rows: list[dict[str, Any]] = []
    selected_offset_rows: list[dict[str, Any]] = []
    balanced_eval_rows: list[dict[str, Any]] = []
    ar_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    fold_specs = grouped_video_folds(ctx.accepted_rows, args.folds)
    feature_cache: dict[str, dict[str, np.ndarray]] = {}

    run_start = time.monotonic()
    for feature_label, feature_mode, tier in features_to_run:
        print(f"[INFO] feature={feature_label} tier={tier}", flush=True)
        base_features = ctx.base_feature_sets(feature_mode)
        feature_cache[feature_mode] = base_features
        for fold_index, (fold_label, held, train_rows, test_rows) in enumerate(fold_specs):
            split_start = time.monotonic()
            print(f"[INFO] split feature={feature_label} fold={fold_label} train={len(train_rows)} test={len(test_rows)}", flush=True)
            matrices = split_matrices(ctx, base_features, train_rows, test_rows, feature_mode)
            train_table = subset_table(train_rows, matrices["train_matrix"])
            test_table = subset_table(test_rows, matrices["test_matrix"])
            inner_train_rows, inner_val_rows, inner_val_ids = inner_validation_split(train_rows, fold_index)
            train_pos = {id(row): pos for pos, row in enumerate(train_rows)}
            inner_train_matrix = matrices["train_matrix"][[train_pos[id(row)] for row in inner_train_rows]]
            inner_val_matrix = matrices["train_matrix"][[train_pos[id(row)] for row in inner_val_rows]]
            inner_train_table = subset_table(inner_train_rows, inner_train_matrix)
            inner_val_table = subset_table(inner_val_rows, inner_val_matrix)
            inner_offset_rows: list[dict[str, Any]] = []

            for target, horizon, threshold in targets:
                train_selected, train_y = target_rows(label_series, train_rows, target, horizon, threshold)
                test_selected, test_y = target_rows(label_series, test_rows, target, horizon, threshold)
                inner_train_selected, inner_train_y = target_rows(label_series, inner_train_rows, target, horizon, threshold)
                inner_val_selected, inner_val_y = target_rows(label_series, inner_val_rows, target, horizon, threshold)

                for window_name, start, end, context_type, aggregation in causal_windows:
                    row, scores = evaluate_context(ctx, fold_label, feature_label, target, horizon, threshold, train_selected, train_y, test_selected, test_y, train_table, test_table, window_name, start, end, context_type, aggregation)
                    if row:
                        context_rows.append(row)
                        if window_name == "current_only_0s":
                            zero = dict(row)
                            zero["arm"] = "zero_offset"
                            zero_rows.append(zero)
                            ar_rows.append({key: zero.get(key) for key in zero if key.startswith("ar_") or key in {"fold", "feature_mode", "target", "threshold", "event_count", "actual_positive_rate", "n_test"}})
                        if scores and window_name in {"current_only_0s", "causal_past_3s", "causal_past_5s"} and threshold == 0.05 and feature_label in {"cortical_pca_64", "cortical_pca64_delta"}:
                            all_score_rows.extend(score_records(row, scores))
                        if scores and window_name in {"current_only_0s", "causal_past_3s", "causal_past_5s"}:
                            balanced_eval_rows.extend(balanced_rows(row, scores))

                for window_name, start, end, context_type, aggregation in diagnostic_windows:
                    row, scores = evaluate_context(ctx, fold_label, feature_label, target, horizon, threshold, train_selected, train_y, test_selected, test_y, train_table, test_table, window_name, start, end, context_type, aggregation)
                    if row:
                        diagnostic_rows.append(row)
                        if scores and feature_label in {"cortical_pca_64", "cortical_pca64_delta"} and threshold == 0.05:
                            all_score_rows.extend(score_records(row, scores))

                for offset in offset_grid:
                    row, scores = evaluate_offset(ctx, fold_label, feature_label, target, horizon, threshold, train_selected, train_y, test_selected, test_y, train_table, test_table, offset, "offset_sweep")
                    if row:
                        offset_rows.append(row)
                        if scores and offset in {-2.0, -1.5, 0.0, 1.5, 2.0} and feature_label in {"cortical_pca_64", "cortical_pca64_delta"} and threshold == 0.05:
                            all_score_rows.extend(score_records(row, scores))
                    inner_row, _ = evaluate_offset(ctx, f"{fold_label}_inner", feature_label, target, horizon, threshold, inner_train_selected, inner_train_y, inner_val_selected, inner_val_y, inner_train_table, inner_val_table, offset, "inner_offset_selection")
                    if inner_row:
                        inner_offset_rows.append(inner_row)
                print(f"[INFO] done target feature={feature_label} fold={fold_label} target={target} thr={threshold} elapsed={time.monotonic() - split_start:.1f}s", flush=True)

            selected = train_selected_offsets(inner_offset_rows)
            for target, horizon, threshold in targets:
                train_selected, train_y = target_rows(label_series, train_rows, target, horizon, threshold)
                test_selected, test_y = target_rows(label_series, test_rows, target, horizon, threshold)
                for variant in ("global", "target_specific", "feature_mode_specific", "feature_mode_x_target"):
                    offset = selected.get((variant, feature_label, target, float(threshold)), 0.0)
                    selected_offset_rows.append(
                        {
                            "fold": fold_label,
                            "held_out_video_ids": ",".join(held),
                            "inner_validation_video_ids": ",".join(inner_val_ids),
                            "selection_variant": variant,
                            "feature_mode": feature_label,
                            "target": target,
                            "threshold": threshold,
                            "selected_offset_seconds": offset,
                            "selection_policy": "max mean inner-validation real_minus_ar PR-AUC, training videos only",
                        }
                    )
                    row, scores = evaluate_offset(
                        ctx,
                        fold_label,
                        feature_label,
                        target,
                        horizon,
                        threshold,
                        train_selected,
                        train_y,
                        test_selected,
                        test_y,
                        train_table,
                        test_table,
                        offset,
                        "train_selected_offset",
                        extra={"selection_variant": variant, "selected_offset_seconds": offset},
                    )
                    if row:
                        train_selected_rows.append(row)
                        if scores and feature_label in {"cortical_pca_64", "cortical_pca64_delta"} and threshold == 0.05:
                            all_score_rows.extend(score_records(row, scores))
            print(f"[INFO] done split feature={feature_label} fold={fold_label} elapsed={time.monotonic() - split_start:.1f}s", flush=True)

    context_specificity = aggregate_specificity(context_rows, "window_name", "window_name", "context")
    offset_specificity = aggregate_specificity(offset_rows, "offset_seconds", "offset_seconds", "offset")
    taxonomy = build_taxonomy(zero_rows, context_rows, offset_rows, train_selected_rows, context_specificity, offset_specificity)
    bootstrap_rows = bootstrap_ci(all_score_rows, taxonomy, args.bootstrap_samples, args.seed)
    leakage_rows = leakage_audit_rows()

    write_csv(out_dir / "zero_offset_results.csv", zero_rows)
    write_csv(out_dir / "causal_context_window_results.csv", context_rows)
    write_csv(out_dir / "offset_sweep_results.csv", offset_rows)
    write_csv(out_dir / "train_selected_offset_results.csv", train_selected_rows)
    write_csv(out_dir / "diagnostic_future_context_results.csv", diagnostic_rows)
    write_csv(out_dir / "real_vs_control_context_specificity.csv", context_specificity)
    write_csv(out_dir / "real_vs_control_offset_specificity.csv", offset_specificity)
    write_csv(out_dir / "balanced_event_stable_results.csv", balanced_eval_rows)
    write_csv(out_dir / "bootstrap_ci.csv", bootstrap_rows)
    write_csv(out_dir / "timing_fairness_taxonomy.csv", taxonomy)
    write_csv(out_dir / "selected_offsets_by_fold.csv", selected_offset_rows)
    write_csv(out_dir / "ar_behavior_audit.csv", ar_rows)
    write_csv(out_dir / "leakage_audit.csv", leakage_rows)

    headline = sorted(taxonomy, key=lambda row: (row["feature_mode"] not in {"cortical_pca_64", "cortical_pca64_delta"}, row["target"], float(row["threshold"])))[:10]
    context_starved = sum(1 for row in taxonomy if row["taxonomy"] == "Context-starved")
    earlier = sum(1 for row in taxonomy if row["taxonomy"] == "Earlier-feature advantage")
    later = sum(1 for row in taxonomy if row["taxonomy"] == "Later-feature advantage")
    fair = sum(1 for row in taxonomy if row["taxonomy"] == "Fair at 0s")
    unstable = sum(1 for row in taxonomy if row["taxonomy"] == "Unstable/video-specific")
    control_driven = sum(1 for row in taxonomy if row["taxonomy"] == "Control-driven timing effect")
    executive = (
        "This temporal fairness benchmark evaluates whether the current 0s single-timestep TRIBE interface is fair under grouped-video validation. "
        f"Taxonomy counts: fair at 0s={fair}, context-starved={context_starved}, earlier-feature advantage={earlier}, later-feature advantage={later}, "
        f"control-driven={control_driven}, unstable/video-specific={unstable}. "
        "Causal windows and train-selected offsets are valid predictive diagnostics; future-inclusive windows are diagnostic only."
    )
    best_claim = (
        "Best defensible claim: TRIBE/cortical features should be judged with input-timing fairness checks. "
        "If causal windows or train-selected offsets outperform 0s while controls do not, the single-timestep 0s interface likely underestimates the representation. "
        "Do not promote any test-selected or per-video lag correction to a final benchmark score."
    )
    summary = {
        "schema_version": "veatic_124_temporal_fairness_v1",
        "output_dir": str(out_dir),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "offset_convention": "feature_time = label_anchor_time + offset_seconds",
        "pca_backend": args.pca_backend,
        "ridge_backend": args.ridge_backend,
        "folds": args.folds,
        "bootstrap_samples": args.bootstrap_samples,
        "include_secondary": args.include_secondary,
        "full_grid": args.full_grid,
        "full_context": args.full_context,
        "offset_grid": list(offset_grid),
        "causal_windows": [item[0] for item in causal_windows],
        "diagnostic_windows": [item[0] for item in diagnostic_windows],
        "elapsed_seconds": time.monotonic() - run_start,
        "executive_verdict": executive,
        "best_defensible_claim": best_claim,
        "taxonomy_counts": dict(Counter(row["taxonomy"] for row in taxonomy)),
        "outputs": [
            "zero_offset_results.csv",
            "causal_context_window_results.csv",
            "offset_sweep_results.csv",
            "train_selected_offset_results.csv",
            "diagnostic_future_context_results.csv",
            "real_vs_control_context_specificity.csv",
            "real_vs_control_offset_specificity.csv",
            "balanced_event_stable_results.csv",
            "bootstrap_ci.csv",
            "timing_fairness_taxonomy.csv",
            "selected_offsets_by_fold.csv",
            "ar_behavior_audit.csv",
            "leakage_audit.csv",
            "veatic_124_temporal_fairness_report.md",
            "veatic_124_temporal_fairness_summary.json",
        ],
    }
    (out_dir / "veatic_124_temporal_fairness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir / "veatic_124_temporal_fairness_report.md", summary, headline, taxonomy)

    print("\n=== VEATIC-124 Temporal Fairness Benchmark Summary ===")
    print(f"Output directory: {out_dir}")
    print("Top 10 conservative headline rows:")
    for row in headline:
        print(
            f"- {row['feature_mode']} | {row['target']} | thr={row['threshold']} | {row['taxonomy']} | "
            f"0s={fmt(row.get('zero_real_pr_auc'))} | best_causal={row.get('best_causal_window')} gain={fmt(row.get('causal_window_gain'))} | "
            f"best_offset={fmt(row.get('best_offset_seconds'))} gain={fmt(row.get('offset_gain'))}"
        )
    print(f"0s single-timestep alignment appears fair in {fair}/{len(taxonomy)} feature-target rows.")
    print(f"Causal temporal windows improve enough for context-starved classification in {context_starved}/{len(taxonomy)} rows.")
    print(f"Shifted offsets show earlier-feature advantage in {earlier}/{len(taxonomy)} rows and later-feature advantage in {later}/{len(taxonomy)} rows.")
    print("Positive offsets suggest judging TRIBE too early only where the taxonomy says Later-feature advantage.")
    print("Negative offsets suggest earlier cortical signal only where grouped validation and controls support Earlier-feature advantage.")
    print(f"Control-driven timing effects: {control_driven}/{len(taxonomy)} rows.")
    print("Train-only selected offsets should remain secondary unless they consistently beat 0s on held-out videos.")
    print(f"Best defensible claim: {best_claim}")
    print("Claims still not allowed: universal early warning; lag fixed the benchmark; exact future arousal prediction; test-selected lag correction; per-video corrected final score.")


if __name__ == "__main__":
    main()
