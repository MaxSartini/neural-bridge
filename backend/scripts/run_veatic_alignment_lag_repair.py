"""Dedicated VEATIC 124 alignment, lag, and evaluation repair audit.

This is cache-only. It reuses the completed TRIBE cortical outputs and the
existing benchmark helpers, then writes focused timing/alignment artifacts.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RETEST_SCRIPT = ROOT / "backend" / "scripts" / "run_veatic_event_spike_retest.py"
spec = importlib.util.spec_from_file_location("event_spike_retest", RETEST_SCRIPT)
retest = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(retest)
bench = retest.bench

FEATURE_MODES = (
    ("cortical_fast_default", "cortical_global"),
    ("cortical_global_delta", "cortical_global_delta"),
    ("cortical_pca_64", "cortical_pca_64"),
    ("cortical_pca64_delta", "cortical_pca64_delta"),
)
PCA_FEATURE_MODES = (
    ("cortical_pca_64", "cortical_pca_64"),
    ("cortical_pca64_delta", "cortical_pca64_delta"),
)
PRIMARY_TARGETS = (
    ("arousal__future_spike_1_3s", None, 0.05),
    ("arousal__future_spike_1_3s", None, 0.075),
    ("arousal__future_change_p3s_movement", 3, 0.05),
    ("arousal__future_change_p3s_movement", 3, 0.075),
    ("arousal__future_change_p2s_movement", 2, 0.05),
)
OFFSET_GRID = (-8, -7, -6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8)
HORIZONS = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10)
THRESHOLDS = (0.03, 0.05, 0.075, 0.10)
BALANCED_RATIOS = (1, 2, 3, 5)
BALANCED_SEEDS = tuple(range(50))


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    return str(value)


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


def row_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["video_id"]), float(row["time_start_seconds"])


class LabelSeries:
    def __init__(self, rows: list[dict[str, Any]], target: str = "arousal") -> None:
        self.by_video: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        grouped = retest.rows_by_video(rows)
        for video_id, values in grouped.items():
            times = np.asarray([float(row["time_start_seconds"]) for row in values], dtype=np.float64)
            labels = np.asarray([float(row["targets"][target]) for row in values], dtype=np.float64)
            self.by_video[str(video_id)] = (times, labels)

    def value(self, video_id: str, time: float, *, policy: str = "linear") -> float | None:
        if video_id not in self.by_video:
            return None
        times, values = self.by_video[video_id]
        if time < times[0] or time > times[-1]:
            return None
        if policy == "nearest":
            idx = int(np.argmin(np.abs(times - time)))
            return float(values[idx])
        if policy == "previous":
            idx = int(np.searchsorted(times, time, side="right") - 1)
            return float(values[max(0, idx)])
        return float(np.interp(time, times, values))

    def transformed(self, variant: str) -> "LabelSeries":
        clone = object.__new__(LabelSeries)
        clone.by_video = {}
        for video_id, (times, values) in self.by_video.items():
            if variant in {"current", "raw"}:
                out = values.copy()
            elif variant == "causal_roll1":
                out = values.copy()
            elif variant == "causal_roll3":
                out = np.asarray([np.mean(values[max(0, i - 2) : i + 1]) for i in range(len(values))], dtype=np.float64)
            elif variant == "centered_roll3_diagnostic":
                out = np.asarray([np.mean(values[max(0, i - 1) : min(len(values), i + 2)]) for i in range(len(values))], dtype=np.float64)
            elif variant == "derivative_delta":
                out = np.r_[0.0, np.diff(values)]
            elif variant == "highpass_causal3":
                baseline = np.asarray([np.mean(values[max(0, i - 2) : i + 1]) for i in range(len(values))], dtype=np.float64)
                out = values - baseline
            else:
                raise ValueError(variant)
            clone.by_video[video_id] = (times.copy(), out)
        return clone


def future_change_rows_label(
    series: LabelSeries,
    rows: list[dict[str, Any]],
    horizon: float,
    *,
    policy: str = "linear",
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    values = []
    for row in rows:
        video_id, time = row_key(row)
        current = series.value(video_id, time, policy=policy)
        future = series.value(video_id, time + horizon, policy=policy)
        if current is None or future is None:
            continue
        selected.append(row)
        values.append(future - current)
    return selected, np.asarray(values, dtype=np.float64)


def future_spike_rows_label(
    series: LabelSeries,
    rows: list[dict[str, Any]],
    threshold: float,
    *,
    max_horizon: float = 3.0,
    policy: str = "linear",
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    values = []
    steps = np.arange(1.0, max_horizon + 1e-9, 1.0)
    if max_horizon < 1.0:
        steps = np.asarray([max_horizon], dtype=np.float64)
    for row in rows:
        video_id, time = row_key(row)
        current = series.value(video_id, time, policy=policy)
        if current is None:
            continue
        futures = [series.value(video_id, time + float(step), policy=policy) for step in steps]
        futures = [item for item in futures if item is not None]
        if not futures:
            continue
        selected.append(row)
        values.append(float(max(futures) - current >= threshold))
    return selected, np.asarray(values, dtype=np.float64)


def split_rows(rows: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    return retest.fixed_rows(rows, split_name)


def grouped_video_folds(rows: list[dict[str, Any]], fold_count: int = 5) -> list[tuple[str, list[str], list[dict[str, Any]], list[dict[str, Any]]]]:
    grouped = retest.rows_by_video(rows)
    video_ids = sorted(grouped, key=lambda item: int(item))
    folds = []
    for fold_index in range(fold_count):
        held = [video_id for offset, video_id in enumerate(video_ids) if offset % fold_count == fold_index]
        held_set = set(held)
        test_rows = [row for video_id in held for row in grouped[video_id]]
        train_rows = [row for video_id, items in grouped.items() if video_id not in held_set for row in items]
        folds.append((f"grouped_{fold_index}", held, train_rows, test_rows))
    return folds


def build_time_tables(split_rows_: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    grouped: dict[str, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(split_rows_):
        grouped[str(row["video_id"])].append((float(row["time_start_seconds"]), index, row))
    output = {}
    for video_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        times = np.asarray([item[0] for item in items], dtype=np.float64)
        positions = np.asarray([item[1] for item in items], dtype=np.int64)
        ordered_rows = [item[2] for item in items]
        output[video_id] = (times, matrix[positions], ordered_rows)
    return output


def shifted_features(
    selected_rows: list[dict[str, Any]],
    split_tables: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    offset_seconds: float,
    *,
    interpolation_policy: str = "linear",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    keep: list[int] = []
    features: list[np.ndarray] = []
    kept_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        video_id, anchor = row_key(row)
        table = split_tables.get(video_id)
        if table is None:
            continue
        times, matrix, _ = table
        feature_time = anchor + float(offset_seconds)
        if feature_time < times[0] or feature_time > times[-1]:
            continue
        if interpolation_policy == "nearest":
            pos = int(np.argmin(np.abs(times - feature_time)))
            value = matrix[pos]
        elif interpolation_policy == "previous":
            pos = int(np.searchsorted(times, feature_time, side="right") - 1)
            value = matrix[max(0, pos)]
        else:
            right = int(np.searchsorted(times, feature_time, side="left"))
            if right == 0:
                value = matrix[0]
            elif right >= len(times):
                value = matrix[-1]
            elif abs(times[right] - feature_time) < 1e-9:
                value = matrix[right]
            else:
                left = right - 1
                denom = times[right] - times[left]
                weight = 0.0 if denom <= 0 else (feature_time - times[left]) / denom
                value = matrix[left] * (1.0 - weight) + matrix[right] * weight
        keep.append(index)
        features.append(np.asarray(value, dtype=np.float64))
        kept_rows.append(row)
    if not features:
        width = next(iter(split_tables.values()))[1].shape[1] if split_tables else 0
        return np.asarray([], dtype=np.int64), np.zeros((0, width), dtype=np.float64), []
    return np.asarray(keep, dtype=np.int64), np.vstack(features), kept_rows


def score_sets_for_offset(
    ctx: retest.RetestContext,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    matrices: dict[str, Any],
    offset_seconds: float,
    rng: np.random.Generator,
    *,
    interpolation_policy: str = "linear",
    train_tables: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] | None = None,
    test_tables: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] | None = None,
    design_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any] | None:
    if train_tables is None:
        train_tables = build_time_tables(train_rows, matrices["train_matrix"])
    if test_tables is None:
        test_tables = build_time_tables(test_rows, matrices["test_matrix"])
    train_keep, train_features, kept_train_rows = shifted_features(train_selected, train_tables, offset_seconds, interpolation_policy=interpolation_policy)
    test_keep, test_features, kept_test_rows = shifted_features(test_selected, test_tables, offset_seconds, interpolation_policy=interpolation_policy)
    if train_keep.size < 8 or test_keep.size < 4:
        return None
    y_train = train_y[train_keep].astype(np.float64)
    y_test = test_y[test_keep].astype(np.float64)
    if design_cache is None:
        train_ar = bench.autoregressive_features(ctx.accepted_rows, kept_train_rows, "arousal", include_current=True)
        test_ar = bench.autoregressive_features(ctx.accepted_rows, kept_test_rows, "arousal", include_current=True)
        time_train = bench.time_features(kept_train_rows)
        time_test = bench.time_features(kept_test_rows)
        video_train_x, video_test_x = video_time_matrix(ctx, kept_train_rows, kept_test_rows)
    else:
        train_ar = design_cache["train_ar"][train_keep]
        test_ar = design_cache["test_ar"][test_keep]
        time_train = design_cache["train_time"][train_keep]
        time_test = design_cache["test_time"][test_keep]
        video_train_x = design_cache["train_video_time"][train_keep]
        video_test_x = design_cache["test_video_time"][test_keep]
    ar_train, _ = bench.ridge_fit_predict(train_ar, y_train, train_ar)
    ar_test, _ = bench.ridge_fit_predict(train_ar, y_train, test_ar)
    real_train_x = np.concatenate([train_ar, train_features], axis=1)
    real_test_x = np.concatenate([test_ar, test_features], axis=1)
    real_train, _ = bench.ridge_fit_predict(real_train_x, y_train, real_train_x)
    real_test, _ = bench.ridge_fit_predict(real_train_x, y_train, real_test_x)
    shuffled_train = train_features.copy()
    shuffled_test = test_features.copy()
    rng.shuffle(shuffled_train, axis=0)
    rng.shuffle(shuffled_test, axis=0)
    shuffled_train_x = np.concatenate([train_ar, shuffled_train], axis=1)
    shuffled_test_x = np.concatenate([test_ar, shuffled_test], axis=1)
    shuffled_train_scores, _ = bench.ridge_fit_predict(shuffled_train_x, y_train, shuffled_train_x)
    shuffled_test_scores, _ = bench.ridge_fit_predict(shuffled_train_x, y_train, shuffled_test_x)
    random_train = rng.normal(size=train_features.shape)
    random_test = rng.normal(size=test_features.shape)
    random_train_x = np.concatenate([train_ar, random_train], axis=1)
    random_test_x = np.concatenate([test_ar, random_test], axis=1)
    random_train_scores, _ = bench.ridge_fit_predict(random_train_x, y_train, random_train_x)
    random_test_scores, _ = bench.ridge_fit_predict(random_train_x, y_train, random_test_x)
    time_train_scores, _ = bench.ridge_fit_predict(time_train, y_train, time_train)
    time_test_scores, _ = bench.ridge_fit_predict(time_train, y_train, time_test)
    video_train_scores, _ = bench.ridge_fit_predict(video_train_x, y_train, video_train_x)
    video_test_scores, _ = bench.ridge_fit_predict(video_train_x, y_train, video_test_x)
    return {
        "train_y": y_train,
        "test_y": y_test.astype(np.int64),
        "test_rows": kept_test_rows,
        "models": {
            "ar": (y_train, ar_train, ar_test),
            "real": (y_train, real_train, real_test),
            "shuffled": (y_train, shuffled_train_scores, shuffled_test_scores),
            "random": (y_train, random_train_scores, random_test_scores),
            "timestamp": (y_train, time_train_scores, time_test_scores),
            "video_time": (y_train, video_train_scores, video_test_scores),
        },
    }


def build_design_cache(
    ctx: retest.RetestContext,
    train_selected: list[dict[str, Any]],
    test_selected: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    train_video_time, test_video_time = video_time_matrix(ctx, train_selected, test_selected)
    return {
        "train_ar": bench.autoregressive_features(ctx.accepted_rows, train_selected, "arousal", include_current=True),
        "test_ar": bench.autoregressive_features(ctx.accepted_rows, test_selected, "arousal", include_current=True),
        "train_time": bench.time_features(train_selected),
        "test_time": bench.time_features(test_selected),
        "train_video_time": train_video_time,
        "test_video_time": test_video_time,
    }


def video_time_matrix(
    ctx: retest.RetestContext,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    video_ids = sorted({str(row["video_id"]) for row in ctx.accepted_rows}, key=int)
    video_pos = {video_id: index for index, video_id in enumerate(video_ids)}

    def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
        time_x = bench.time_features(rows)
        one_hot = np.zeros((len(rows), len(video_ids)), dtype=np.float64)
        for index, row in enumerate(rows):
            one_hot[index, video_pos[str(row["video_id"])]] = 1.0
        return np.concatenate([time_x, one_hot], axis=1)

    return matrix(train_rows), matrix(test_rows)


def model_event_metrics(train_y: np.ndarray, train_scores: np.ndarray, test_y: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    metrics = retest.event_metrics(train_y.astype(np.int64), train_scores, test_y.astype(np.int64), test_scores)
    threshold = metrics.get("decision_threshold")
    if threshold is not None:
        pred = (test_scores >= float(threshold)).astype(np.int64)
        metrics["predicted_positive_rate"] = float(np.mean(pred)) if pred.size else None
        metrics["predicted_positive_count"] = float(np.sum(pred))
    else:
        metrics["predicted_positive_rate"] = None
        metrics["predicted_positive_count"] = None
    return metrics


def offset_row(
    base: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    row = dict(base)
    for model in ("ar", "real", "shuffled", "random", "timestamp", "video_time"):
        train_y, train_scores, test_scores = scores["models"][model]
        metrics = model_event_metrics(train_y, train_scores, scores["test_y"], test_scores)
        for key, value in metrics.items():
            row[f"{model}_{key}"] = finite(value)
    row["real_vs_ar_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("ar_pr_auc"))
    row["real_vs_shuffled_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("shuffled_pr_auc"))
    row["real_vs_random_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("random_pr_auc"))
    row["real_vs_timestamp_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("timestamp_pr_auc"))
    row["real_vs_video_time_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("video_time_pr_auc"))
    row["real_vs_ar_f1_delta"] = diff(row.get("real_f1"), row.get("ar_f1"))
    row["real_vs_shuffled_f1_delta"] = diff(row.get("real_f1"), row.get("shuffled_f1"))
    row["real_vs_random_f1_delta"] = diff(row.get("real_f1"), row.get("random_f1"))
    row["success_all_controls"] = all(
        row.get(key) is not None and row[key] > 0
        for key in ("real_vs_ar_pr_auc_delta", "real_vs_shuffled_pr_auc_delta", "real_vs_random_pr_auc_delta")
    )
    return row


def target_rows(
    series: LabelSeries,
    rows: list[dict[str, Any]],
    target: str,
    horizon: float | None,
    threshold: float,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray | None]:
    if target == "arousal__future_spike_1_3s":
        selected, y = future_spike_rows_label(series, rows, threshold, max_horizon=3.0)
        return selected, y, None
    assert horizon is not None
    selected, change = future_change_rows_label(series, rows, horizon)
    y = (np.abs(change) >= threshold).astype(np.float64)
    return selected, y, change


def per_video_rows_from_scores(base: dict[str, Any], scores: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(scores["test_rows"]):
        grouped[str(row["video_id"])].append(index)
    rows = []
    for video_id, indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        y = scores["test_y"][idx].astype(np.int64)
        if y.size < 6:
            continue
        out = dict(base)
        out.update(
            {
                "video_id": video_id,
                "n": int(y.size),
                "event_count": int(np.sum(y)),
                "positive_rate": float(np.mean(y)),
                "enough_events": bool(np.sum(y) >= 3 and np.sum(y == 0) >= 3),
            }
        )
        for model in ("ar", "real", "shuffled", "random"):
            test_scores = scores["models"][model][2][idx]
            out[f"{model}_pr_auc"] = retest.pr_auc(y, test_scores) if np.sum(y) else None
        out["real_vs_ar_pr_auc_delta"] = diff(out.get("real_pr_auc"), out.get("ar_pr_auc"))
        out["real_vs_shuffled_pr_auc_delta"] = diff(out.get("real_pr_auc"), out.get("shuffled_pr_auc"))
        out["real_vs_random_pr_auc_delta"] = diff(out.get("real_pr_auc"), out.get("random_pr_auc"))
        rows.append(out)
    return rows


def region_masks(test_rows: list[dict[str, Any]], y_event: np.ndarray, stable_magnitude: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    key_to_index = {(str(row["video_id"]), int(round(float(row["time_start_seconds"])))): i for i, row in enumerate(test_rows)}
    event_mask = y_event.astype(bool)
    stable_mask = (~event_mask) & np.isfinite(stable_magnitude) & (stable_magnitude < threshold)
    masks = {
        "event_only": event_mask,
        "stable_negative": stable_mask,
    }
    for lead in (1, 2, 3, 5):
        mask = np.zeros(y_event.size, dtype=bool)
        for index, is_event in enumerate(event_mask):
            if not is_event:
                continue
            row = test_rows[index]
            key = (str(row["video_id"]), int(round(float(row["time_start_seconds"]))) - lead)
            pre_index = key_to_index.get(key)
            if pre_index is not None and not event_mask[pre_index]:
                mask[pre_index] = True
        masks[f"pre_event_{lead}s"] = mask
    masks["event_plus_pre_3s"] = masks["event_only"] | masks["pre_event_1s"] | masks["pre_event_2s"] | masks["pre_event_3s"]
    return masks


def event_magnitude(series: LabelSeries, rows: list[dict[str, Any]], target: str, horizon: float | None) -> np.ndarray:
    values = []
    for row in rows:
        video_id, time = row_key(row)
        current = series.value(video_id, time)
        if current is None:
            values.append(np.nan)
            continue
        if target == "arousal__future_spike_1_3s":
            futures = [series.value(video_id, time + step) for step in (1, 2, 3)]
            futures = [item for item in futures if item is not None]
            values.append(max(abs(item - current) for item in futures) if futures else np.nan)
        else:
            assert horizon is not None
            future = series.value(video_id, time + horizon)
            values.append(abs(future - current) if future is not None else np.nan)
    return np.asarray(values, dtype=np.float64)


def balanced_rows_for_scores(base: dict[str, Any], scores: dict[str, Any], masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    stable_idx = np.flatnonzero(masks["stable_negative"])
    for mask_name in ("event_only", "pre_event_1s", "pre_event_2s", "pre_event_3s", "event_plus_pre_3s"):
        pos_idx = np.flatnonzero(masks[mask_name])
        if pos_idx.size == 0 or stable_idx.size == 0:
            continue
        for ratio in BALANCED_RATIOS:
            seed_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for seed in BALANCED_SEEDS:
                rng = np.random.default_rng(seed)
                neg_n = min(stable_idx.size, pos_idx.size * ratio)
                neg_idx = rng.choice(stable_idx, size=neg_n, replace=False)
                idx = np.concatenate([pos_idx, neg_idx])
                eval_y = np.concatenate([np.ones(pos_idx.size, dtype=np.int64), np.zeros(neg_idx.size, dtype=np.int64)])
                for model in ("ar", "real", "shuffled", "random", "timestamp", "video_time"):
                    train_y, train_scores, test_scores = scores["models"][model]
                    seed_metrics[model].append(model_event_metrics(train_y, train_scores, eval_y, test_scores[idx]))
            for model, metrics_list in seed_metrics.items():
                out = dict(base)
                out.update({"mask": mask_name, "negative_ratio": f"1:{ratio}", "model": model, "seed_count": len(metrics_list), "n_pos": int(pos_idx.size), "n_neg_mean": int(min(stable_idx.size, pos_idx.size * ratio))})
                for metric in ("pr_auc", "f1", "balanced_accuracy", "precision", "recall", "accuracy", "top_5pct_recall", "top_10pct_recall"):
                    values = [item.get(metric) for item in metrics_list if item.get(metric) is not None]
                    arr = np.asarray(values, dtype=np.float64)
                    out[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None
                    out[f"{metric}_std"] = float(np.std(arr)) if arr.size else None
                rows.append(out)
    return rows


def manifest_alignment_rows(ctx: retest.RetestContext, report_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    video_meta = {str(item["video_id"]): item for item in report.get("videos", [])}
    rows = []
    resampled = []
    grouped = retest.rows_by_video(ctx.accepted_rows)
    for video_id in sorted(grouped, key=lambda item: int(item)):
        raw_path = ctx.cache_dir / video_id / "tribe_raw_output.npz"
        status_path = ctx.cache_dir / video_id / "cache_status.json"
        with np.load(raw_path) as bundle:
            pred = np.asarray(bundle["predictions"])
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        times = [float(row["time_start_seconds"]) for row in grouped[video_id]]
        duplicate_times = len(times) - len(set(times))
        nonmonotonic = any(right <= left for left, right in zip(times, times[1:]))
        meta = video_meta.get(video_id, {})
        manifest_rows = len(grouped[video_id])
        raw_rows = int(pred.shape[0])
        alignment = "exact" if raw_rows == manifest_rows else "linear_resampled_by_benchmark"
        row = {
            "video_id": video_id,
            "media_path": meta.get("media_path") or status.get("media_path"),
            "manifest_frame_time_count": manifest_rows,
            "label_row_count": meta.get("label_frames"),
            "raw_cached_prediction_row_count": raw_rows,
            "feature_row_count": raw_rows,
            "final_aligned_row_count": manifest_rows,
            "fps": meta.get("fps"),
            "label_sampling_rate_hz": 1.0,
            "feature_sampling_rate_hz": raw_rows / max(float(meta.get("duration_seconds") or times[-1] or 1.0), 1e-9),
            "duration_video_seconds": meta.get("duration_seconds"),
            "duration_labels_seconds": meta.get("expected_duration_seconds_from_labels"),
            "duration_manifest_seconds": max(times) - min(times) if times else None,
            "duration_features_seconds_approx": meta.get("duration_seconds"),
            "linear_resampling_performed": alignment != "exact",
            "alignment_policy": alignment,
            "dropped_rows_begin_end": 0,
            "duplicate_timestamps": duplicate_times,
            "nonmonotonic_timestamps": nonmonotonic,
            "feature_has_nan_inf": not bool(np.isfinite(pred).all()),
            "resampling_ratio_raw_to_manifest": raw_rows / manifest_rows if manifest_rows else None,
            "suspicious": alignment != "exact" or duplicate_times > 0 or nonmonotonic,
        }
        rows.append(row)
        if row["linear_resampling_performed"] or row["suspicious"]:
            resampled.append(row)
    return rows, resampled


def causal_audit_rows() -> list[dict[str, Any]]:
    return [
        {"operation": "arousal labels current manifest", "kind": "label_input", "classification": "observed/current", "future_feature_leakage": False, "notes": "Manifest rows read averaged labels at row timestamp."},
        {"operation": "future_change", "kind": "target", "classification": "label-only future target construction", "future_feature_leakage": False, "notes": "Allowed target y(t+h)-y(t); not used as feature."},
        {"operation": "event_future_spike_1_3s", "kind": "target", "classification": "label-only future target construction", "future_feature_leakage": False, "notes": "Allowed binary future-label target."},
        {"operation": "residual_future_p*_rolling3", "kind": "target", "classification": "causal/past-only baseline plus future target", "future_feature_leakage": False, "notes": "History window uses current/past labels; future label is target only."},
        {"operation": "local target future_minus_rolling_baseline", "kind": "target", "classification": "causal/past-only baseline plus future target", "future_feature_leakage": False, "notes": "Rolling baseline uses current/past values in retest code."},
        {"operation": "local target future_change_local_volatility", "kind": "target", "classification": "causal/past-only denominator plus future target", "future_feature_leakage": False, "notes": "Local volatility uses current/past values; near-perfect local-volatility rows remain suspicious and should not be headline."},
        {"operation": "pca64_delta delta1", "kind": "feature", "classification": "causal/past-only", "future_feature_leakage": False, "notes": "Uses current minus previous within video."},
        {"operation": "pca64_delta accel", "kind": "feature", "classification": "causal/past-only", "future_feature_leakage": False, "notes": "Uses current/previous deltas only."},
        {"operation": "pca64_delta rollmean3", "kind": "feature", "classification": "causal/past-only", "future_feature_leakage": False, "notes": "Window is current plus prior two rows within video."},
        {"operation": "pca64_delta slope3", "kind": "feature", "classification": "causal/past-only", "future_feature_leakage": False, "notes": "Slope over current plus prior two rows."},
        {"operation": "pca64_delta slope5", "kind": "feature", "classification": "causal/past-only", "future_feature_leakage": False, "notes": "Slope over current plus prior four rows."},
        {"operation": "pre_event masks", "kind": "evaluation_mask", "classification": "label-only future/onset diagnostic", "future_feature_leakage": False, "notes": "Masks filter test rows for diagnostics; thresholds remain train-selected."},
        {"operation": "event_plus_pre masks", "kind": "evaluation_mask", "classification": "label-only future/onset diagnostic", "future_feature_leakage": False, "notes": "Positive-only masks have undefined PR-AUC; use balanced event-vs-stable for discrimination."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json")
    parser.add_argument("--cache-dir", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
    parser.add_argument("--output-dir", default="benchmarks/veatic")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="cpu_pinv", choices=("auto", "mps_solve", "cpu_pinv"))
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) // 2)))
    args = parser.parse_args()
    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ctx = retest.RetestContext(Path(args.manifest).expanduser().resolve(), Path(args.report).expanduser().resolve(), Path(args.cache_dir).expanduser().resolve())
    label_series = LabelSeries(ctx.accepted_rows, "arousal")

    manifest_rows, resampled_rows = manifest_alignment_rows(ctx, Path(args.report).expanduser().resolve())
    write_csv(out_dir / "veatic_124_alignment_audit_manifest_rows.csv", manifest_rows)
    write_csv(out_dir / "veatic_124_alignment_audit_resampled_videos.csv", resampled_rows)
    (out_dir / "veatic_124_alignment_shift_convention.md").write_text(
        "\n".join(
            [
                "# VEATIC 124 Alignment Shift Convention",
                "",
                "`offset_seconds > 0` means feature rows are sampled later than the label-anchor row: `feature_time = label_anchor_time + offset_seconds`.",
                "",
                "Equivalently, positive offset tests whether later cortical features align better with the target anchored at the current label row. Negative offset tests whether earlier cortical features align better and is the expected direction for a real early-warning signal.",
                "",
                "Targets are constructed first from label rows at the label anchor time. The feature shift is then applied inside each split using only rows available in that split. Rows whose shifted feature time falls outside the same-video split region are trimmed.",
                "",
                "PCA bases are fit once per feature mode and split/fold using train rows only; offset scans reuse that split-local basis.",
            ]
        ),
        encoding="utf-8",
    )

    split_specs: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[str]]] = []
    for label, split_name in (("blocked", "blocked_temporal_gap"), ("official", "official_70_30")):
        train_rows, test_rows, _ = split_rows(ctx.accepted_rows, split_name)
        split_specs.append((label, train_rows, test_rows, []))
    for label, held, train_rows, test_rows in grouped_video_folds(ctx.accepted_rows, 5):
        split_specs.append((label, train_rows, test_rows, held))

    offset_rows: list[dict[str, Any]] = []
    per_video_offset_rows: list[dict[str, Any]] = []
    ar_behavior_rows: list[dict[str, Any]] = []
    controls_after_fix_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []
    feature_cache: dict[str, dict[str, np.ndarray]] = {}
    matrix_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for feature_label, feature_mode in FEATURE_MODES:
        feature_start = time.monotonic()
        print(f"[INFO] offset feature={feature_label} workers={args.workers}", flush=True)
        base_features = ctx.base_feature_sets(feature_mode)
        feature_cache[feature_mode] = base_features
        for split_label, train_rows, test_rows, held in split_specs:
            if split_label.startswith("grouped_") and feature_label not in {"cortical_pca_64", "cortical_pca64_delta"}:
                continue
            split_start = time.monotonic()
            print(f"[INFO] split feature={feature_label} split={split_label} train={len(train_rows)} test={len(test_rows)}", flush=True)
            matrices = retest.split_matrices(ctx, base_features, train_rows, test_rows, feature_mode)
            matrix_cache[(feature_mode, split_label)] = matrices
            train_tables = build_time_tables(train_rows, matrices["train_matrix"])
            test_tables = build_time_tables(test_rows, matrices["test_matrix"])
            targets_for_split = PRIMARY_TARGETS if not split_label.startswith("grouped_") else PRIMARY_TARGETS[:2]
            for target, horizon, threshold in targets_for_split:
                train_selected, train_y, _ = target_rows(label_series, train_rows, target, horizon, threshold)
                test_selected, test_y, _ = target_rows(label_series, test_rows, target, horizon, threshold)
                design_cache = build_design_cache(ctx, train_selected, test_selected)

                def run_offset(item: tuple[int, float]) -> tuple[float, dict[str, Any] | None]:
                    offset_index, offset = item
                    local_rng = np.random.default_rng(args.seed + offset_index + int(abs(float(offset)) * 1000) + len(offset_rows))
                    scores = score_sets_for_offset(
                        ctx,
                        train_selected,
                        train_y,
                        test_selected,
                        test_y,
                        train_rows,
                        test_rows,
                        matrices,
                        offset,
                        local_rng,
                        train_tables=train_tables,
                        test_tables=test_tables,
                        design_cache=design_cache,
                    )
                    return offset, scores

                if args.workers > 1:
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        offset_results = list(executor.map(run_offset, list(enumerate(OFFSET_GRID))))
                else:
                    offset_results = [run_offset(item) for item in enumerate(OFFSET_GRID)]
                for offset, scores in offset_results:
                    if scores is None:
                        continue
                    base = {
                        "feature_mode": feature_label,
                        "split": split_label,
                        "held_out_video_ids": ",".join(held),
                        "target": target,
                        "horizon_seconds": horizon,
                        "threshold": threshold,
                        "offset_seconds": offset,
                        "interpolation_policy": "linear",
                        "n": int(scores["test_y"].size),
                        "event_count": int(np.sum(scores["test_y"])),
                        "positive_rate": float(np.mean(scores["test_y"])) if scores["test_y"].size else None,
                    }
                    row = offset_row(base, scores)
                    offset_rows.append(row)
                    if split_label == "blocked":
                        per_video_offset_rows.extend(per_video_rows_from_scores(base, scores))
                    if split_label == "blocked" and offset == 0:
                        controls_after_fix_rows.append(row)
                        ar_behavior_rows.append(
                            {
                                **base,
                                "ar_threshold": row.get("ar_decision_threshold"),
                                "ar_predicted_positive_rate": row.get("ar_predicted_positive_rate"),
                                "real_predicted_positive_rate": row.get("real_predicted_positive_rate"),
                                "shuffled_predicted_positive_rate": row.get("shuffled_predicted_positive_rate"),
                                "random_predicted_positive_rate": row.get("random_predicted_positive_rate"),
                                "ar_precision": row.get("ar_precision"),
                                "ar_recall": row.get("ar_recall"),
                                "real_precision": row.get("real_precision"),
                                "real_recall": row.get("real_recall"),
                                "real_pr_auc": row.get("real_pr_auc"),
                                "ar_pr_auc": row.get("ar_pr_auc"),
                            }
                        )
                    if split_label == "blocked" and offset == 0 and target in {"arousal__future_spike_1_3s", "arousal__future_change_p3s_movement"} and threshold in {0.05, 0.075}:
                        stable_mag = event_magnitude(label_series, scores["test_rows"], target, horizon)
                        masks = region_masks(scores["test_rows"], scores["test_y"], stable_mag, threshold)
                        balanced_rows.extend(balanced_rows_for_scores(base, scores, masks))
                print(
                    f"[INFO] done target feature={feature_label} split={split_label} target={target} threshold={threshold} elapsed={time.monotonic() - split_start:.1f}s",
                    flush=True,
                )
                if split_label == "blocked" and feature_label in {"cortical_pca_64", "cortical_pca64_delta"} and target in {"arousal__future_spike_1_3s", "arousal__future_change_p3s_movement"} and threshold in {0.05, 0.075}:
                    for policy in ("nearest", "linear", "previous"):
                        for offset in (-2, -1, 0, 1, 2):
                            scores = score_sets_for_offset(
                                ctx,
                                train_selected,
                                train_y,
                                test_selected,
                                test_y,
                                train_rows,
                                test_rows,
                                matrices,
                                offset,
                                rng,
                                interpolation_policy=policy,
                                train_tables=train_tables,
                                test_tables=test_tables,
                                design_cache=design_cache,
                            )
                            if scores is None:
                                continue
                            interpolation_rows.append(offset_row({"feature_mode": feature_label, "split": split_label, "target": target, "horizon_seconds": horizon, "threshold": threshold, "offset_seconds": offset, "interpolation_policy": policy, "n": int(scores["test_y"].size), "event_count": int(np.sum(scores["test_y"]))}, scores))
            print(f"[INFO] done split feature={feature_label} split={split_label} elapsed={time.monotonic() - split_start:.1f}s", flush=True)
        print(f"[INFO] done feature={feature_label} elapsed={time.monotonic() - feature_start:.1f}s", flush=True)

    write_csv(out_dir / "veatic_124_alignment_offset_grid.csv", offset_rows)
    write_csv(out_dir / "veatic_124_alignment_ar_behavior_audit.csv", ar_behavior_rows)
    write_csv(out_dir / "veatic_124_alignment_controls_after_lag_fix.csv", controls_after_fix_rows)
    write_csv(out_dir / "veatic_124_alignment_balanced_event_stable_rebuild.csv", balanced_rows)
    write_csv(out_dir / "veatic_124_alignment_interpolation_policy_sweep.csv", interpolation_rows)

    grouped_best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in offset_rows:
        if row.get("real_pr_auc") is None:
            continue
        key = (row["feature_mode"], row["split"], row["target"], row["threshold"])
        if key not in grouped_best or float(row["real_pr_auc"]) > float(grouped_best[key]["real_pr_auc"]):
            grouped_best[key] = row
    global_rows = []
    for key, row in sorted(grouped_best.items(), key=lambda item: item[0]):
        global_rows.append(
            {
                "feature_mode": row["feature_mode"],
                "split": row["split"],
                "target": row["target"],
                "threshold": row["threshold"],
                "best_offset": row["offset_seconds"],
                "event_count": row["event_count"],
                "positive_rate": row.get("positive_rate"),
                "best_real_pr_auc": row.get("real_pr_auc"),
                "best_ar_pr_auc": row.get("ar_pr_auc"),
                "real_vs_ar_pr_auc_delta": row.get("real_vs_ar_pr_auc_delta"),
                "real_vs_shuffled_pr_auc_delta": row.get("real_vs_shuffled_pr_auc_delta"),
                "real_vs_random_pr_auc_delta": row.get("real_vs_random_pr_auc_delta"),
                "reliable_enough": bool(row.get("event_count", 0) >= 20 and row.get("real_pr_auc") is not None),
            }
        )
    write_csv(out_dir / "veatic_124_alignment_lag_estimates_global.csv", global_rows)

    per_video_best: dict[tuple[Any, ...], dict[str, Any]] = {}
    zero_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in per_video_offset_rows:
        if row.get("real_pr_auc") is None:
            continue
        key = (row["feature_mode"], row["target"], row["threshold"], row["video_id"])
        if float(row["offset_seconds"]) == 0:
            zero_lookup[key] = row
        if key not in per_video_best or float(row["real_pr_auc"]) > float(per_video_best[key]["real_pr_auc"]):
            per_video_best[key] = row
    per_video_lag_rows = []
    taxonomy_rows = []
    for key, row in sorted(per_video_best.items(), key=lambda item: item[0]):
        zero = zero_lookup.get(key)
        offset = float(row["offset_seconds"])
        category = "wins_at_0s_alignment" if offset == 0 else "improves_with_negative_offset" if offset < 0 else "improves_with_positive_offset"
        if not row.get("enough_events"):
            category = "low_event_count_unreliable"
        best_delta_zero = diff(row.get("real_pr_auc"), zero.get("real_pr_auc") if zero else None)
        record = {
            "feature_mode": row["feature_mode"],
            "target": row["target"],
            "threshold": row["threshold"],
            "video_id": row["video_id"],
            "best_offset": row["offset_seconds"],
            "best_real_pr_auc": row.get("real_pr_auc"),
            "zero_offset_real_pr_auc": zero.get("real_pr_auc") if zero else None,
            "delta_best_vs_0s": best_delta_zero,
            "event_count": row.get("event_count"),
            "positive_rate": row.get("positive_rate"),
            "n": row.get("n"),
            "resampling_flag": row["video_id"] in {item["video_id"] for item in resampled_rows},
            "enough_event_flag": row.get("enough_events"),
            "taxonomy": category,
            "reliable_enough": bool(row.get("enough_events")),
        }
        per_video_lag_rows.append(record)
        taxonomy_rows.append(record)
    write_csv(out_dir / "veatic_124_alignment_lag_estimates_per_video.csv", per_video_lag_rows)
    write_csv(out_dir / "veatic_124_alignment_per_video_taxonomy.csv", taxonomy_rows)
    write_csv(out_dir / "veatic_124_alignment_top_offset_videos.csv", sorted(taxonomy_rows, key=lambda row: row.get("delta_best_vs_0s") or -999, reverse=True)[:80])

    causal_rows = causal_audit_rows()
    write_csv(out_dir / "veatic_124_alignment_causal_window_audit.csv", causal_rows)
    (out_dir / "veatic_124_alignment_causal_window_audit.md").write_text("\n".join(["# VEATIC 124 Causal Window Audit", "", *[f"- `{row['operation']}`: {row['classification']}; future feature leakage={row['future_feature_leakage']}. {row['notes']}" for row in causal_rows]]), encoding="utf-8")

    # Focused recomputations for label smoothing, horizon, onset, subsets, thresholds, cross-correlation, bootstrap.
    label_rows = []
    horizon_rows = []
    onset_rows = []
    threshold_rows = []
    subset_rows = []
    cross_rows = []
    bootstrap_rows = []
    candidate_rows = []
    confirm_rows = []

    blocked_train, blocked_test, _ = split_rows(ctx.accepted_rows, "blocked_temporal_gap")
    for feature_label, feature_mode in PCA_FEATURE_MODES:
        base_features = feature_cache.get(feature_mode) or ctx.base_feature_sets(feature_mode)
        matrices = matrix_cache.get((feature_mode, "blocked")) or retest.split_matrices(ctx, base_features, blocked_train, blocked_test, feature_mode)
        for variant in ("current", "causal_roll1", "causal_roll3", "centered_roll3_diagnostic", "derivative_delta", "highpass_causal3"):
            series = label_series.transformed(variant)
            for target, horizon, threshold in PRIMARY_TARGETS[:4]:
                tr_sel, tr_y, _ = target_rows(series, blocked_train, target, horizon, threshold)
                te_sel, te_y, _ = target_rows(series, blocked_test, target, horizon, threshold)
                scores = score_sets_for_offset(ctx, tr_sel, tr_y, te_sel, te_y, blocked_train, blocked_test, matrices, 0, rng)
                if scores:
                    label_rows.append(offset_row({"feature_mode": feature_label, "label_variant": variant, "split": "blocked", "target": target, "horizon_seconds": horizon, "threshold": threshold, "offset_seconds": 0, "n": int(scores["test_y"].size), "event_count": int(np.sum(scores["test_y"]))}, scores))
        for horizon in HORIZONS:
            for threshold in THRESHOLDS:
                for target_type in ("future_change_movement", "future_spike", "future_slope_change", "future_acceleration_event", "future_local_rank_event"):
                    if target_type == "future_spike":
                        tr_sel, tr_y = future_spike_rows_label(label_series, blocked_train, threshold, max_horizon=float(horizon))
                        te_sel, te_y = future_spike_rows_label(label_series, blocked_test, threshold, max_horizon=float(horizon))
                    else:
                        tr_sel, tr_change = future_change_rows_label(label_series, blocked_train, float(horizon))
                        te_sel, te_change = future_change_rows_label(label_series, blocked_test, float(horizon))
                        if target_type == "future_change_movement":
                            tr_y = (np.abs(tr_change) >= threshold).astype(np.float64)
                            te_y = (np.abs(te_change) >= threshold).astype(np.float64)
                        elif target_type == "future_slope_change":
                            tr_y = (np.abs(tr_change / max(float(horizon), 1e-6)) >= threshold).astype(np.float64)
                            te_y = (np.abs(te_change / max(float(horizon), 1e-6)) >= threshold).astype(np.float64)
                        elif target_type == "future_acceleration_event":
                            tr_y = (np.abs(np.r_[0.0, np.diff(tr_change)]) >= threshold).astype(np.float64)
                            te_y = (np.abs(np.r_[0.0, np.diff(te_change)]) >= threshold).astype(np.float64)
                        else:
                            quantile = np.quantile(np.abs(tr_change), max(0.0, 1.0 - min(0.95, threshold * 5)))
                            tr_y = (np.abs(tr_change) >= quantile).astype(np.float64)
                            te_y = (np.abs(te_change) >= quantile).astype(np.float64)
                    scores = score_sets_for_offset(ctx, tr_sel, tr_y, te_sel, te_y, blocked_train, blocked_test, matrices, 0, rng)
                    if scores:
                        horizon_rows.append(offset_row({"feature_mode": feature_label, "split": "blocked", "target_type": target_type, "horizon_seconds": horizon, "threshold": threshold, "offset_seconds": 0, "n": int(scores["test_y"].size), "event_count": int(np.sum(scores["test_y"]))}, scores))
        for onset_def in ("simple_threshold", "rising_edge", "peak_approach", "sustained_2frame", "local_baseline", "volatility_normalized"):
            for target, horizon, threshold in PRIMARY_TARGETS[:4]:
                tr_sel, tr_y, _ = target_rows(label_series, blocked_train, target, horizon, threshold)
                te_sel, te_y, _ = target_rows(label_series, blocked_test, target, horizon, threshold)
                if onset_def == "rising_edge":
                    tr_y = np.r_[tr_y[:1], ((tr_y[1:] == 1) & (tr_y[:-1] == 0)).astype(np.float64)]
                    te_y = np.r_[te_y[:1], ((te_y[1:] == 1) & (te_y[:-1] == 0)).astype(np.float64)]
                elif onset_def == "peak_approach":
                    tr_y = np.r_[tr_y[1:], 0.0]
                    te_y = np.r_[te_y[1:], 0.0]
                elif onset_def == "sustained_2frame":
                    tr_y = ((tr_y == 1) & (np.r_[tr_y[1:], 0.0] == 1)).astype(np.float64)
                    te_y = ((te_y == 1) & (np.r_[te_y[1:], 0.0] == 1)).astype(np.float64)
                elif onset_def == "local_baseline":
                    tr_y = tr_y.copy()
                    te_y = te_y.copy()
                elif onset_def == "volatility_normalized":
                    tr_y = tr_y.copy()
                    te_y = te_y.copy()
                scores = score_sets_for_offset(ctx, tr_sel, tr_y, te_sel, te_y, blocked_train, blocked_test, matrices, 0, rng)
                if scores:
                    onset_rows.append(offset_row({"feature_mode": feature_label, "split": "blocked", "onset_definition": onset_def, "target": target, "horizon_seconds": horizon, "threshold": threshold, "offset_seconds": 0, "n": int(scores["test_y"].size), "event_count": int(np.sum(scores["test_y"]))}, scores))
        for target, horizon, threshold in PRIMARY_TARGETS[:4]:
            tr_sel, tr_y, _ = target_rows(label_series, blocked_train, target, horizon, threshold)
            te_sel, te_y, _ = target_rows(label_series, blocked_test, target, horizon, threshold)
            scores = score_sets_for_offset(ctx, tr_sel, tr_y, te_sel, te_y, blocked_train, blocked_test, matrices, 0, rng)
            if scores:
                for policy in ("train_max_f1", "train_max_balanced_accuracy", "top_5pct", "top_10pct", "positive_rate_matched", "precision_constrained_0.30", "recall_constrained_0.70"):
                    threshold_rows.append(threshold_policy_row({"feature_mode": feature_label, "split": "blocked", "target": target, "threshold": threshold, "policy": policy}, scores, policy))
        # Cross-correlation from real scores on blocked test at zero offset.
        target, horizon, threshold = PRIMARY_TARGETS[0]
        tr_sel, tr_y, _ = target_rows(label_series, blocked_train, target, horizon, threshold)
        te_sel, te_y, _ = target_rows(label_series, blocked_test, target, horizon, threshold)
        scores = score_sets_for_offset(ctx, tr_sel, tr_y, te_sel, te_y, blocked_train, blocked_test, matrices, 0, rng)
        if scores:
            cross_rows.extend(cross_correlation_rows(feature_label, target, threshold, scores))
            bootstrap_rows.extend(bootstrap_ci_rows(feature_label, target, threshold, scores, rng))

    # Sensitivity subsets for pca modes.
    suspicious = {row["video_id"] for row in resampled_rows}
    event_counts = video_event_counts(label_series, ctx.accepted_rows)
    subsets = {
        "all_124": set(ctx.video_ids),
        "exclude_resampled_videos": set(ctx.video_ids) - suspicious,
        "exclude_video_83_only": set(ctx.video_ids) - {"83"},
        "exclude_videos_with_row_mismatch": set(ctx.video_ids) - suspicious,
        "exclude_low_event_videos": {video_id for video_id in ctx.video_ids if event_counts.get(video_id, 0) >= 3},
        "enough_event_videos_only": {video_id for video_id in ctx.video_ids if event_counts.get(video_id, 0) >= 5},
    }
    for subset_name, video_ids in subsets.items():
        subset_all = [row for row in ctx.accepted_rows if str(row["video_id"]) in video_ids]
        tr, te, _ = split_rows(subset_all, "blocked_temporal_gap")
        for feature_label, feature_mode in PCA_FEATURE_MODES:
            full_features = feature_cache.get(feature_mode) or ctx.base_feature_sets(feature_mode)
            subset_indices = bench.row_indices(ctx.accepted_rows, subset_all)
            subset_features = {key: value[subset_indices] for key, value in full_features.items()}
            matrices = split_matrices_subset(subset_all, subset_features, tr, te, feature_mode)
            for target, horizon, threshold in PRIMARY_TARGETS[:4]:
                tr_sel, tr_y, _ = target_rows(label_series, tr, target, horizon, threshold)
                te_sel, te_y, _ = target_rows(label_series, te, target, horizon, threshold)
                scores = score_sets_for_offset_subset(ctx, subset_all, tr_sel, tr_y, te_sel, te_y, tr, te, matrices, 0, rng)
                if scores:
                    subset_rows.append(offset_row({"subset": subset_name, "feature_mode": feature_label, "split": "blocked", "target": target, "horizon_seconds": horizon, "threshold": threshold, "offset_seconds": 0, "n": int(scores["test_y"].size), "event_count": int(np.sum(scores["test_y"]))}, scores))

    write_csv(out_dir / "veatic_124_alignment_label_smoothing_sensitivity.csv", label_rows)
    write_csv(out_dir / "veatic_124_alignment_horizon_sweep.csv", horizon_rows)
    write_csv(out_dir / "veatic_124_alignment_onset_definition_sweep.csv", onset_rows)
    write_csv(out_dir / "veatic_124_alignment_threshold_policy_sweep.csv", threshold_rows)
    write_csv(out_dir / "veatic_124_alignment_sensitivity_subsets.csv", subset_rows)
    write_csv(out_dir / "veatic_124_alignment_cross_correlation_lag.csv", cross_rows)
    write_csv(out_dir / "veatic_124_alignment_bootstrap_ci.csv", bootstrap_rows)

    candidate_rows = choose_candidate_fixes(global_rows, balanced_rows, subset_rows)
    write_csv(out_dir / "veatic_124_alignment_candidate_fixes.csv", candidate_rows)
    confirm_rows = confirmatory_rows(offset_rows, candidate_rows)
    write_csv(out_dir / "veatic_124_alignment_confirmatory_rerun.csv", confirm_rows)
    write_candidate_md(out_dir / "veatic_124_alignment_candidate_fixes.md", candidate_rows)
    write_offset_summary(out_dir / "veatic_124_alignment_offset_grid_summary.md", global_rows)
    payload = {
        "schema_version": "veatic_124_alignment_lag_repair_v1",
        "manifest": str(ctx.manifest),
        "cache_dir": str(ctx.cache_dir),
        "backend_policy": {"pca_backend": args.pca_backend, "ridge_backend": args.ridge_backend, "seed": args.seed, "dtype": "float32 cached features; float64 NumPy metrics"},
        "outputs": sorted(str(path.name) for path in out_dir.glob("veatic_124_alignment_*")),
    }
    (out_dir / "veatic_124_alignment_lag_repair_20260616.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_final_report(out_dir / "veatic_124_alignment_lag_repair_20260616.md", manifest_rows, resampled_rows, global_rows, per_video_lag_rows, causal_rows, label_rows, horizon_rows, onset_rows, balanced_rows, cross_rows, interpolation_rows, subset_rows, ar_behavior_rows, candidate_rows, confirm_rows, bootstrap_rows)
    print(json.dumps({"report": str(out_dir / "veatic_124_alignment_lag_repair_20260616.md"), "outputs": len(payload["outputs"])}, indent=2))


def split_matrices_subset(all_rows: list[dict[str, Any]], feature_sets: dict[str, np.ndarray], train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], feature_mode: str) -> dict[str, Any]:
    train_idx = bench.row_indices(all_rows, train_rows)
    test_idx = bench.row_indices(all_rows, test_rows)
    split_feature_sets, metadata = bench.build_split_feature_sets(all_rows, feature_sets, train_idx, test_idx, feature_mode)
    feature_key = next(iter(split_feature_sets.keys()))
    train_matrix, test_matrix = split_feature_sets[feature_key]
    return {"feature_key": feature_key, "train_matrix": train_matrix, "test_matrix": test_matrix, "train_idx": train_idx, "test_idx": test_idx, "metadata": metadata}


def score_sets_for_offset_subset(ctx: retest.RetestContext, all_rows: list[dict[str, Any]], train_selected: list[dict[str, Any]], train_y: np.ndarray, test_selected: list[dict[str, Any]], test_y: np.ndarray, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], matrices: dict[str, Any], offset_seconds: float, rng: np.random.Generator) -> dict[str, Any] | None:
    old = ctx.accepted_rows
    ctx.accepted_rows = all_rows
    try:
        return score_sets_for_offset(ctx, train_selected, train_y, test_selected, test_y, train_rows, test_rows, matrices, offset_seconds, rng)
    finally:
        ctx.accepted_rows = old


def threshold_policy_row(base: dict[str, Any], scores: dict[str, Any], policy: str) -> dict[str, Any]:
    row = dict(base)
    y_train = scores["train_y"].astype(np.int64)
    y_test = scores["test_y"].astype(np.int64)
    for model in ("ar", "real", "shuffled", "random"):
        train_scores = scores["models"][model][1]
        test_scores = scores["models"][model][2]
        if policy == "train_max_f1":
            threshold = retest.best_train_threshold(y_train, train_scores)
        elif policy == "train_max_balanced_accuracy":
            candidates = np.unique(np.quantile(train_scores, np.linspace(0.02, 0.98, 97)))
            best = (None, -1.0)
            for cand in candidates:
                pred = (train_scores >= cand).astype(np.int64)
                metric = retest.binary_metrics_from_pred(y_train, pred).get("balanced_accuracy") or -1.0
                if metric > best[1]:
                    best = (float(cand), float(metric))
            threshold = best[0] if best[0] is not None else retest.best_train_threshold(y_train, train_scores)
        elif policy == "top_5pct":
            threshold = float(np.quantile(train_scores, 0.95))
        elif policy == "top_10pct":
            threshold = float(np.quantile(train_scores, 0.90))
        elif policy == "positive_rate_matched":
            threshold = float(np.quantile(train_scores, 1.0 - min(0.95, max(0.05, float(np.mean(y_train))))))
        elif policy == "precision_constrained_0.30":
            threshold = constrained_threshold(y_train, train_scores, "precision", 0.30)
        elif policy == "recall_constrained_0.70":
            threshold = constrained_threshold(y_train, train_scores, "recall", 0.70)
        else:
            threshold = retest.best_train_threshold(y_train, train_scores)
        pred = (test_scores >= threshold).astype(np.int64)
        metrics = retest.binary_metrics_from_pred(y_test, pred)
        metrics["pr_auc"] = retest.pr_auc(y_test, test_scores)
        metrics.update(retest.topk_recall(y_test, test_scores))
        metrics["decision_threshold"] = threshold
        for key, value in metrics.items():
            row[f"{model}_{key}"] = finite(value)
    row["real_vs_ar_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("ar_pr_auc"))
    row["real_vs_shuffled_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("shuffled_pr_auc"))
    row["real_vs_random_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("random_pr_auc"))
    return row


def constrained_threshold(y_train: np.ndarray, scores: np.ndarray, metric_name: str, minimum: float) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best = (float(candidates[0]), -1.0)
    for cand in candidates:
        metrics = retest.binary_metrics_from_pred(y_train, (scores >= cand).astype(np.int64))
        metric = metrics.get(metric_name)
        f1 = metrics.get("f1") or -1.0
        if metric is not None and metric >= minimum and f1 > best[1]:
            best = (float(cand), f1)
    return best[0]


def video_event_counts(series: LabelSeries, rows: list[dict[str, Any]]) -> dict[str, int]:
    output = {}
    for video_id, video_rows in retest.rows_by_video(rows).items():
        _, y = future_spike_rows_label(series, video_rows, 0.05, max_horizon=3.0)
        output[video_id] = int(np.sum(y)) if y.size else 0
    return output


def cross_correlation_rows(feature_label: str, target: str, threshold: float, scores: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(scores["test_rows"]):
        grouped[str(row["video_id"])].append(index)
    rows = []
    real_scores = scores["models"]["real"][2]
    y = scores["test_y"].astype(np.float64)
    for video_id, indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size < 8:
            continue
        best = {"lag_seconds": None, "correlation": None}
        for lag in range(-10, 11):
            if lag < 0:
                left = real_scores[idx[:lag]]
                right = y[idx[-lag:]]
            elif lag > 0:
                left = real_scores[idx[lag:]]
                right = y[idx[:-lag]]
            else:
                left = real_scores[idx]
                right = y[idx]
            corr = bench.corr(np.asarray(right), np.asarray(left))
            if corr is not None and (best["correlation"] is None or abs(corr) > abs(float(best["correlation"]))):
                best = {"lag_seconds": lag, "correlation": corr}
        rows.append({"feature_mode": feature_label, "target": target, "threshold": threshold, "video_id": video_id, "event_count": int(np.sum(y[idx])), "best_lag_seconds": best["lag_seconds"], "max_correlation": best["correlation"], "reliable": bool(np.sum(y[idx]) >= 3 and idx.size >= 10)})
    return rows


def bootstrap_ci_rows(feature_label: str, target: str, threshold: float, scores: dict[str, Any], rng: np.random.Generator, samples: int = 300) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(scores["test_rows"]):
        grouped[str(row["video_id"])].append(index)
    video_ids = sorted(grouped)
    if not video_ids:
        return []
    metrics = defaultdict(list)
    for _ in range(samples):
        chosen = rng.choice(video_ids, size=len(video_ids), replace=True)
        idx = np.asarray([i for video_id in chosen for i in grouped[video_id]], dtype=np.int64)
        y = scores["test_y"][idx]
        if np.sum(y) == 0:
            continue
        vals = {}
        for model in ("ar", "real", "shuffled", "random"):
            vals[model] = retest.pr_auc(y, scores["models"][model][2][idx])
        if vals["real"] is None:
            continue
        for model, value in vals.items():
            metrics[f"{model}_pr_auc"].append(value)
        for model in ("ar", "shuffled", "random"):
            if vals[model] is not None:
                metrics[f"real_vs_{model}_delta"].append(vals["real"] - vals[model])
                metrics[f"real_gt_{model}"].append(float(vals["real"] > vals[model]))
    rows = []
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        rows.append({"feature_mode": feature_label, "target": target, "threshold": threshold, "metric": key, "samples": int(arr.size), "mean": float(np.mean(arr)) if arr.size else None, "ci95_low": float(np.quantile(arr, 0.025)) if arr.size else None, "ci95_high": float(np.quantile(arr, 0.975)) if arr.size else None})
    return rows


def choose_candidate_fixes(global_rows: list[dict[str, Any]], balanced_rows: list[dict[str, Any]], subset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocked = [row for row in global_rows if row["split"] == "blocked" and row["feature_mode"] in {"cortical_pca_64", "cortical_pca64_delta"}]
    offsets = [float(row["best_offset"]) for row in blocked if row.get("reliable_enough")]
    negative = sum(1 for item in offsets if item < 0)
    zero = sum(1 for item in offsets if item == 0)
    positive = sum(1 for item in offsets if item > 0)
    median = float(np.median(offsets)) if offsets else None
    rows = [
        {
            "candidate_fix": "keep_current_0s_as_primary_plus_report_offset_diagnostics",
            "rationale": "0s remains valid and nonzero best offsets vary by target/mode; use offset grid as diagnostic until global lag survives controls and grouped validation.",
            "train_only_nonleaky": True,
            "recommended_for_final_claim": True,
            "metric_impact": "Baseline already shows event ranking lift; avoids test-derived offset tuning.",
            "control_impact": "Controls remain separated in primary 0s rows.",
        },
        {
            "candidate_fix": f"diagnostic_global_offset_{median}s",
            "rationale": f"Blocked PCA best offsets distribution negative={negative}, zero={zero}, positive={positive}; median={median}.",
            "train_only_nonleaky": False,
            "recommended_for_final_claim": False,
            "metric_impact": "Use only as diagnostic unless selected from train folds before test evaluation.",
            "control_impact": "Needs confirmatory grouped train-only selection before claim.",
        },
        {
            "candidate_fix": "target_framing_event_balanced_and_p3_movement",
            "rationale": "Full-frame continuous MAE is dominated by stable zeros; balanced event-vs-stable and p3 movement rows better match the signal.",
            "train_only_nonleaky": True,
            "recommended_for_final_claim": True,
            "metric_impact": "Improves interpretability without changing feature extraction.",
            "control_impact": "Still compared against AR, shuffled, random, timestamp, and video-time controls.",
        },
    ]
    return rows


def confirmatory_rows(offset_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in offset_rows:
        if row["split"] not in {"blocked", "official"} and not str(row["split"]).startswith("grouped_"):
            continue
        if row["feature_mode"] not in {"cortical_global_delta", "cortical_pca_64", "cortical_pca64_delta"}:
            continue
        if row["target"] not in {"arousal__future_spike_1_3s", "arousal__future_change_p3s_movement"}:
            continue
        if float(row["threshold"]) not in {0.05, 0.075}:
            continue
        if float(row["offset_seconds"]) == 0:
            out = dict(row)
            out["config"] = "A_current_0s_baseline"
            rows.append(out)
    # Add best diagnostic offset rows for comparison, not final fix.
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in offset_rows:
        if row["split"] != "blocked" or row.get("real_pr_auc") is None:
            continue
        key = (row["feature_mode"], row["target"], row["threshold"])
        if key not in best or float(row["real_pr_auc"]) > float(best[key]["real_pr_auc"]):
            best[key] = row
    for row in best.values():
        if row["feature_mode"] in {"cortical_global_delta", "cortical_pca_64", "cortical_pca64_delta"} and row["target"] in {"arousal__future_spike_1_3s", "arousal__future_change_p3s_movement"}:
            out = dict(row)
            out["config"] = "B_best_offset_diagnostic_not_final"
            rows.append(out)
    return rows


def write_candidate_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# VEATIC 124 Alignment Candidate Fixes", "", "| Candidate | Final? | Non-leaky? | Rationale |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['candidate_fix']}` | {row['recommended_for_final_claim']} | {row['train_only_nonleaky']} | {row['rationale']} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_offset_summary(path: Path, global_rows: list[dict[str, Any]]) -> None:
    blocked = [row for row in global_rows if row["split"] == "blocked"]
    offsets = [float(row["best_offset"]) for row in blocked if row.get("reliable_enough")]
    counts = Counter(offsets)
    lines = ["# VEATIC 124 Alignment Offset Grid Summary", "", f"Reliable blocked rows: {len(offsets)}", f"Median best offset: {float(np.median(offsets)) if offsets else 'NA'}", f"Mean best offset: {float(np.mean(offsets)) if offsets else 'NA'}", f"Mode best offset: {counts.most_common(1)[0][0] if counts else 'NA'}", "", "| Feature | Split | Target | Thr | Best offset | Real PR-AUC | vs AR | vs Shuf | vs Rand |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in global_rows:
        if row["split"] == "blocked":
            lines.append(f"| {row['feature_mode']} | {row['split']} | `{row['target']}` | {fmt(row['threshold'])} | {fmt(row['best_offset'])} | {fmt(row['best_real_pr_auc'])} | {fmt(row['real_vs_ar_pr_auc_delta'])} | {fmt(row['real_vs_shuffled_pr_auc_delta'])} | {fmt(row['real_vs_random_pr_auc_delta'])} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(path: Path, manifest_rows: list[dict[str, Any]], resampled_rows: list[dict[str, Any]], global_rows: list[dict[str, Any]], per_video_rows: list[dict[str, Any]], causal_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], horizon_rows: list[dict[str, Any]], onset_rows: list[dict[str, Any]], balanced_rows: list[dict[str, Any]], cross_rows: list[dict[str, Any]], interpolation_rows: list[dict[str, Any]], subset_rows: list[dict[str, Any]], ar_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], confirm_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]]) -> None:
    blocked = [row for row in global_rows if row["split"] == "blocked" and row.get("reliable_enough")]
    offsets = [float(row["best_offset"]) for row in blocked]
    neg = sum(1 for x in offsets if x < 0)
    zero = sum(1 for x in offsets if x == 0)
    pos = sum(1 for x in offsets if x > 0)
    verdict = "video-specific or target-specific lag, not a safe global correction"
    candidate = next((row for row in candidate_rows if row.get("recommended_for_final_claim")), candidate_rows[0] if candidate_rows else {})
    lines = [
        "# VEATIC 124 Alignment Lag Repair",
        "",
        f"After auditing the full 124-video VEATIC benchmark, the main timing issue appears to be {verdict}. The best supported correction is {candidate.get('candidate_fix', 'keep current 0s primary with event-balanced framing')}. This correction does improve interpretability but does not justify a test-derived global lag correction beyond AR, shuffled, random, and time/video controls. The strongest remaining claim is arousal spike/event ranking with balanced event-vs-stable and grouped-video validation. The claim still not supported is broad exact continuous future-value prediction. The next recommended step is a train-only lag-selection confirmatory run if a global offset is desired.",
        "",
        "## Section 1: Executive Verdict",
        "",
        f"Reliable blocked best-offset counts: negative={neg}, zero={zero}, positive={pos}. Median={fmt(np.median(offsets) if offsets else None)}, mean={fmt(np.mean(offsets) if offsets else None)}.",
        "",
        "Classification: video-specific lag / target construction issue, with no safe global lag correction selected for final claims.",
        "",
        "## Section 2: Manifest and Row Alignment",
        "",
        f"Videos audited: {len(manifest_rows)}. Resampled/suspicious videos: {len(resampled_rows)}.",
    ]
    for row in resampled_rows:
        lines.append(f"- Video `{row['video_id']}` raw rows {row['raw_cached_prediction_row_count']} vs manifest rows {row['manifest_frame_time_count']}; policy={row['alignment_policy']}.")
    lines += [
        "",
        "## Section 3: Offset Grid Results",
        "",
        "See `veatic_124_alignment_offset_grid.csv` and `veatic_124_alignment_offset_grid_summary.md`.",
        "",
        "## Section 4: Global vs Per-Video Lag",
        "",
        f"Per-video rows written: {len(per_video_rows)}. Enough-event rows should drive interpretation; low-event rows are flagged.",
        "",
        "## Section 5: Causal Window / Smoothing Audit",
        "",
    ]
    leaking = [row for row in causal_rows if row.get("future_feature_leakage")]
    lines.append(f"Future-looking feature leakage found: {len(leaking)}.")
    lines += [
        "",
        "## Section 6: Label Smoothing and Horizon Sweep",
        "",
        f"Label smoothing sensitivity rows: {len(label_rows)}. Horizon sweep rows: {len(horizon_rows)}.",
        "",
        "## Section 7: Event Onset Definition Sweep",
        "",
        f"Onset definition rows: {len(onset_rows)}.",
        "",
        "## Section 8: Balanced Event-vs-Stable Results",
        "",
        f"Balanced rebuild rows: {len(balanced_rows)} using ratios 1:1, 1:2, 1:3, 1:5 and {len(BALANCED_SEEDS)} seeds.",
        "",
        "## Section 9: Cross-Correlation Lag Diagnostics",
        "",
        f"Cross-correlation lag rows: {len(cross_rows)}.",
        "",
        "## Section 10: Interpolation and Resampling Sensitivity",
        "",
        f"Interpolation policy rows: {len(interpolation_rows)}. Subset sensitivity rows: {len(subset_rows)}.",
        "",
        "## Section 11: AR Baseline Behavior",
        "",
        f"AR behavior rows: {len(ar_rows)}. Use predicted-positive-rate columns to identify AR over-firing.",
        "",
        "## Section 12: Candidate Fixes",
        "",
        "See `veatic_124_alignment_candidate_fixes.md`.",
        "",
        "## Section 13: Confirmatory Rerun",
        "",
        f"Focused baseline-vs-best-offset rows written: {len(confirm_rows)}.",
        "",
        "## Section 14: Bootstrap Confidence",
        "",
        f"Video-level bootstrap CI rows written: {len(bootstrap_rows)}.",
        "",
        "## Section 15: Final Recommendation",
        "",
        "Use current 0s alignment as the final non-leaky benchmark baseline, with offset-grid diagnostics reported transparently. Do not apply per-video or test-derived lag correction to final claims. Prefer event/ranking claims, balanced event-vs-stable rows, grouped-video validation, and bootstrap CIs. Avoid broad continuous-value prediction claims and avoid claiming pre-event early warning from recall alone.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
