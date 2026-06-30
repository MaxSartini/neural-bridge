"""Post-hoc VEATIC event/spike retest with train-only calibration.

This script deliberately does not change TRIBE extraction or model
architectures. It recomputes fixed-split ridge predictions from cached
features and evaluates event/spike diagnostics with train-calibrated decision
thresholds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(ROOT / "external_assets"))).expanduser()
BENCHMARK_SCRIPT = ROOT / "backend" / "scripts" / "run_veatic_neuro_benchmark.py"
spec = importlib.util.spec_from_file_location("veatic_benchmark", BENCHMARK_SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench)

FEATURE_MODES = (
    ("cortical_fast_default", "cortical_global"),
    ("cortical_global_delta", "cortical_global_delta"),
    ("cortical_pca_64", "cortical_pca_64"),
    ("cortical_pca64_delta", "cortical_pca64_delta"),
)
SPLITS = (
    ("blocked", "blocked_temporal_gap"),
    ("official", "official_70_30"),
)
HORIZONS = (1, 2, 3)
EVENT_THRESHOLDS = (0.03, 0.05, 0.075, 0.10)
SHIFT_OFFSETS = (-5, -3, -2, -1, 0, 1, 2, 3, 5)
TOP_FRACS = (0.01, 0.05, 0.10)
LOCAL_WINDOWS = (1, 3, 5)
ONSET_LEADS = (1, 2, 3, 5)


def rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda item: float(item["time_start_seconds"]))
    return dict(grouped)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, spearman: bool = False) -> float | None:
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return None
    left = bench.rankdata(y_true) if spearman else y_true
    right = bench.rankdata(y_pred) if spearman else y_pred
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))) if y_true.size else None,
        "rmse": float(np.sqrt(np.mean(np.square(y_true - y_pred)))) if y_true.size else None,
        "pearson": safe_corr(y_true, y_pred),
        "spearman": safe_corr(y_true, y_pred, spearman=True),
    }


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y_true == 1))
    if positives == 0 or y_true.size == 0:
        return None
    order = np.argsort(-scores)
    sorted_y = y_true[order]
    tp = np.cumsum(sorted_y == 1)
    fp = np.cumsum(sorted_y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def binary_metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    counts = binary_counts(y_true, y_pred)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    accuracy = (tp + tn) / max(1.0, tp + fp + fn + tn)
    return {
        "f1": f1,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        **counts,
    }


def best_train_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    if np.sum(y_true == 1) == 0:
        return float(np.max(scores) + 1e-6)
    if np.sum(y_true == 0) == 0:
        return float(np.min(scores) - 1e-6)
    quantiles = np.linspace(0.02, 0.98, 97)
    candidates = np.unique(np.quantile(scores, quantiles))
    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        metrics = binary_metrics_from_pred(y_true, pred)
        key = (
            metrics["f1"] if metrics["f1"] is not None else -1.0,
            metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0,
            metrics["recall"] if metrics["recall"] is not None else -1.0,
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def event_metrics(
    train_y: np.ndarray,
    train_scores: np.ndarray,
    test_y: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float | None]:
    threshold = best_train_threshold(train_y, train_scores)
    pred = (test_scores >= threshold).astype(np.int64)
    metrics = binary_metrics_from_pred(test_y, pred)
    metrics.update(
        {
            "decision_threshold": threshold,
            "pr_auc": pr_auc(test_y, test_scores),
            "positive_class_rate": float(np.mean(test_y)) if test_y.size else None,
            "event_count": float(np.sum(test_y)),
        }
    )
    metrics.update(topk_recall(test_y, test_scores))
    return metrics


def majority_metrics(train_y: np.ndarray, test_y: np.ndarray) -> dict[str, float | None]:
    majority = int(np.mean(train_y) >= 0.5) if train_y.size else 0
    pred = np.full(test_y.shape, majority, dtype=np.int64)
    metrics = binary_metrics_from_pred(test_y, pred)
    metrics.update(
        {
            "decision_threshold": None,
            "pr_auc": None,
            "positive_class_rate": float(np.mean(test_y)) if test_y.size else None,
            "event_count": float(np.sum(test_y)),
        }
    )
    return metrics


def topk_recall(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(y_true == 1))
    output: dict[str, float | None] = {}
    if y_true.size == 0 or positives == 0:
        for frac in TOP_FRACS:
            output[f"top_{int(frac * 100)}pct_recall"] = None
        output["top_event_count_recall"] = None
        return output
    order = np.argsort(-scores)
    for frac in TOP_FRACS:
        k = max(1, int(math.ceil(y_true.size * frac)))
        output[f"top_{int(frac * 100)}pct_recall"] = float(np.sum(y_true[order[:k]] == 1) / positives)
    output["top_event_count_recall"] = float(np.sum(y_true[order[:positives]] == 1) / positives)
    return output


def sign_class(values: np.ndarray, threshold: float) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.int64)
    output[values >= threshold] = 1
    output[values <= -threshold] = -1
    return output


def temporal_lookup(rows: list[dict[str, Any]], target: str = "arousal") -> dict[tuple[str, int], float]:
    return {
        (str(row["video_id"]), int(round(float(row["time_start_seconds"])))): float(row["targets"][target])
        for row in rows
    }


def fixed_rows(rows: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    train = []
    test = []
    gap = 0
    for row in rows:
        split = row["splits"][split_name]
        if split == "train":
            train.append(row)
        elif split == "test":
            test.append(row)
        elif split == "gap":
            gap += 1
    return train, test, gap


class RetestContext:
    def __init__(self, manifest: Path, report: Path, cache_dir: Path) -> None:
        self.manifest = manifest
        self.report = report
        self.cache_dir = cache_dir
        self.rows = bench.load_manifest(manifest)
        report_data = json.loads(report.read_text(encoding="utf-8"))
        self.video_ids = [str(item) for item in report_data["complete_video_ids"]]
        grouped_manifest = rows_by_video(self.rows)
        self.accepted_rows: list[dict[str, Any]] = []
        self.features_by_video: dict[str, dict[str, np.ndarray]] = {}
        for video_id in self.video_ids:
            video_rows = grouped_manifest[video_id]
            feature_sets, _ = bench.load_cached_video_features(
                cache_dir,
                video_id,
                len(video_rows),
            )
            self.features_by_video[video_id] = feature_sets
            self.accepted_rows.extend(video_rows)
        self.lookup = temporal_lookup(self.accepted_rows, "arousal")
        self.index_by_key = {
            (str(row["video_id"]), int(round(float(row["time_start_seconds"])))): index
            for index, row in enumerate(self.accepted_rows)
        }
        self.grouped_rows = rows_by_video(self.accepted_rows)

    def base_feature_sets(self, feature_mode: str) -> dict[str, np.ndarray]:
        blocks: dict[str, list[np.ndarray]] = defaultdict(list)
        selected = bench.cache_feature_keys_for(feature_mode, "cortical_fast_default")
        for video_id in self.video_ids:
            for key in selected:
                blocks[key].append(self.features_by_video[video_id][key])
        feature_sets = {key: np.concatenate(values, axis=0) for key, values in blocks.items()}
        if feature_mode == "cortical_global_delta":
            feature_sets["cortical_global_delta"] = bench.temporal_dynamics_features(
                self.accepted_rows,
                feature_sets["cortical_global"],
                include_base=True,
            )
        return feature_sets


def future_change_rows(
    ctx: RetestContext,
    rows: list[dict[str, Any]],
    horizon: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    values = []
    for row in rows:
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        current = ctx.lookup.get((video_id, second))
        future = ctx.lookup.get((video_id, second + horizon))
        if current is None or future is None:
            continue
        selected.append(row)
        values.append(future - current)
    return selected, np.asarray(values, dtype=np.float64)


def future_spike_rows(
    ctx: RetestContext,
    rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    values = []
    for row in rows:
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        current = ctx.lookup.get((video_id, second))
        if current is None:
            continue
        futures = [ctx.lookup.get((video_id, second + horizon)) for horizon in (1, 2, 3)]
        futures = [value for value in futures if value is not None]
        if not futures:
            continue
        selected.append(row)
        values.append(float(max(futures) - current >= threshold))
    return selected, np.asarray(values, dtype=np.float64)


def onset_rows(
    ctx: RetestContext,
    rows: list[dict[str, Any]],
    threshold: float,
    leadup_seconds: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    values = []
    for row in rows:
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        current = ctx.lookup.get((video_id, second))
        if current is None:
            continue
        futures = [ctx.lookup.get((video_id, second + horizon)) for horizon in (1, 2, 3)]
        futures = [value for value in futures if value is not None]
        if not futures:
            continue
        stable = True
        previous = current
        for offset in range(1, leadup_seconds + 1):
            value = ctx.lookup.get((video_id, second - offset))
            if value is None:
                stable = False
                break
            if abs(previous - value) > threshold / 2.0:
                stable = False
                break
            previous = value
        selected.append(row)
        values.append(float(stable and max(futures) - current >= threshold))
    return selected, np.asarray(values, dtype=np.float64)


def rolling_values(ctx: RetestContext, video_id: str, second: int, window: int) -> list[float]:
    values = []
    for offset in range(window):
        value = ctx.lookup.get((video_id, second - offset))
        if value is not None:
            values.append(value)
    return values


def video_percentiles(ctx: RetestContext) -> dict[tuple[str, int], float]:
    output = {}
    for video_id, rows in ctx.grouped_rows.items():
        values = np.asarray([row["targets"]["arousal"] for row in rows], dtype=np.float64)
        ranks = bench.rankdata(values) / max(1, len(values) - 1)
        for row, rank in zip(rows, ranks):
            second = int(round(float(row["time_start_seconds"])))
            output[(video_id, second)] = float(rank)
    return output


def local_target_rows(
    ctx: RetestContext,
    rows: list[dict[str, Any]],
    variant: str,
    horizon: int,
    window: int,
    threshold: float,
    percentiles: dict[tuple[str, int], float],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected = []
    values = []
    for row in rows:
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        current = ctx.lookup.get((video_id, second))
        future = ctx.lookup.get((video_id, second + horizon))
        if current is None or future is None:
            continue
        history = rolling_values(ctx, video_id, second, window)
        previous = ctx.lookup.get((video_id, second - window))
        if variant == "future_minus_rolling_baseline":
            if not history:
                continue
            score = future - float(np.mean(history))
        elif variant == "future_slope_minus_prior_slope":
            if previous is None:
                continue
            score = (future - current) / horizon - (current - previous) / window
        elif variant == "future_acceleration":
            if previous is None:
                continue
            score = (future - current) - (current - previous)
        elif variant == "future_rank_percentile_delta":
            current_rank = percentiles.get((video_id, second))
            future_rank = percentiles.get((video_id, second + horizon))
            if current_rank is None or future_rank is None:
                continue
            score = future_rank - current_rank
        elif variant == "future_change_local_volatility":
            if len(history) < 2:
                continue
            volatility = float(np.std(np.diff(np.asarray(list(reversed(history)), dtype=np.float64))))
            if volatility < 1e-6:
                continue
            score = (future - current) / volatility
        else:
            raise ValueError(variant)
        selected.append(row)
        values.append(float(abs(score) >= threshold))
    return selected, np.asarray(values, dtype=np.float64)


def map_shifted_features(
    ctx: RetestContext,
    selected_rows: list[dict[str, Any]],
    split_position: dict[int, int],
    split_matrix: np.ndarray,
    offset_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = []
    features = []
    for index, row in enumerate(selected_rows):
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"]))) + offset_seconds
        global_index = ctx.index_by_key.get((video_id, second))
        if global_index is None or global_index not in split_position:
            continue
        keep.append(index)
        features.append(split_matrix[split_position[global_index]])
    if not keep:
        return np.asarray([], dtype=np.int64), np.zeros((0, split_matrix.shape[1]), dtype=np.float64)
    return np.asarray(keep, dtype=np.int64), np.asarray(features, dtype=np.float64)


def fit_scores(
    ctx: RetestContext,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    train_position: dict[int, int],
    test_position: dict[int, int],
    *,
    offset_seconds: int = 0,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    train_keep, train_features = map_shifted_features(
        ctx, train_selected, train_position, train_matrix, offset_seconds
    )
    test_keep, test_features = map_shifted_features(
        ctx, test_selected, test_position, test_matrix, offset_seconds
    )
    if train_keep.size < 8 or test_keep.size < 4:
        return None
    train_rows = [train_selected[int(index)] for index in train_keep]
    test_rows = [test_selected[int(index)] for index in test_keep]
    y_train = train_y[train_keep]
    train_ar = bench.autoregressive_features(
        ctx.accepted_rows, train_rows, "arousal", include_current=True
    )
    test_ar = bench.autoregressive_features(
        ctx.accepted_rows, test_rows, "arousal", include_current=True
    )

    ar_train_scores, _ = bench.ridge_fit_predict(train_ar, y_train, train_ar)
    ar_test_scores, _ = bench.ridge_fit_predict(train_ar, y_train, test_ar)

    real_train_x = np.concatenate([train_ar, train_features], axis=1)
    real_test_x = np.concatenate([test_ar, test_features], axis=1)
    real_train_scores, _ = bench.ridge_fit_predict(real_train_x, y_train, real_train_x)
    real_test_scores, _ = bench.ridge_fit_predict(real_train_x, y_train, real_test_x)

    shuffled_train = train_features.copy()
    shuffled_test = test_features.copy()
    rng.shuffle(shuffled_train, axis=0)
    rng.shuffle(shuffled_test, axis=0)
    shuffled_train_x = np.concatenate([train_ar, shuffled_train], axis=1)
    shuffled_test_x = np.concatenate([test_ar, shuffled_test], axis=1)
    shuffled_train_scores, _ = bench.ridge_fit_predict(
        shuffled_train_x, y_train, shuffled_train_x
    )
    shuffled_test_scores, _ = bench.ridge_fit_predict(
        shuffled_train_x, y_train, shuffled_test_x
    )

    random_train = rng.normal(size=train_features.shape)
    random_test = rng.normal(size=test_features.shape)
    random_train_x = np.concatenate([train_ar, random_train], axis=1)
    random_test_x = np.concatenate([test_ar, random_test], axis=1)
    random_train_scores, _ = bench.ridge_fit_predict(random_train_x, y_train, random_train_x)
    random_test_scores, _ = bench.ridge_fit_predict(random_train_x, y_train, random_test_x)

    return {
        "train_keep": train_keep,
        "test_keep": test_keep,
        "test_rows": test_rows,
        "train_y": y_train,
        "ar_train": ar_train_scores,
        "ar_test": ar_test_scores,
        "real_train": real_train_scores,
        "real_test": real_test_scores,
        "shuffled_train": shuffled_train_scores,
        "shuffled_test": shuffled_test_scores,
        "random_train": random_train_scores,
        "random_test": random_test_scores,
        "time_train": bench.time_features(train_rows),
        "time_test": bench.time_features(test_rows),
    }


def add_event_metrics(row: dict[str, Any], prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = finite_float(value)


def evaluate_event(
    ctx: RetestContext,
    feature_label: str,
    split_label: str,
    target_name: str,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    matrices: dict[str, Any],
    threshold: float | None,
    rng: np.random.Generator,
    *,
    offset_seconds: int = 0,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    scores = fit_scores(
        ctx,
        train_selected,
        train_y,
        test_selected,
        matrices["train_matrix"],
        matrices["test_matrix"],
        matrices["train_position"],
        matrices["test_position"],
        offset_seconds=offset_seconds,
        rng=rng,
    )
    if scores is None:
        return None, None
    y_test = test_y[scores["test_keep"]].astype(np.int64)
    row: dict[str, Any] = {
        "feature_mode": feature_label,
        "split": split_label,
        "target": target_name,
        "threshold": threshold,
        "offset_seconds": offset_seconds,
        "n": int(y_test.size),
    }
    for prefix, train_key, test_key in (
        ("ar", "ar_train", "ar_test"),
        ("real", "real_train", "real_test"),
        ("shuffled", "shuffled_train", "shuffled_test"),
        ("random", "random_train", "random_test"),
    ):
        add_event_metrics(
            row,
            prefix,
            event_metrics(scores["train_y"], scores[train_key], y_test, scores[test_key]),
        )
    add_event_metrics(row, "majority", majority_metrics(scores["train_y"], y_test))
    row["real_vs_ar_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("ar_pr_auc"))
    row["real_vs_shuffled_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("shuffled_pr_auc"))
    row["real_vs_random_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("random_pr_auc"))
    row["real_vs_ar_f1_delta"] = diff(row.get("real_f1"), row.get("ar_f1"))
    row["real_vs_shuffled_f1_delta"] = diff(row.get("real_f1"), row.get("shuffled_f1"))
    row["real_vs_random_f1_delta"] = diff(row.get("real_f1"), row.get("random_f1"))
    row["success_all_controls"] = all(
        value is not None and value > 0
        for value in (
            row["real_vs_ar_pr_auc_delta"],
            row["real_vs_shuffled_pr_auc_delta"],
            row["real_vs_random_pr_auc_delta"],
            row["real_vs_ar_f1_delta"],
            row["real_vs_shuffled_f1_delta"],
            row["real_vs_random_f1_delta"],
        )
    )
    return row, scores


def diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def evaluate_continuous(
    ctx: RetestContext,
    feature_label: str,
    split_label: str,
    target_name: str,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    matrices: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    scores = fit_scores(
        ctx,
        train_selected,
        train_y,
        test_selected,
        matrices["train_matrix"],
        matrices["test_matrix"],
        matrices["train_position"],
        matrices["test_position"],
        offset_seconds=0,
        rng=rng,
    )
    if scores is None:
        return None
    y_test = test_y[scores["test_keep"]]
    zero = np.zeros_like(y_test)
    row: dict[str, Any] = {
        "feature_mode": feature_label,
        "split": split_label,
        "target": target_name,
        "n": int(y_test.size),
    }
    for prefix, pred in (
        ("ar", scores["ar_test"]),
        ("real", scores["real_test"]),
        ("shuffled", scores["shuffled_test"]),
        ("random", scores["random_test"]),
        ("zero", zero),
    ):
        for key, value in regression_metrics(y_test, pred).items():
            row[f"{prefix}_{key}"] = finite_float(value)
    row["real_vs_ar_mae_delta"] = diff(row.get("ar_mae"), row.get("real_mae"))
    row["real_vs_shuffled_mae_delta"] = diff(row.get("shuffled_mae"), row.get("real_mae"))
    row["real_vs_random_mae_delta"] = diff(row.get("random_mae"), row.get("real_mae"))
    row["real_vs_zero_mae_delta"] = diff(row.get("zero_mae"), row.get("real_mae"))
    row["zero_baseline_pass_mae"] = (
        row["real_vs_zero_mae_delta"] is not None and row["real_vs_zero_mae_delta"] > 0
    )
    for threshold in EVENT_THRESHOLDS:
        true_sign = sign_class(y_test, threshold)
        pred_sign = sign_class(scores["real_test"], threshold)
        row[f"sign_accuracy_thr_{threshold}"] = float(np.mean(true_sign == pred_sign))
        train_move = (np.abs(scores["train_y"]) >= threshold).astype(np.int64)
        test_move = (np.abs(y_test) >= threshold).astype(np.int64)
        movement_metrics = event_metrics(
            train_move,
            np.abs(scores["real_train"]),
            test_move,
            np.abs(scores["real_test"]),
        )
        for key, value in movement_metrics.items():
            row[f"movement_thr_{threshold}_{key}"] = finite_float(value)
    nonzero = np.sign(y_test) != 0
    row["directional_accuracy_nonzero"] = (
        float(np.mean(np.sign(y_test[nonzero]) == np.sign(scores["real_test"][nonzero])))
        if np.any(nonzero)
        else None
    )
    return row


def per_video_event_rows(
    feature_label: str,
    split_label: str,
    target_name: str,
    threshold: float,
    scores: dict[str, Any],
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(scores["test_rows"]):
        grouped[str(row["video_id"])].append(index)
    output = []
    for video_id, indices in sorted(grouped.items(), key=lambda item: int(item[0])):
        idx = np.asarray(indices, dtype=np.int64)
        y = y_test[idx].astype(np.int64)
        if y.size == 0:
            continue
        video_row: dict[str, Any] = {
            "feature_mode": feature_label,
            "split": split_label,
            "target": target_name,
            "threshold": threshold,
            "video_id": video_id,
            "n": int(y.size),
            "event_count": float(np.sum(y)),
            "positive_rate": float(np.mean(y)),
            "enough_events": bool(np.sum(y) >= 3 and y.size >= 10),
        }
        for prefix, test_key in (
            ("ar", "ar_test"),
            ("real", "real_test"),
            ("shuffled", "shuffled_test"),
            ("random", "random_test"),
        ):
            # Use a video-local prevalence threshold only for describing the
            # held-out video's concentration; global split metrics use train-only
            # thresholds.
            if np.sum(y) == 0:
                metrics = {key: None for key in ("f1", "pr_auc", "recall")}
            else:
                local_threshold = np.quantile(scores[test_key][idx], 1.0 - min(0.95, max(0.05, np.mean(y))))
                pred = (scores[test_key][idx] >= local_threshold).astype(np.int64)
                metrics = binary_metrics_from_pred(y, pred)
                metrics["pr_auc"] = pr_auc(y, scores[test_key][idx])
            video_row[f"{prefix}_f1"] = finite_float(metrics.get("f1"))
            video_row[f"{prefix}_pr_auc"] = finite_float(metrics.get("pr_auc"))
            video_row[f"{prefix}_recall"] = finite_float(metrics.get("recall"))
        video_row["real_vs_ar_pr_auc_delta"] = diff(video_row.get("real_pr_auc"), video_row.get("ar_pr_auc"))
        video_row["real_vs_shuffled_pr_auc_delta"] = diff(video_row.get("real_pr_auc"), video_row.get("shuffled_pr_auc"))
        video_row["real_vs_random_pr_auc_delta"] = diff(video_row.get("real_pr_auc"), video_row.get("random_pr_auc"))
        video_row["win_vs_ar"] = video_row["real_vs_ar_pr_auc_delta"] is not None and video_row["real_vs_ar_pr_auc_delta"] > 0
        video_row["win_vs_shuffled"] = video_row["real_vs_shuffled_pr_auc_delta"] is not None and video_row["real_vs_shuffled_pr_auc_delta"] > 0
        video_row["win_vs_random"] = video_row["real_vs_random_pr_auc_delta"] is not None and video_row["real_vs_random_pr_auc_delta"] > 0
        output.append(video_row)
    return output


def split_matrices(
    ctx: RetestContext,
    base_feature_sets: dict[str, np.ndarray],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_mode: str,
) -> dict[str, Any]:
    train_idx = bench.row_indices(ctx.accepted_rows, train_rows)
    test_idx = bench.row_indices(ctx.accepted_rows, test_rows)
    split_feature_sets, metadata = bench.build_split_feature_sets(
        ctx.accepted_rows,
        base_feature_sets,
        train_idx,
        test_idx,
        feature_mode,
    )
    feature_key = next(iter(split_feature_sets.keys()))
    train_matrix, test_matrix = split_feature_sets[feature_key]
    return {
        "feature_key": feature_key,
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_position": {int(index): pos for pos, index in enumerate(train_idx)},
        "test_position": {int(index): pos for pos, index in enumerate(test_idx)},
        "metadata": metadata,
    }


def time_only_baseline(
    ctx: RetestContext,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
) -> dict[str, Any]:
    if len(train_selected) < 8 or len(test_selected) < 4:
        return {}
    train_x = bench.time_features(train_selected)
    test_x = bench.time_features(test_selected)
    train_scores, _ = bench.ridge_fit_predict(train_x, train_y, train_x)
    test_scores, _ = bench.ridge_fit_predict(train_x, train_y, test_x)
    return event_metrics(train_y, train_scores, test_y.astype(np.int64), test_scores)


def video_time_baseline(
    ctx: RetestContext,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
) -> dict[str, Any]:
    video_ids = sorted({str(row["video_id"]) for row in ctx.accepted_rows}, key=int)
    video_pos = {video_id: index for index, video_id in enumerate(video_ids)}

    def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
        time_x = bench.time_features(rows)
        one_hot = np.zeros((len(rows), len(video_ids)), dtype=np.float64)
        for index, row in enumerate(rows):
            one_hot[index, video_pos[str(row["video_id"])]] = 1.0
        return np.concatenate([time_x, one_hot], axis=1)

    if len(train_selected) < 8 or len(test_selected) < 4:
        return {}
    train_x = matrix(train_selected)
    test_x = matrix(test_selected)
    train_scores, _ = bench.ridge_fit_predict(train_x, train_y, train_x)
    test_scores, _ = bench.ridge_fit_predict(train_x, train_y, test_x)
    return event_metrics(train_y, train_scores, test_y.astype(np.int64), test_scores)


def shuffle_by_video(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    shuffled = values.copy()
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["video_id"])].append(index)
    for indices in grouped.values():
        if len(indices) > 1:
            idx = np.asarray(indices, dtype=np.int64)
            shuffled[idx] = shuffled[rng.permutation(idx)]
    return shuffled


def anti_leakage_controls(
    ctx: RetestContext,
    feature_label: str,
    split_label: str,
    target_name: str,
    threshold: float,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    matrices: dict[str, Any],
    spike_row: dict[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    train_keep, train_features = map_shifted_features(
        ctx,
        train_selected,
        matrices["train_position"],
        matrices["train_matrix"],
        0,
    )
    test_keep, test_features = map_shifted_features(
        ctx,
        test_selected,
        matrices["test_position"],
        matrices["test_matrix"],
        0,
    )
    if train_keep.size < 8 or test_keep.size < 4:
        return []
    train_rows = [train_selected[int(index)] for index in train_keep]
    test_rows = [test_selected[int(index)] for index in test_keep]
    y_train = train_y[train_keep].astype(np.float64)
    y_test = test_y[test_keep].astype(np.int64)
    train_ar = bench.autoregressive_features(
        ctx.accepted_rows, train_rows, "arousal", include_current=True
    )
    test_ar = bench.autoregressive_features(
        ctx.accepted_rows, test_rows, "arousal", include_current=True
    )
    real_train_x = np.concatenate([train_ar, train_features], axis=1)
    real_test_x = np.concatenate([test_ar, test_features], axis=1)

    def base(control: str, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "feature_mode": feature_label,
            "split": split_label,
            "target": target_name,
            "threshold": threshold,
            "control": control,
            "pr_auc": finite_float(metrics.get("pr_auc")),
            "f1": finite_float(metrics.get("f1")),
            "balanced_accuracy": finite_float(metrics.get("balanced_accuracy")),
            "precision": finite_float(metrics.get("precision")),
            "recall": finite_float(metrics.get("recall")),
            "event_count": finite_float(metrics.get("event_count")),
            "positive_class_rate": finite_float(metrics.get("positive_class_rate")),
        }

    rows = [
        base(
            "real_cortical",
            {
                "pr_auc": spike_row.get("real_pr_auc"),
                "f1": spike_row.get("real_f1"),
                "balanced_accuracy": spike_row.get("real_balanced_accuracy"),
                "precision": spike_row.get("real_precision"),
                "recall": spike_row.get("real_recall"),
                "event_count": spike_row.get("real_event_count"),
                "positive_class_rate": spike_row.get("real_positive_class_rate"),
            },
        )
    ]

    for control, y_control in (
        ("label_shuffle_across_videos", rng.permutation(y_train)),
        ("label_shuffle_within_video", shuffle_by_video(train_rows, y_train, rng)),
    ):
        train_scores, _ = bench.ridge_fit_predict(real_train_x, y_control, real_train_x)
        test_scores, _ = bench.ridge_fit_predict(real_train_x, y_control, real_test_x)
        rows.append(base(control, event_metrics(y_control, train_scores, y_test, test_scores)))

    for control, train_feature_control, test_feature_control in (
        (
            "feature_shuffle_across_videos",
            rng.permutation(train_features),
            rng.permutation(test_features),
        ),
        (
            "feature_shuffle_within_video",
            shuffle_by_video(train_rows, train_features, rng),
            shuffle_by_video(test_rows, test_features, rng),
        ),
    ):
        train_x = np.concatenate([train_ar, train_feature_control], axis=1)
        test_x = np.concatenate([test_ar, test_feature_control], axis=1)
        train_scores, _ = bench.ridge_fit_predict(train_x, y_train, train_x)
        test_scores, _ = bench.ridge_fit_predict(train_x, y_train, test_x)
        rows.append(base(control, event_metrics(y_train, train_scores, y_test, test_scores)))

    time_metrics = time_only_baseline(ctx, train_selected, train_y, test_selected, test_y)
    video_time_metrics = video_time_baseline(ctx, train_selected, train_y, test_selected, test_y)
    rows.append(base("timestamp_only", time_metrics))
    rows.append(base("video_id_time_only", video_time_metrics))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_89_complete_20260615.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_89_complete_20260615.report.json")
    parser.add_argument("--cache-dir", default=str(EXTERNAL_ROOT / "benchmarks" / "veatic" / "tribe_cache"))
    parser.add_argument("--output-prefix", default="benchmarks/veatic/veatic_89_retest_event_spike_core_20260616")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="auto", choices=("auto", "mps_solve", "cpu_pinv"))
    args = parser.parse_args()
    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend

    ctx = RetestContext(
        Path(args.manifest).expanduser().resolve(),
        Path(args.report).expanduser().resolve(),
        Path(args.cache_dir).expanduser().resolve(),
    )
    prefix = Path(args.output_prefix).expanduser().resolve()
    rng = np.random.default_rng(args.seed)
    percentiles = video_percentiles(ctx)

    diagnostics: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    onset_rows_out: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    for feature_label, feature_mode in FEATURE_MODES:
        print(f"[INFO] feature={feature_label}", flush=True)
        base_feature_sets = ctx.base_feature_sets(feature_mode)
        for split_label, split_name in SPLITS:
            train_rows, test_rows, gap_rows = fixed_rows(ctx.accepted_rows, split_name)
            matrices = split_matrices(ctx, base_feature_sets, train_rows, test_rows, feature_mode)
            for horizon in HORIZONS:
                train_selected, train_y = future_change_rows(ctx, train_rows, horizon)
                test_selected, test_y = future_change_rows(ctx, test_rows, horizon)
                cont = evaluate_continuous(
                    ctx,
                    feature_label,
                    split_label,
                    f"arousal__future_change_p{horizon}s",
                    train_selected,
                    train_y,
                    test_selected,
                    test_y,
                    matrices,
                    rng,
                )
                if cont is not None:
                    cont["gap_rows"] = gap_rows
                    diagnostics.append(cont)
                    for threshold in EVENT_THRESHOLDS:
                        train_event = (np.abs(train_y) >= threshold).astype(np.float64)
                        test_event = (np.abs(test_y) >= threshold).astype(np.float64)
                        event_row, _ = evaluate_event(
                            ctx,
                            feature_label,
                            split_label,
                            f"arousal__future_change_p{horizon}s_movement",
                            train_selected,
                            train_event,
                            test_selected,
                            test_event,
                            matrices,
                            threshold,
                            rng,
                        )
                        if event_row is not None:
                            event_row["horizon_seconds"] = horizon
                            diagnostics.append(event_row)
            for threshold in EVENT_THRESHOLDS:
                train_selected, train_y = future_spike_rows(ctx, train_rows, threshold)
                test_selected, test_y = future_spike_rows(ctx, test_rows, threshold)
                spike_row, scores = evaluate_event(
                    ctx,
                    feature_label,
                    split_label,
                    "arousal__future_spike_1_3s",
                    train_selected,
                    train_y,
                    test_selected,
                    test_y,
                    matrices,
                    threshold,
                    rng,
                )
                if spike_row is not None:
                    diagnostics.append(spike_row)
                    if split_label == "blocked" and abs(threshold - 0.05) < 1e-9 and scores is not None:
                        y_test = test_y[scores["test_keep"]]
                        per_video_rows.extend(
                            per_video_event_rows(
                                feature_label,
                                split_label,
                                "arousal__future_spike_1_3s",
                                threshold,
                                scores,
                                y_test,
                            )
                        )
                    if split_label == "blocked" and threshold in {0.05, 0.075}:
                        leakage_rows.extend(
                            anti_leakage_controls(
                                ctx,
                                feature_label,
                                split_label,
                                "arousal__future_spike_1_3s",
                                threshold,
                                train_selected,
                                train_y,
                                test_selected,
                                test_y,
                                matrices,
                                spike_row,
                                rng,
                            )
                        )
                if split_label == "blocked" and threshold in {0.05, 0.075}:
                    for offset in SHIFT_OFFSETS:
                        shifted, _ = evaluate_event(
                            ctx,
                            feature_label,
                            split_label,
                            "arousal__future_spike_1_3s",
                            train_selected,
                            train_y,
                            test_selected,
                            test_y,
                            matrices,
                            threshold,
                            rng,
                            offset_seconds=offset,
                        )
                        if shifted is not None:
                            shift_rows.append(shifted)
                if split_label == "blocked":
                    for leadup in ONSET_LEADS:
                        train_onset, train_onset_y = onset_rows(ctx, train_rows, threshold, leadup)
                        test_onset, test_onset_y = onset_rows(ctx, test_rows, threshold, leadup)
                        onset, _ = evaluate_event(
                            ctx,
                            feature_label,
                            split_label,
                            "arousal__future_spike_onset",
                            train_onset,
                            train_onset_y,
                            test_onset,
                            test_onset_y,
                            matrices,
                            threshold,
                            rng,
                        )
                        if onset is not None:
                            onset["leadup_seconds"] = leadup
                            onset_rows_out.append(onset)
            if split_label == "blocked":
                for horizon in (2, 3):
                    train_selected, train_y = future_change_rows(ctx, train_rows, horizon)
                    test_selected, test_y = future_change_rows(ctx, test_rows, horizon)
                    for threshold in (0.05,):
                        for offset in SHIFT_OFFSETS:
                            train_event = (np.abs(train_y) >= threshold).astype(np.float64)
                            test_event = (np.abs(test_y) >= threshold).astype(np.float64)
                            shifted, _ = evaluate_event(
                                ctx,
                                feature_label,
                                split_label,
                                f"arousal__future_change_p{horizon}s_movement",
                                train_selected,
                                train_event,
                                test_selected,
                                test_event,
                                matrices,
                                threshold,
                                rng,
                                offset_seconds=offset,
                            )
                            if shifted is not None:
                                shifted["horizon_seconds"] = horizon
                                shift_rows.append(shifted)
                for variant in (
                    "future_minus_rolling_baseline",
                    "future_slope_minus_prior_slope",
                    "future_acceleration",
                    "future_rank_percentile_delta",
                    "future_change_local_volatility",
                ):
                    for window in LOCAL_WINDOWS:
                        for threshold in EVENT_THRESHOLDS:
                            train_local, train_local_y = local_target_rows(
                                ctx, train_rows, variant, 2, window, threshold, percentiles
                            )
                            test_local, test_local_y = local_target_rows(
                                ctx, test_rows, variant, 2, window, threshold, percentiles
                            )
                            local, _ = evaluate_event(
                                ctx,
                                feature_label,
                                split_label,
                                f"local__{variant}",
                                train_local,
                                train_local_y,
                                test_local,
                                test_local_y,
                                matrices,
                                threshold,
                                rng,
                            )
                            if local is not None:
                                local["window_seconds"] = window
                                local["horizon_seconds"] = 2
                                local_rows.append(local)

    json_path = prefix.with_suffix(".json")
    diagnostics_path = prefix.with_suffix(".diagnostics.csv")
    shift_path = prefix.with_suffix(".shift_audit.csv")
    per_video_path = prefix.with_suffix(".per_video.csv")
    onset_path = prefix.with_suffix(".onset_only.csv")
    local_path = prefix.with_suffix(".local_targets.csv")
    md_path = prefix.with_suffix(".md")

    payload = {
        "schema_version": "veatic_event_spike_retest_v2",
        "manifest": str(ctx.manifest),
        "cache_dir": str(ctx.cache_dir),
        "methodology": {
            "classification_decision_thresholds": "selected on train predictions only by F1, then applied to test",
            "event_label_thresholds": list(EVENT_THRESHOLDS),
            "shift_offsets_seconds": list(SHIFT_OFFSETS),
            "feature_modes": [label for label, _ in FEATURE_MODES],
            "splits": [label for label, _ in SPLITS],
            "leave_video_out": "TODO: not run in this retest because no cheap existing implementation was present; add as a separate confirmatory pass after the blocked and official diagnostics.",
            "anti_leakage_checks": [
                "label_shuffle_within_video",
                "label_shuffle_across_videos",
                "feature_shuffle_within_video",
                "feature_shuffle_across_videos",
                "timestamp_only",
                "video_id_time_only",
            ],
            "architectures_added": False,
            "feature_extraction_changed": False,
        },
        "diagnostics": diagnostics,
        "shift_audit": shift_rows,
        "per_video": per_video_rows,
        "onset_only": onset_rows_out,
        "local_targets": local_rows,
        "anti_leakage": leakage_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(diagnostics_path, diagnostics)
    write_csv(shift_path, shift_rows)
    write_csv(per_video_path, per_video_rows)
    write_csv(onset_path, onset_rows_out)
    write_csv(local_path, local_rows)

    write_markdown(
        md_path,
        json_path,
        diagnostics_path,
        shift_path,
        per_video_path,
        onset_path,
        local_path,
        diagnostics,
        shift_rows,
        per_video_rows,
        onset_rows_out,
        local_rows,
        leakage_rows,
    )
    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "json": str(json_path),
                "diagnostics_csv": str(diagnostics_path),
                "shift_csv": str(shift_path),
                "per_video_csv": str(per_video_path),
                "onset_csv": str(onset_path),
                "local_csv": str(local_path),
                "diagnostic_rows": len(diagnostics),
                "shift_rows": len(shift_rows),
                "per_video_rows": len(per_video_rows),
                "onset_rows": len(onset_rows_out),
                "local_rows": len(local_rows),
            },
            indent=2,
        )
    )


def best_rows(
    rows: list[dict[str, Any]],
    *,
    split: str | None = None,
    target: str | None = None,
    threshold: float | None = None,
    metric: str = "real_pr_auc",
    limit: int = 12,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        if target is not None and row.get("target") != target:
            continue
        if threshold is not None and row.get("threshold") != threshold:
            continue
        if row.get(metric) is None:
            continue
        selected.append(row)
    return sorted(selected, key=lambda item: item.get(metric) or -1.0, reverse=True)[:limit]


def pass_row(row: dict[str, Any]) -> bool:
    required = (
        "real_vs_ar_pr_auc_delta",
        "real_vs_shuffled_pr_auc_delta",
        "real_vs_random_pr_auc_delta",
        "real_vs_ar_f1_delta",
        "real_vs_shuffled_f1_delta",
        "real_vs_random_f1_delta",
    )
    return all(row.get(key) is not None and row[key] > 0 for key in required)


def write_markdown(
    path: Path,
    json_path: Path,
    diagnostics_path: Path,
    shift_path: Path,
    per_video_path: Path,
    onset_path: Path,
    local_path: Path,
    diagnostics: list[dict[str, Any]],
    shift_rows: list[dict[str, Any]],
    per_video_rows: list[dict[str, Any]],
    onset_rows: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
) -> None:
    blocked_spikes = [
        row
        for row in diagnostics
        if row.get("split") == "blocked"
        and row.get("target") == "arousal__future_spike_1_3s"
        and row.get("threshold") in {0.05, 0.075}
    ]
    blocked_passes = [row for row in blocked_spikes if pass_row(row)]
    best_blocked = best_rows(
        diagnostics,
        split="blocked",
        target="arousal__future_spike_1_3s",
        metric="real_pr_auc",
        limit=8,
    )
    official_spikes = [
        row
        for row in diagnostics
        if row.get("split") == "official"
        and row.get("target") == "arousal__future_spike_1_3s"
        and row.get("threshold") in {0.05, 0.075}
    ]
    continuous = [
        row
        for row in diagnostics
        if row.get("target", "").startswith("arousal__future_change_p")
        and not row.get("target", "").endswith("_movement")
    ]
    zero_passes = sum(1 for row in continuous if row.get("zero_baseline_pass_mae") is True)
    movement = [
        row
        for row in diagnostics
        if row.get("target", "").endswith("_movement")
        and row.get("split") == "blocked"
    ]
    onset_best = best_rows(onset_rows, split="blocked", target="arousal__future_spike_onset", metric="real_pr_auc", limit=8)
    local_best = best_rows(local_rows, split="blocked", metric="real_pr_auc", limit=8)
    per_video_enough = [row for row in per_video_rows if row.get("enough_events") is True]
    per_video_wins = sum(1 for row in per_video_enough if row.get("win_vs_ar") and row.get("win_vs_shuffled") and row.get("win_vs_random"))
    leakage_focus = [
        row
        for row in leakage_rows
        if row.get("split") == "blocked"
        and row.get("target") == "arousal__future_spike_1_3s"
        and row.get("threshold") in {0.05, 0.075}
    ]
    best_shift = best_shift_summary(shift_rows)
    positive_best = sum(1 for row in best_shift if row["best_offset"] > 0)
    negative_best = sum(1 for row in best_shift if row["best_offset"] < 0)
    zero_best = sum(1 for row in best_shift if row["best_offset"] == 0)
    verdict = (
        "promising but needs alignment fix"
        if blocked_passes and (positive_best or negative_best) and zero_best < max(positive_best, negative_best)
        else "weak but worth checking on 124"
        if blocked_passes
        else "dead / likely artifact"
    )

    lines = [
        "# VEATIC Event/Spike Retest",
        "",
        "## SECTION 1: Executive Verdict",
        "",
        f"Classification: **{verdict}**.",
        "",
        (
            "Supported claim: cortical/TRIBE features show useful signal for upcoming "
            "arousal spike/event ranking under blocked validation when evaluated as "
            "events rather than exact future decimal values."
            if blocked_passes
            else "Supported claim: no robust event/spike claim is supported under the strict all-control criterion."
        ),
        "",
        "Not supported: exact continuous future arousal-change prediction as the primary claim, especially where zero-change MAE remains competitive.",
        "",
        "Decision thresholds for classification were calibrated on train predictions only. No test-threshold tuning was used.",
        "",
        "Leave-video-out was not run in this pass because no cheap existing implementation was present; add it as a separate confirmatory TODO after the blocked/official diagnostics.",
        "",
        "## SECTION 2: Main Blocked Spike Result",
        "",
        "| Feature | Thr | PR-AUC AR | PR-AUC real | PR-AUC shuf | PR-AUC rand | F1 real | BalAcc real | Prec real | Rec real | Pass controls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in blocked_spikes:
        lines.append(
            f"| {row['feature_mode']} | {format_value(row['threshold'])} | "
            f"{format_value(row.get('ar_pr_auc'))} | {format_value(row.get('real_pr_auc'))} | "
            f"{format_value(row.get('shuffled_pr_auc'))} | {format_value(row.get('random_pr_auc'))} | "
            f"{format_value(row.get('real_f1'))} | {format_value(row.get('real_balanced_accuracy'))} | "
            f"{format_value(row.get('real_precision'))} | {format_value(row.get('real_recall'))} | {pass_row(row)} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 3: Official Split Result",
            "",
            "| Feature | Thr | PR-AUC AR | PR-AUC real | F1 real | BalAcc real | Rec real | Pass controls |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in official_spikes:
        lines.append(
            f"| {row['feature_mode']} | {format_value(row['threshold'])} | "
            f"{format_value(row.get('ar_pr_auc'))} | {format_value(row.get('real_pr_auc'))} | "
            f"{format_value(row.get('real_f1'))} | {format_value(row.get('real_balanced_accuracy'))} | "
            f"{format_value(row.get('real_recall'))} | {pass_row(row)} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 4: Continuous Future-Change Diagnostic",
            "",
            f"Zero-baseline MAE was beaten in {zero_passes}/{len(continuous)} continuous checks. Continuous MAE should remain diagnostic, not the primary verdict.",
            "",
            "| Feature | Split | Target | Real MAE | Zero MAE | Zero pass | Real-vs-AR MAE delta |",
            "|---|---|---|---:|---:|---|---:|",
        ]
    )
    for row in continuous:
        if row.get("split") == "blocked":
            lines.append(
                f"| {row['feature_mode']} | {row['split']} | `{row['target']}` | "
                f"{format_value(row.get('real_mae'))} | {format_value(row.get('zero_mae'))} | "
                f"{row.get('zero_baseline_pass_mae')} | {format_value(row.get('real_vs_ar_mae_delta'))} |"
            )
    lines.extend(
        [
            "",
            "## SECTION 5: Movement/Event Threshold Sweep",
            "",
            "Best blocked movement rows by real PR-AUC:",
            "",
            "| Feature | Target | Thr | PR-AUC real | F1 real | Recall real | Pass controls |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in best_rows(movement, metric="real_pr_auc", limit=12):
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {format_value(row.get('threshold'))} | "
            f"{format_value(row.get('real_pr_auc'))} | {format_value(row.get('real_f1'))} | "
            f"{format_value(row.get('real_recall'))} | {pass_row(row)} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 6: Shift/Alignment Audit",
            "",
            f"Best-shift counts: negative={negative_best}, zero={zero_best}, positive={positive_best}. Non-zero best shifts are not automatically leakage; they may reflect annotation lag, smoothing lag, feature lag, or anticipatory signal.",
            "",
            "| Feature | Target | Thr | Best offset | Best PR-AUC | 0s PR-AUC | 0s competitive |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in best_shift:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {format_value(row.get('threshold'))} | "
            f"{row['best_offset']} | {format_value(row['best_pr_auc'])} | "
            f"{format_value(row['zero_pr_auc'])} | {row['zero_competitive']} |"
        )
    lines.extend(
        [
            "",
            "Anti-leakage checks for blocked spike thresholds:",
            "",
            "| Feature | Thr | Control | PR-AUC | F1 | BalAcc | Recall |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in leakage_focus:
        lines.append(
            f"| {row['feature_mode']} | {format_value(row.get('threshold'))} | `{row.get('control')}` | "
            f"{format_value(row.get('pr_auc'))} | {format_value(row.get('f1'))} | "
            f"{format_value(row.get('balanced_accuracy'))} | {format_value(row.get('recall'))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 7: Onset-Only Evaluation",
            "",
            "| Feature | Thr | Lead-up | PR-AUC real | F1 real | BalAcc real | Rec real | Event count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in onset_best:
        lines.append(
            f"| {row['feature_mode']} | {format_value(row.get('threshold'))} | {row.get('leadup_seconds')} | "
            f"{format_value(row.get('real_pr_auc'))} | {format_value(row.get('real_f1'))} | "
            f"{format_value(row.get('real_balanced_accuracy'))} | {format_value(row.get('real_recall'))} | "
            f"{format_value(row.get('real_event_count'))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 8: Local-Normalized Target Evaluation",
            "",
            "| Feature | Variant | Window | Thr | PR-AUC real | F1 real | Pass controls |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in local_best:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {row.get('window_seconds')} | "
            f"{format_value(row.get('threshold'))} | {format_value(row.get('real_pr_auc'))} | "
            f"{format_value(row.get('real_f1'))} | {pass_row(row)} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 9: Per-Video Robustness",
            "",
            f"Enough-event video rows: {len(per_video_enough)}. Wins versus AR, shuffled, and random by per-video PR-AUC: {per_video_wins}/{len(per_video_enough)}.",
            "",
            "Low-event videos are explicitly flagged in the per-video CSV; do not treat their per-video F1/PR-AUC as stable.",
            "",
            "## SECTION 10: Final Recommendation",
            "",
            f"Recommendation: **{verdict}**.",
            "",
            "For follow-up or replication:",
            "",
            "1. Blocked arousal__future_spike_1_3s at thresholds 0.05 and 0.075 for all four feature modes.",
            "2. Blocked movement threshold sweep for p2/p3 future-change at 0.05 and 0.075.",
            "3. Shift audit at -5,-3,-2,-1,0,+1,+2,+3,+5 before making any investor-facing claim.",
            "4. Onset-only spike detection at lead-up windows 2s, 3s, and 5s.",
            "5. Per-video contribution audit to verify gains are not dominated by a few high-event videos.",
            "",
            "## Output Files",
            "",
            f"- JSON: `{json_path}`",
            f"- Diagnostics CSV: `{diagnostics_path}`",
            f"- Shift audit CSV: `{shift_path}`",
            f"- Per-video CSV: `{per_video_path}`",
            f"- Onset-only CSV: `{onset_path}`",
            f"- Local targets CSV: `{local_path}`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def best_shift_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["feature_mode"], row["target"], float(row["threshold"]))].append(row)
    output = []
    for (feature_mode, target, threshold), values in sorted(grouped.items()):
        best = max(values, key=lambda item: item.get("real_pr_auc") or -1.0)
        zero = next((item for item in values if item.get("offset_seconds") == 0), None)
        best_value = best.get("real_pr_auc")
        zero_value = zero.get("real_pr_auc") if zero else None
        output.append(
            {
                "feature_mode": feature_mode,
                "target": target,
                "threshold": threshold,
                "best_offset": int(best["offset_seconds"]),
                "best_pr_auc": best_value,
                "zero_pr_auc": zero_value,
                "zero_competitive": (
                    zero_value is not None
                    and best_value is not None
                    and zero_value >= 0.95 * best_value
                ),
            }
        )
    return output


if __name__ == "__main__":
    main()
