"""Event-conditioned VEATIC retest.

This is a diagnostic layer on top of the full-frame benchmark. It reuses the
same cached TRIBE/cortical features and ridge/PCA machinery, then evaluates
model scores on event, pre-event, stable, and balanced event-vs-stable regions.
Decision thresholds are selected on train predictions only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
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

FEATURE_MODES = retest.FEATURE_MODES
SPLITS = retest.SPLITS
EVENT_THRESHOLDS = retest.EVENT_THRESHOLDS
HORIZONS = retest.HORIZONS
PRE_EVENT_LEADS = (1, 2, 3, 5)
BALANCED_RATIOS = (1, 2, 3)
BALANCED_SEEDS = tuple(range(20))
PRIMARY_FOCUS = {
    ("arousal__future_spike_1_3s", 0.05),
    ("arousal__future_spike_1_3s", 0.075),
    ("arousal__future_change_p2s_movement", 0.05),
    ("arousal__future_change_p3s_movement", 0.05),
    ("arousal__future_change_p3s_movement", 0.075),
}
ANTI_LEAKAGE_MASKS = ("all_frames", "event_plus_pre_3s", "pre_event_1s", "pre_event_2s", "pre_event_3s", "pre_event_5s")


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def strict_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_true.size == 0:
        return None
    positives = int(np.sum(y_true == 1))
    if positives == 0 or positives == y_true.size:
        return None
    return retest.pr_auc(y_true, scores)


def event_subset_metrics(
    train_y: np.ndarray,
    train_scores: np.ndarray,
    eval_y: np.ndarray,
    eval_scores: np.ndarray,
) -> dict[str, float | None]:
    threshold = retest.best_train_threshold(train_y.astype(np.int64), train_scores)
    pred = (eval_scores >= threshold).astype(np.int64)
    metrics = retest.binary_metrics_from_pred(eval_y.astype(np.int64), pred)
    metrics.update(
        {
            "decision_threshold": threshold,
            "pr_auc": strict_pr_auc(eval_y, eval_scores),
            "positive_rate": float(np.mean(eval_y)) if eval_y.size else None,
            "event_count": float(np.sum(eval_y)),
            "n": float(eval_y.size),
            "n_pos": float(np.sum(eval_y == 1)),
            "n_neg": float(np.sum(eval_y == 0)),
        }
    )
    metrics.update(retest.topk_recall(eval_y.astype(np.int64), eval_scores))
    return metrics


def majority_subset_metrics(train_y: np.ndarray, eval_y: np.ndarray) -> dict[str, float | None]:
    majority = int(np.mean(train_y) >= 0.5) if train_y.size else 0
    pred = np.full(eval_y.shape, majority, dtype=np.int64)
    metrics = retest.binary_metrics_from_pred(eval_y.astype(np.int64), pred)
    metrics.update(
        {
            "decision_threshold": None,
            "pr_auc": None,
            "positive_rate": float(np.mean(eval_y)) if eval_y.size else None,
            "event_count": float(np.sum(eval_y)),
            "n": float(eval_y.size),
            "n_pos": float(np.sum(eval_y == 1)),
            "n_neg": float(np.sum(eval_y == 0)),
        }
    )
    metrics.update(retest.topk_recall(eval_y.astype(np.int64), np.zeros(eval_y.shape)))
    return metrics


def add_metric_row(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    model: str,
    metrics: dict[str, Any],
) -> None:
    row = dict(base)
    row["model"] = model
    for key, value in metrics.items():
        row[key] = finite(value)
    rows.append(row)


def event_magnitude_for_rows(
    ctx: retest.RetestContext,
    selected_rows: list[dict[str, Any]],
    target_name: str,
    horizon: int | None,
) -> np.ndarray:
    values = []
    for row in selected_rows:
        video_id = str(row["video_id"])
        second = int(round(float(row["time_start_seconds"])))
        current = ctx.lookup.get((video_id, second))
        if current is None:
            values.append(np.nan)
            continue
        if target_name == "arousal__future_spike_1_3s":
            futures = [
                ctx.lookup.get((video_id, second + offset))
                for offset in (1, 2, 3)
            ]
            futures = [value for value in futures if value is not None]
            values.append(max(abs(value - current) for value in futures) if futures else np.nan)
        else:
            assert horizon is not None
            future = ctx.lookup.get((video_id, second + horizon))
            values.append(abs(future - current) if future is not None else np.nan)
    return np.asarray(values, dtype=np.float64)


def build_region_masks(
    test_rows: list[dict[str, Any]],
    y_event: np.ndarray,
    stable_magnitude: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    n = y_event.size
    key_to_index = {
        (str(row["video_id"]), int(round(float(row["time_start_seconds"])))): index
        for index, row in enumerate(test_rows)
    }
    event_mask = y_event.astype(bool)
    stable_mask = (~event_mask) & np.isfinite(stable_magnitude) & (stable_magnitude < threshold)
    masks = {
        "all_frames": np.ones(n, dtype=bool),
        "stable_negative_only": stable_mask,
        "event_only": event_mask,
    }
    pre_masks = {}
    for lead in PRE_EVENT_LEADS:
        mask = np.zeros(n, dtype=bool)
        for index, is_event in enumerate(event_mask):
            if not is_event:
                continue
            row = test_rows[index]
            key = (str(row["video_id"]), int(round(float(row["time_start_seconds"]))) - lead)
            pre_index = key_to_index.get(key)
            if pre_index is None:
                continue
            if event_mask[pre_index]:
                continue
            mask[pre_index] = True
        pre_masks[f"pre_event_{lead}s"] = mask
    masks.update(pre_masks)
    masks["event_plus_pre_3s"] = (
        event_mask | pre_masks["pre_event_1s"] | pre_masks["pre_event_2s"] | pre_masks["pre_event_3s"]
    )
    return masks


def labels_for_mask(mask_name: str, masks: dict[str, np.ndarray], y_event: np.ndarray) -> np.ndarray:
    if mask_name == "all_frames":
        return y_event.astype(np.int64)
    if mask_name == "stable_negative_only":
        return np.zeros(int(np.sum(masks[mask_name])), dtype=np.int64)
    return np.ones(int(np.sum(masks[mask_name])), dtype=np.int64)


def model_score_sets(
    ctx: retest.RetestContext,
    train_selected: list[dict[str, Any]],
    train_y: np.ndarray,
    test_selected: list[dict[str, Any]],
    test_y: np.ndarray,
    matrices: dict[str, Any],
    rng: np.random.Generator,
    *,
    include_anti_leakage: bool,
) -> dict[str, Any] | None:
    scores = retest.fit_scores(
        ctx,
        train_selected,
        train_y,
        test_selected,
        matrices["train_matrix"],
        matrices["test_matrix"],
        matrices["train_position"],
        matrices["test_position"],
        rng=rng,
    )
    if scores is None:
        return None
    output = {
        "test_rows": scores["test_rows"],
        "test_keep": scores["test_keep"],
        "train_y": scores["train_y"].astype(np.float64),
        "true_test_y": test_y[scores["test_keep"]].astype(np.float64),
        "models": {
            "ar": (scores["train_y"], scores["ar_train"], scores["ar_test"]),
            "real": (scores["train_y"], scores["real_train"], scores["real_test"]),
            "shuffled": (scores["train_y"], scores["shuffled_train"], scores["shuffled_test"]),
            "random": (scores["train_y"], scores["random_train"], scores["random_test"]),
        },
    }
    if not include_anti_leakage:
        return output

    train_keep, train_features = retest.map_shifted_features(
        ctx,
        train_selected,
        matrices["train_position"],
        matrices["train_matrix"],
        0,
    )
    test_keep, test_features = retest.map_shifted_features(
        ctx,
        test_selected,
        matrices["test_position"],
        matrices["test_matrix"],
        0,
    )
    if not np.array_equal(train_keep, np.arange(len(scores["train_y"]))) or not np.array_equal(test_keep, scores["test_keep"]):
        return output
    train_rows = [train_selected[int(index)] for index in train_keep]
    test_rows = scores["test_rows"]
    y_train = scores["train_y"].astype(np.float64)
    train_ar = bench.autoregressive_features(ctx.accepted_rows, train_rows, "arousal", include_current=True)
    test_ar = bench.autoregressive_features(ctx.accepted_rows, test_rows, "arousal", include_current=True)
    real_train_x = np.concatenate([train_ar, train_features], axis=1)
    real_test_x = np.concatenate([test_ar, test_features], axis=1)

    for name, y_control in (
        ("label_shuffle_across_videos", rng.permutation(y_train)),
        ("label_shuffle_within_video", retest.shuffle_by_video(train_rows, y_train, rng)),
    ):
        train_scores, _ = bench.ridge_fit_predict(real_train_x, y_control, real_train_x)
        test_scores, _ = bench.ridge_fit_predict(real_train_x, y_control, real_test_x)
        output["models"][name] = (y_control, train_scores, test_scores)

    for name, train_feature_control, test_feature_control in (
        ("feature_shuffle_across_videos", rng.permutation(train_features), rng.permutation(test_features)),
        (
            "feature_shuffle_within_video",
            retest.shuffle_by_video(train_rows, train_features, rng),
            retest.shuffle_by_video(test_rows, test_features, rng),
        ),
    ):
        train_x = np.concatenate([train_ar, train_feature_control], axis=1)
        test_x = np.concatenate([test_ar, test_feature_control], axis=1)
        train_scores, _ = bench.ridge_fit_predict(train_x, y_train, train_x)
        test_scores, _ = bench.ridge_fit_predict(train_x, y_train, test_x)
        output["models"][name] = (y_train, train_scores, test_scores)

    baseline_matrices = [
        ("timestamp_only", bench.time_features(train_rows), bench.time_features(test_rows)),
    ]
    video_train_x, video_test_x = video_time_matrix(ctx, train_rows, test_rows)
    baseline_matrices.append(("video_id_time_only", video_train_x, video_test_x))
    for name, train_x, test_x in baseline_matrices:
        train_scores, _ = bench.ridge_fit_predict(train_x, y_train, train_x)
        test_scores, _ = bench.ridge_fit_predict(train_x, y_train, test_x)
        output["models"][name] = (y_train, train_scores, test_scores)
    return output


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


def evaluate_event_masks(
    output_rows: list[dict[str, Any]],
    base: dict[str, Any],
    score_sets: dict[str, Any],
    masks: dict[str, np.ndarray],
    *,
    anti_leakage_only: bool = False,
) -> None:
    y_event = score_sets["true_test_y"].astype(np.int64)
    base_models = {"ar", "real", "shuffled", "random", "majority"}
    models = score_sets["models"]
    for mask_name, mask in masks.items():
        idx = np.flatnonzero(mask)
        eval_y = labels_for_mask(mask_name, masks, y_event)
        if idx.size != eval_y.size:
            raise ValueError(f"mask/label size mismatch for {mask_name}")
        row_base = dict(base)
        row_base.update({"mask": mask_name, "task_type": "classification"})
        if idx.size == 0:
            continue
        if not anti_leakage_only:
            for model in ("ar", "real", "shuffled", "random"):
                train_y, train_scores, test_scores = models[model]
                add_metric_row(
                    output_rows,
                    row_base,
                    model,
                    event_subset_metrics(train_y, train_scores, eval_y, test_scores[idx]),
                )
            add_metric_row(
                output_rows,
                row_base,
                "majority",
                majority_subset_metrics(score_sets["train_y"], eval_y),
            )
        for model, (train_y, train_scores, test_scores) in models.items():
            if model in base_models:
                continue
            if mask_name not in ANTI_LEAKAGE_MASKS:
                continue
            add_metric_row(
                output_rows,
                row_base,
                model,
                event_subset_metrics(train_y, train_scores, eval_y, test_scores[idx]),
            )


def regression_subset_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return retest.regression_metrics(y_true, y_pred)


def evaluate_continuous_masks(
    output_rows: list[dict[str, Any]],
    base: dict[str, Any],
    score_sets: dict[str, Any],
    y_cont: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    full_y = y_cont[score_sets["test_keep"]].astype(np.float64)
    for mask_name, mask in masks.items():
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        y = full_y[idx]
        row_base = dict(base)
        row_base.update({"mask": mask_name, "task_type": "continuous"})
        predictions = {
            "ar": score_sets["models"]["ar"][2][idx],
            "real": score_sets["models"]["real"][2][idx],
            "shuffled": score_sets["models"]["shuffled"][2][idx],
            "random": score_sets["models"]["random"][2][idx],
            "zero": np.zeros(idx.size, dtype=np.float64),
        }
        metrics_by_model = {
            model: regression_subset_metrics(y, pred)
            for model, pred in predictions.items()
        }
        for model, metrics in metrics_by_model.items():
            row = dict(row_base)
            row["model"] = model
            row["n"] = float(idx.size)
            row["event_count"] = float(np.sum(mask & masks["event_only"]))
            for key, value in metrics.items():
                row[key] = finite(value)
            if model == "real":
                row["real_vs_ar_mae_delta"] = diff(metrics_by_model["ar"]["mae"], metrics["mae"])
                row["real_vs_shuffled_mae_delta"] = diff(metrics_by_model["shuffled"]["mae"], metrics["mae"])
                row["real_vs_random_mae_delta"] = diff(metrics_by_model["random"]["mae"], metrics["mae"])
                row["real_vs_zero_mae_delta"] = diff(metrics_by_model["zero"]["mae"], metrics["mae"])
            output_rows.append(row)


def evaluate_balanced_sampling(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    score_sets: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> None:
    positive_mask = masks["event_plus_pre_3s"]
    stable_mask = masks["stable_negative_only"]
    pos_idx = np.flatnonzero(positive_mask)
    stable_idx = np.flatnonzero(stable_mask)
    if pos_idx.size == 0 or stable_idx.size == 0:
        return
    for ratio in BALANCED_RATIOS:
        seed_metrics: dict[str, list[dict[str, float | None]]] = defaultdict(list)
        for seed in BALANCED_SEEDS:
            rng = np.random.default_rng(seed)
            neg_n = min(stable_idx.size, pos_idx.size * ratio)
            neg_idx = rng.choice(stable_idx, size=neg_n, replace=False)
            idx = np.concatenate([pos_idx, neg_idx])
            eval_y = np.concatenate([
                np.ones(pos_idx.size, dtype=np.int64),
                np.zeros(neg_idx.size, dtype=np.int64),
            ])
            for model in ("ar", "real", "shuffled", "random"):
                train_y, train_scores, test_scores = score_sets["models"][model]
                seed_metrics[model].append(
                    event_subset_metrics(train_y, train_scores, eval_y, test_scores[idx])
                )
            seed_metrics["majority"].append(majority_subset_metrics(score_sets["train_y"], eval_y))
        for model, values in seed_metrics.items():
            row = dict(base)
            row.update(
                {
                    "task_type": "classification",
                    "mask": "balanced_event_vs_stable",
                    "negative_ratio": f"1:{ratio}",
                    "model": model,
                    "seed_count": len(values),
                    "n_pos": float(pos_idx.size),
                    "n_neg_mean": float(min(stable_idx.size, pos_idx.size * ratio)),
                }
            )
            for metric in ("pr_auc", "f1", "balanced_accuracy", "precision", "recall", "accuracy"):
                arr = np.asarray([value[metric] for value in values if value.get(metric) is not None], dtype=np.float64)
                row[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None
                row[f"{metric}_std"] = float(np.std(arr)) if arr.size else None
            rows.append(row)


def per_video_rows(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    score_sets: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> None:
    target_mask = masks["event_plus_pre_3s"]
    stable_mask = masks["stable_negative_only"]
    eval_mask = target_mask | stable_mask
    idx_all = np.flatnonzero(eval_mask)
    if idx_all.size == 0:
        return
    y_eval_full = np.zeros(target_mask.shape, dtype=np.int64)
    y_eval_full[target_mask] = 1
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in idx_all:
        grouped[str(score_sets["test_rows"][int(index)]["video_id"])].append(int(index))
    for video_id, indices in sorted(grouped.items(), key=lambda item: int(item[0])):
        idx = np.asarray(indices, dtype=np.int64)
        y = y_eval_full[idx]
        if y.size < 6:
            continue
        row_base = dict(base)
        row_base.update(
            {
                "mask": "event_plus_pre_3s_vs_stable",
                "video_id": video_id,
                "n": int(y.size),
                "event_count": float(np.sum(y)),
                "positive_rate": float(np.mean(y)),
                "enough_events": bool(np.sum(y) >= 3 and np.sum(y == 0) >= 3),
            }
        )
        model_metrics = {}
        for model in ("ar", "real", "shuffled", "random"):
            _, _, test_scores = score_sets["models"][model]
            model_metrics[model] = {
                "pr_auc": strict_pr_auc(y, test_scores[idx]),
                "top_10pct_recall": retest.topk_recall(y, test_scores[idx]).get("top_10pct_recall"),
            }
        row = dict(row_base)
        for model, metrics in model_metrics.items():
            row[f"{model}_pr_auc"] = finite(metrics["pr_auc"])
            row[f"{model}_top_10pct_recall"] = finite(metrics["top_10pct_recall"])
        row["real_vs_ar_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("ar_pr_auc"))
        row["real_vs_shuffled_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("shuffled_pr_auc"))
        row["real_vs_random_pr_auc_delta"] = diff(row.get("real_pr_auc"), row.get("random_pr_auc"))
        row["win_vs_ar"] = row["real_vs_ar_pr_auc_delta"] is not None and row["real_vs_ar_pr_auc_delta"] > 0
        row["win_vs_shuffled"] = row["real_vs_shuffled_pr_auc_delta"] is not None and row["real_vs_shuffled_pr_auc_delta"] > 0
        row["win_vs_random"] = row["real_vs_random_pr_auc_delta"] is not None and row["real_vs_random_pr_auc_delta"] > 0
        rows.append(row)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def row_lookup(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str,
    split: str,
    target: str,
    threshold: float,
    mask: str,
    model: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("feature_mode") == feature_mode
            and row.get("split") == split
            and row.get("target") == target
            and row.get("threshold") == threshold
            and row.get("mask") == mask
            and row.get("model") == model
        ):
            return row
    return None


def model_delta_rows(rows: list[dict[str, Any]], metric: str = "pr_auc") -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    keys = ("feature_mode", "split", "target", "threshold", "mask", "task_type")
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)][row["model"]] = row
    output = []
    for key, models in grouped.items():
        if "real" not in models:
            continue
        real = models["real"].get(metric)
        if real is None:
            continue
        record = {keys[index]: key[index] for index in range(len(keys))}
        record["real"] = real
        for model in ("ar", "shuffled", "random"):
            record[model] = models.get(model, {}).get(metric)
            record[f"real_vs_{model}"] = diff(real, record[model])
        output.append(record)
    return output


def write_markdown(
    path: Path,
    event_rows: list[dict[str, Any]],
    balanced_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> None:
    focus = [
        row for row in model_delta_rows(event_rows)
        if row["split"] == "blocked"
        and (row["target"], row["threshold"]) in PRIMARY_FOCUS
        and row["mask"] in {"all_frames", "event_plus_pre_3s", "pre_event_1s", "pre_event_2s", "pre_event_3s", "pre_event_5s"}
    ]
    full_vs_event = [
        row for row in focus
        if row["mask"] in {"all_frames", "event_plus_pre_3s"}
    ]
    pre_event = [
        row for row in focus
        if str(row["mask"]).startswith("pre_event_")
    ]
    event_only = [
        row for row in event_rows
        if row["split"] == "blocked"
        and row["mask"] == "event_only"
        and row["model"] == "real"
        and (row["target"], row["threshold"]) in PRIMARY_FOCUS
    ]
    pre_event_direct = [
        row for row in event_rows
        if row["split"] == "blocked"
        and str(row["mask"]).startswith("pre_event_")
        and row["model"] == "real"
        and (row["target"], row["threshold"]) in PRIMARY_FOCUS
    ]
    event_plus_pre_direct = [
        row for row in event_rows
        if row["split"] == "blocked"
        and row["mask"] == "event_plus_pre_3s"
        and row["model"] == "real"
        and (row["target"], row["threshold"]) in PRIMARY_FOCUS
    ]
    balanced_real = [
        row for row in balanced_rows
        if row["split"] == "blocked"
        and row["model"] == "real"
        and (row["target"], row["threshold"]) in PRIMARY_FOCUS
    ]
    balanced_passes = []
    for row in balanced_real:
        ar = row_lookup_balanced(balanced_rows, row, "ar")
        shuffled = row_lookup_balanced(balanced_rows, row, "shuffled")
        random = row_lookup_balanced(balanced_rows, row, "random")
        if ar and shuffled and random and all(
            row.get("pr_auc_mean") is not None
            and other.get("pr_auc_mean") is not None
            and row["pr_auc_mean"] > other["pr_auc_mean"]
            for other in (ar, shuffled, random)
        ):
            balanced_passes.append(row)
    if balanced_passes:
        verdict = "pre-event promising"
    else:
        verdict = "needs alignment correction"

    enough = [row for row in video_rows if row.get("enough_events")]
    wins = [
        row for row in enough
        if row.get("win_vs_ar") and row.get("win_vs_shuffled") and row.get("win_vs_random")
    ]
    lines = [
        "# VEATIC Event-Conditioned Retest",
        "",
        "## SECTION 1: Executive Verdict",
        "",
        f"Classification: **{verdict}**.",
        "",
        "The full-frame benchmark remains the real-world baseline. These rows test whether stable zero/non-event frames are suppressing cortical signal in event-relevant regions.",
        "",
        "Decision thresholds were selected on train predictions only and then applied unchanged to every filtered test subset. Filtered test subsets were not used for threshold tuning.",
        "",
        "## SECTION 2: Full-Frame vs Event-Conditioned Comparison",
        "",
        "| Feature | Target | Thr | Mask | Real PR-AUC | Real-vs-AR | Real-vs-shuf | Real-vs-rand |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(full_vs_event, key=lambda item: (item["target"], item["threshold"], item["feature_mode"], item["mask"])):
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | {row['mask']} | "
            f"{fmt(row.get('real'))} | {fmt(row.get('real_vs_ar'))} | {fmt(row.get('real_vs_shuffled'))} | {fmt(row.get('real_vs_random'))} |"
        )
    lines.extend(
        [
            "",
            "Event-plus-pre rows are positive-only masks, so PR-AUC is undefined there; balanced event-vs-stable sampling below is the clean event-conditioned discrimination test.",
            "",
            "| Feature | Target | Thr | Mask | Real recall | Real top-10% recall | Count |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(event_plus_pre_direct, key=lambda item: (item["target"], item["threshold"], item["feature_mode"])):
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | {row['mask']} | "
            f"{fmt(row.get('recall'))} | {fmt(row.get('top_10pct_recall'))} | {fmt(row.get('event_count'))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 3: Pre-Event Detection",
            "",
            "Pre-event rows treat the pre-event frame as the positive early-warning region while preserving train-only thresholds. PR-AUC is undefined for positive-only subsets, so recall/top-k and balanced sampling carry more weight.",
            "",
            "| Feature | Target | Thr | Mask | Real recall | Real top-10% recall | Real-vs-AR recall | Real-vs-shuf recall | Real-vs-rand recall |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(pre_event_direct, key=lambda item: (item["target"], item["threshold"], item["mask"], item["feature_mode"])):
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | {row['mask']} | "
            f"{fmt(row.get('recall'))} | {fmt(row.get('top_10pct_recall'))} | "
            f"{fmt(diff(row.get('recall'), (row_lookup(event_rows, feature_mode=row['feature_mode'], split=row['split'], target=row['target'], threshold=row['threshold'], mask=row['mask'], model='ar') or {}).get('recall')))} | "
            f"{fmt(diff(row.get('recall'), (row_lookup(event_rows, feature_mode=row['feature_mode'], split=row['split'], target=row['target'], threshold=row['threshold'], mask=row['mask'], model='shuffled') or {}).get('recall')))} | "
            f"{fmt(diff(row.get('recall'), (row_lookup(event_rows, feature_mode=row['feature_mode'], split=row['split'], target=row['target'], threshold=row['threshold'], mask=row['mask'], model='random') or {}).get('recall')))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 4: Event-Only Detection",
            "",
            "| Feature | Target | Thr | Real recall | Real top-10% recall | Event count |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(event_only, key=lambda item: (item["target"], item["threshold"], item["feature_mode"])):
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | "
            f"{fmt(row.get('recall'))} | {fmt(row.get('top_10pct_recall'))} | "
            f"{fmt(row.get('event_count'))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 5: Balanced Event-vs-Stable Sampling",
            "",
            "| Feature | Target | Thr | Ratio | Model | PR-AUC mean +/- std | F1 mean +/- std | BalAcc mean +/- std | Recall mean +/- std |",
            "|---|---|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        [r for r in balanced_rows if r["split"] == "blocked" and (r["target"], r["threshold"]) in PRIMARY_FOCUS],
        key=lambda item: (item["target"], item["threshold"], item["negative_ratio"], item["feature_mode"], item["model"]),
    ):
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | {row['negative_ratio']} | {row['model']} | "
            f"{fmt(row.get('pr_auc_mean'))} +/- {fmt(row.get('pr_auc_std'))} | "
            f"{fmt(row.get('f1_mean'))} +/- {fmt(row.get('f1_std'))} | "
            f"{fmt(row.get('balanced_accuracy_mean'))} +/- {fmt(row.get('balanced_accuracy_std'))} | "
            f"{fmt(row.get('recall_mean'))} +/- {fmt(row.get('recall_std'))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 6: Controls",
            "",
            "Anti-leakage controls were run for blocked primary-focus targets and masks: label shuffle within/across videos, feature shuffle within/across videos, timestamp-only, and video-ID/time-only.",
            "",
            "| Feature | Target | Thr | Mask | Control | PR-AUC | F1 | Recall |",
            "|---|---|---:|---|---|---:|---:|---:|",
        ]
    )
    control_rows = [
        row for row in event_rows
        if row["split"] == "blocked"
        and (row["target"], row["threshold"]) in PRIMARY_FOCUS
        and row["mask"] in ANTI_LEAKAGE_MASKS
        and row["model"] not in {"ar", "real", "shuffled", "random", "majority", "zero"}
    ]
    for row in sorted(control_rows, key=lambda item: (item["target"], item["threshold"], item["mask"], item["feature_mode"], item["model"]))[:160]:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {fmt(row['threshold'])} | {row['mask']} | `{row['model']}` | "
            f"{fmt(row.get('pr_auc'))} | {fmt(row.get('f1'))} | {fmt(row.get('recall'))} |"
        )
    if len(control_rows) > 160:
        lines.append(f"| ... | ... | ... | ... | ... | {len(control_rows) - 160} additional control rows in CSV | ... | ... |")
    lines.extend(
        [
            "",
            "## SECTION 7: Per-Video Robustness",
            "",
            f"Enough-event video rows: {len(enough)}. Wins versus AR, shuffled, and random by per-video PR-AUC: {len(wins)}/{len(enough)}.",
            "",
            "| Feature | Target | Thr | Enough videos | Triple-control wins |",
            "|---|---|---:|---:|---:|",
        ]
    )
    grouped_video: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in enough:
        grouped_video[(row["feature_mode"], row["target"], row["threshold"])].append(row)
    for (feature, target, threshold), values in sorted(grouped_video.items()):
        triple = [row for row in values if row.get("win_vs_ar") and row.get("win_vs_shuffled") and row.get("win_vs_random")]
        lines.append(f"| {feature} | `{target}` | {fmt(threshold)} | {len(values)} | {len(triple)} |")
    lines.extend(
        [
            "",
            "## SECTION 8: Final Recommendation",
            "",
            "For the full 124 benchmark: keep current full-frame metrics, add event-conditioned and balanced event-vs-stable metrics, prioritize blocked pre-event spike detection, and do not scale timing claims until the alignment/shift issue is resolved.",
            "",
            "Claim to carry forward only if replicated on 124: frame-wide continuous MAE underestimates cortical signal because most frames are stable; conditioned on upcoming arousal-change regions, cortical/TRIBE features improve early detection of emotionally meaningful events.",
            "",
            "## Output Files",
            "",
        ]
    )
    for label, value in output_paths.items():
        lines.append(f"- {label}: `{value}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def row_lookup_balanced(rows: list[dict[str, Any]], source: dict[str, Any], model: str) -> dict[str, Any] | None:
    for row in rows:
        if (
            row.get("feature_mode") == source.get("feature_mode")
            and row.get("split") == source.get("split")
            and row.get("target") == source.get("target")
            and row.get("threshold") == source.get("threshold")
            and row.get("negative_ratio") == source.get("negative_ratio")
            and row.get("model") == model
        ):
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_89_complete_20260615.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_89_complete_20260615.report.json")
    parser.add_argument("--cache-dir", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
    parser.add_argument("--output-prefix", default="benchmarks/veatic/veatic_89_event_conditioned_retest_20260616")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="auto", choices=("auto", "mps_solve", "cpu_pinv"))
    args = parser.parse_args()
    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend

    ctx = retest.RetestContext(
        Path(args.manifest).expanduser().resolve(),
        Path(args.report).expanduser().resolve(),
        Path(args.cache_dir).expanduser().resolve(),
    )
    prefix = Path(args.output_prefix).expanduser().resolve()
    rng = np.random.default_rng(args.seed)

    event_rows: list[dict[str, Any]] = []
    balanced_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []

    for feature_label, feature_mode in FEATURE_MODES:
        print(f"[INFO] feature={feature_label}", flush=True)
        base_feature_sets = ctx.base_feature_sets(feature_mode)
        for split_label, split_name in SPLITS:
            train_rows, test_rows, _ = retest.fixed_rows(ctx.accepted_rows, split_name)
            matrices = retest.split_matrices(ctx, base_feature_sets, train_rows, test_rows, feature_mode)

            for threshold in EVENT_THRESHOLDS:
                train_selected, train_y = retest.future_spike_rows(ctx, train_rows, threshold)
                test_selected, test_y = retest.future_spike_rows(ctx, test_rows, threshold)
                include_controls = split_label == "blocked" and ("arousal__future_spike_1_3s", threshold) in PRIMARY_FOCUS
                scores = model_score_sets(
                    ctx, train_selected, train_y, test_selected, test_y, matrices, rng, include_anti_leakage=include_controls
                )
                if scores is not None:
                    stable_mag = event_magnitude_for_rows(
                        ctx,
                        scores["test_rows"],
                        "arousal__future_spike_1_3s",
                        None,
                    )
                    masks = build_region_masks(
                        scores["test_rows"],
                        scores["true_test_y"].astype(np.int64),
                        stable_mag,
                        threshold,
                    )
                    base = {
                        "feature_mode": feature_label,
                        "split": split_label,
                        "target": "arousal__future_spike_1_3s",
                        "threshold": threshold,
                    }
                    evaluate_event_masks(event_rows, base, scores, masks)
                    evaluate_balanced_sampling(balanced_rows, base, scores, masks)
                    if split_label == "blocked" and threshold in {0.05, 0.075}:
                        per_video_rows(video_rows, base, scores, masks)

            for horizon in HORIZONS:
                train_selected_cont, train_y_cont = retest.future_change_rows(ctx, train_rows, horizon)
                test_selected_cont, test_y_cont = retest.future_change_rows(ctx, test_rows, horizon)
                cont_scores = model_score_sets(
                    ctx,
                    train_selected_cont,
                    train_y_cont,
                    test_selected_cont,
                    test_y_cont,
                    matrices,
                    rng,
                    include_anti_leakage=False,
                )
                for threshold in EVENT_THRESHOLDS:
                    train_event = (np.abs(train_y_cont) >= threshold).astype(np.float64)
                    test_event = (np.abs(test_y_cont) >= threshold).astype(np.float64)
                    target = f"arousal__future_change_p{horizon}s_movement"
                    include_controls = split_label == "blocked" and (target, threshold) in PRIMARY_FOCUS
                    class_scores = model_score_sets(
                        ctx,
                        train_selected_cont,
                        train_event,
                        test_selected_cont,
                        test_event,
                        matrices,
                        rng,
                        include_anti_leakage=include_controls,
                    )
                    if class_scores is None:
                        continue
                    stable_mag = event_magnitude_for_rows(
                        ctx,
                        class_scores["test_rows"],
                        target,
                        horizon,
                    )
                    masks = build_region_masks(
                        class_scores["test_rows"],
                        class_scores["true_test_y"].astype(np.int64),
                        stable_mag,
                        threshold,
                    )
                    base = {
                        "feature_mode": feature_label,
                        "split": split_label,
                        "target": target,
                        "threshold": threshold,
                    }
                    evaluate_event_masks(event_rows, base, class_scores, masks)
                    evaluate_balanced_sampling(balanced_rows, base, class_scores, masks)
                    if cont_scores is not None:
                        cont_base = {
                            "feature_mode": feature_label,
                            "split": split_label,
                            "target": f"arousal__future_change_p{horizon}s",
                            "event_threshold": threshold,
                            "threshold": threshold,
                        }
                        evaluate_continuous_masks(event_rows, cont_base, cont_scores, test_y_cont, masks)
                    if split_label == "blocked" and (target, threshold) in PRIMARY_FOCUS:
                        per_video_rows(video_rows, base, class_scores, masks)

    json_path = prefix.with_suffix(".json")
    masks_path = prefix.with_suffix(".event_masks.csv")
    balanced_path = prefix.with_suffix(".balanced_sampling.csv")
    per_video_path = prefix.with_suffix(".per_video.csv")
    md_path = prefix.with_suffix(".md")
    payload = {
        "schema_version": "veatic_event_conditioned_retest_v2",
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "methodology": {
            "full_frame_preserved": True,
            "decision_thresholds": "selected on train predictions only; no filtered-test threshold tuning",
            "event_masks": [
                "all_frames",
                "stable_negative_only",
                "event_only",
                "pre_event_1s",
                "pre_event_2s",
                "pre_event_3s",
                "pre_event_5s",
                "event_plus_pre_3s",
            ],
            "balanced_event_vs_stable": {"ratios": list(BALANCED_RATIOS), "seeds": len(BALANCED_SEEDS)},
            "anti_leakage_controls": [
                "label_shuffle_within_video",
                "label_shuffle_across_videos",
                "feature_shuffle_within_video",
                "feature_shuffle_across_videos",
                "timestamp_only",
                "video_id_time_only",
            ],
        },
        "event_masks": event_rows,
        "balanced_sampling": balanced_rows,
        "per_video": video_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(masks_path, event_rows)
    write_csv(balanced_path, balanced_rows)
    write_csv(per_video_path, video_rows)
    write_markdown(
        md_path,
        event_rows,
        balanced_rows,
        video_rows,
        {
            "JSON": json_path,
            "Event masks CSV": masks_path,
            "Balanced sampling CSV": balanced_path,
            "Per-video CSV": per_video_path,
        },
    )
    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "json": str(json_path),
                "event_masks_csv": str(masks_path),
                "balanced_sampling_csv": str(balanced_path),
                "per_video_csv": str(per_video_path),
                "event_mask_rows": len(event_rows),
                "balanced_rows": len(balanced_rows),
                "per_video_rows": len(video_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
