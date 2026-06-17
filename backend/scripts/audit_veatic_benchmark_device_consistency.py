"""Audit VEATIC 89 benchmark-side CPU vs MPS consistency.

This script does not re-encode videos. It reloads the frozen 89-video manifest
and cached TRIBE/cortical feature arrays, then reruns the suspicious
event-conditioned rows under CPU and MPS benchmark backends.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(ROOT / "external_assets"))).expanduser()
COND_SCRIPT = ROOT / "backend" / "scripts" / "run_veatic_event_conditioned_retest.py"
spec = importlib.util.spec_from_file_location("event_conditioned", COND_SCRIPT)
cond = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cond)
retest = cond.retest
bench = cond.bench

FEATURE_MODES = cond.FEATURE_MODES
SPLITS = cond.SPLITS
FOCUS = (
    ("arousal__future_spike_1_3s", 0.05, None),
    ("arousal__future_spike_1_3s", 0.075, None),
    ("arousal__future_change_p2s_movement", 0.05, 2),
    ("arousal__future_change_p3s_movement", 0.05, 3),
    ("arousal__future_change_p3s_movement", 0.075, 3),
)
MASKS = (
    "all_frames",
    "event_only",
    "pre_event_1s",
    "pre_event_2s",
    "pre_event_3s",
    "event_plus_pre_3s",
)
BALANCED_RATIOS = (1, 2)
MODELS = ("ar", "real", "shuffled", "random")
CONTROL_MODELS = (
    "label_shuffle_within_video",
    "label_shuffle_across_videos",
    "feature_shuffle_within_video",
    "feature_shuffle_across_videos",
    "timestamp_only",
    "video_id_time_only",
)


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def finite(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def safe_corr(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> float | None:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    if spearman:
        left = bench.rankdata(left)
        right = bench.rankdata(right)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def score_diff_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "n": int(min(left.size, right.size)),
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "rank_correlation": None,
            "pct_gt_1e_6": None,
            "pct_gt_1e_5": None,
            "pct_gt_1e_4": None,
            "pct_gt_1e_3": None,
        }
    diff = np.abs(left - right)
    return {
        "shape_match": True,
        "n": int(left.size),
        "mean_abs_diff": float(np.mean(diff)) if diff.size else None,
        "max_abs_diff": float(np.max(diff)) if diff.size else None,
        "rmse": float(np.sqrt(np.mean(np.square(left - right)))) if diff.size else None,
        "pearson": safe_corr(left, right),
        "spearman": safe_corr(left, right, spearman=True),
        "rank_correlation": safe_corr(left, right, spearman=True),
        "pct_gt_1e_6": float(np.mean(diff > 1e-6)) if diff.size else None,
        "pct_gt_1e_5": float(np.mean(diff > 1e-5)) if diff.size else None,
        "pct_gt_1e_4": float(np.mean(diff > 1e-4)) if diff.size else None,
        "pct_gt_1e_3": float(np.mean(diff > 1e-3)) if diff.size else None,
    }


def metric_class(pr_auc_delta: Any, f1_delta: Any, recall_delta: Any) -> str:
    pr_auc_delta = abs(float(pr_auc_delta or 0.0))
    f1_delta = abs(float(f1_delta or 0.0))
    recall_delta = abs(float(recall_delta or 0.0))
    if pr_auc_delta > 0.02 or f1_delta > 0.03 or recall_delta > 0.05:
        return "material"
    if pr_auc_delta <= 0.005 and f1_delta <= 0.01 and recall_delta <= 0.02:
        return "stable"
    return "minor"


def configure_backend(kind: str) -> dict[str, Any]:
    if kind == "cpu":
        bench.PCA_BACKEND = "cpu_svd"
        bench.RIDGE_BACKEND = "cpu_pinv"
        bench.RIDGE_MPS_MIN_FEATURES = 2048
        return {
            "pca_backend": "cpu_svd",
            "ridge_backend": "cpu_pinv",
            "ridge_mps_min_features": 2048,
        }
    if kind == "mps":
        bench.PCA_BACKEND = "mps_gram"
        bench.RIDGE_BACKEND = "mps_solve"
        bench.RIDGE_MPS_MIN_FEATURES = 0
        return {
            "pca_backend": "mps_gram",
            "ridge_backend": "mps_solve",
            "ridge_mps_min_features": 0,
        }
    raise ValueError(kind)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass


def target_rows(
    ctx: retest.RetestContext,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    target: str,
    threshold: float,
    horizon: int | None,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], np.ndarray]:
    if target == "arousal__future_spike_1_3s":
        train_selected, train_y = retest.future_spike_rows(ctx, train_rows, threshold)
        test_selected, test_y = retest.future_spike_rows(ctx, test_rows, threshold)
    else:
        assert horizon is not None
        train_selected, train_cont = retest.future_change_rows(ctx, train_rows, horizon)
        test_selected, test_cont = retest.future_change_rows(ctx, test_rows, horizon)
        train_y = (np.abs(train_cont) >= threshold).astype(np.float64)
        test_y = (np.abs(test_cont) >= threshold).astype(np.float64)
    return train_selected, train_y, test_selected, test_y


def run_device_pass(
    ctx: retest.RetestContext,
    *,
    kind: str,
    repeat: int,
    base_seed: int,
) -> dict[str, Any]:
    backend = configure_backend(kind)
    set_all_seeds(base_seed + repeat)
    run: dict[str, Any] = {
        "kind": kind,
        "repeat": repeat,
        "backend": backend,
        "scores": {},
        "metrics": [],
        "thresholds": [],
        "per_video": [],
        "pca_metadata": [],
    }
    for feature_label, feature_mode in FEATURE_MODES:
        print(f"[INFO] {kind}{repeat} feature={feature_label}", flush=True)
        base_feature_sets = ctx.base_feature_sets(feature_mode)
        for split_label, split_name in SPLITS:
            train_rows, test_rows, _ = retest.fixed_rows(ctx.accepted_rows, split_name)
            matrices = retest.split_matrices(ctx, base_feature_sets, train_rows, test_rows, feature_mode)
            run["pca_metadata"].append(
                {
                    "kind": kind,
                    "repeat": repeat,
                    "feature_mode": feature_label,
                    "split": split_label,
                    "feature_key": matrices["feature_key"],
                    "metadata": matrices["metadata"],
                }
            )
            for target, threshold, horizon in FOCUS:
                train_selected, train_y, test_selected, test_y = target_rows(
                    ctx, train_rows, test_rows, target, threshold, horizon
                )
                include_controls = split_label == "blocked"
                rng = np.random.default_rng(stable_seed(base_seed, feature_label, split_label, target, threshold))
                score_sets = cond.model_score_sets(
                    ctx,
                    train_selected,
                    train_y,
                    test_selected,
                    test_y,
                    matrices,
                    rng,
                    include_anti_leakage=include_controls,
                )
                if score_sets is None:
                    continue
                magnitude = cond.event_magnitude_for_rows(ctx, score_sets["test_rows"], target, horizon)
                masks = cond.build_region_masks(
                    score_sets["test_rows"],
                    score_sets["true_test_y"].astype(np.int64),
                    magnitude,
                    threshold,
                )
                config = {
                    "feature_mode": feature_label,
                    "split": split_label,
                    "target": target,
                    "threshold": threshold,
                }
                for model, (model_train_y, train_scores, test_scores) in score_sets["models"].items():
                    if model not in MODELS and (split_label != "blocked" or model not in CONTROL_MODELS):
                        continue
                    score_key = score_id(config, model)
                    run["scores"][score_key] = {
                        **config,
                        "model": model,
                        "train_y": model_train_y.astype(np.float64),
                        "train_scores": train_scores.astype(np.float64),
                        "test_y": score_sets["true_test_y"].astype(np.int64),
                        "test_scores": test_scores.astype(np.float64),
                        "test_rows": score_sets["test_rows"],
                        "masks": masks,
                    }
                    train_threshold = retest.best_train_threshold(
                        model_train_y.astype(np.int64),
                        train_scores.astype(np.float64),
                    )
                    for mask_name in MASKS:
                        idx = np.flatnonzero(masks[mask_name])
                        if idx.size == 0:
                            continue
                        eval_y = cond.labels_for_mask(
                            mask_name,
                            masks,
                            score_sets["true_test_y"].astype(np.int64),
                        )
                        metrics = cond.event_subset_metrics(
                            model_train_y,
                            train_scores,
                            eval_y,
                            test_scores[idx],
                        )
                        run["metrics"].append(
                            {
                                **config,
                                "kind": kind,
                                "repeat": repeat,
                                "mask": mask_name,
                                "balanced_ratio": None,
                                "model": model,
                                **{key: finite(value) for key, value in metrics.items()},
                            }
                        )
                        run["thresholds"].append(
                            {
                                **config,
                                "kind": kind,
                                "repeat": repeat,
                                "mask": mask_name,
                                "balanced_ratio": None,
                                "model": model,
                                "train_threshold": float(train_threshold),
                                "eval_n": int(idx.size),
                            }
                        )
                    if model in MODELS:
                        for ratio in BALANCED_RATIOS:
                            for seed in cond.BALANCED_SEEDS:
                                eval_idx, eval_y = balanced_indices(masks, ratio, seed)
                                if eval_idx.size == 0:
                                    continue
                                metrics = cond.event_subset_metrics(
                                    model_train_y,
                                    train_scores,
                                    eval_y,
                                    test_scores[eval_idx],
                                )
                                run["metrics"].append(
                                    {
                                        **config,
                                        "kind": kind,
                                        "repeat": repeat,
                                        "mask": "balanced_event_vs_stable",
                                        "balanced_ratio": f"1:{ratio}",
                                        "balanced_seed": seed,
                                        "model": model,
                                        **{key: finite(value) for key, value in metrics.items()},
                                    }
                                )
                                run["thresholds"].append(
                                    {
                                        **config,
                                        "kind": kind,
                                        "repeat": repeat,
                                        "mask": "balanced_event_vs_stable",
                                        "balanced_ratio": f"1:{ratio}",
                                        "balanced_seed": seed,
                                        "model": model,
                                        "train_threshold": float(train_threshold),
                                        "eval_n": int(eval_idx.size),
                                    }
                                )
                add_per_video_rows(run["per_video"], config, score_sets, masks, kind, repeat)
    return run


def balanced_indices(masks: dict[str, np.ndarray], ratio: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(masks["event_plus_pre_3s"])
    stable = np.flatnonzero(masks["stable_negative_only"])
    if positive.size == 0 or stable.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    neg_n = min(stable.size, positive.size * ratio)
    negatives = rng.choice(stable, size=neg_n, replace=False)
    return (
        np.concatenate([positive, negatives]),
        np.concatenate([np.ones(positive.size, dtype=np.int64), np.zeros(negatives.size, dtype=np.int64)]),
    )


def add_per_video_rows(
    output: list[dict[str, Any]],
    config: dict[str, Any],
    score_sets: dict[str, Any],
    masks: dict[str, np.ndarray],
    kind: str,
    repeat: int,
) -> None:
    target_mask = masks["event_plus_pre_3s"]
    stable_mask = masks["stable_negative_only"]
    eval_mask = target_mask | stable_mask
    idx_all = np.flatnonzero(eval_mask)
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
        row = {
            **config,
            "kind": kind,
            "repeat": repeat,
            "video_id": video_id,
            "mask": "event_plus_pre_3s_vs_stable",
            "n": int(y.size),
            "event_count": float(np.sum(y)),
            "positive_rate": float(np.mean(y)),
            "enough_events": bool(np.sum(y) >= 3 and np.sum(y == 0) >= 3),
        }
        for model in MODELS:
            _, _, scores = score_sets["models"][model]
            row[f"{model}_pr_auc"] = finite(cond.strict_pr_auc(y, scores[idx]))
            pred = (scores[idx] >= retest.best_train_threshold(score_sets["models"][model][0].astype(np.int64), score_sets["models"][model][1])).astype(np.int64)
            binary = retest.binary_metrics_from_pred(y, pred)
            row[f"{model}_recall"] = finite(binary.get("recall"))
        row["real_vs_ar_pr_auc_delta"] = delta(row.get("real_pr_auc"), row.get("ar_pr_auc"))
        row["real_vs_shuffled_pr_auc_delta"] = delta(row.get("real_pr_auc"), row.get("shuffled_pr_auc"))
        row["real_vs_random_pr_auc_delta"] = delta(row.get("real_pr_auc"), row.get("random_pr_auc"))
        row["win_vs_ar"] = row["real_vs_ar_pr_auc_delta"] is not None and row["real_vs_ar_pr_auc_delta"] > 0
        row["win_vs_shuffled"] = row["real_vs_shuffled_pr_auc_delta"] is not None and row["real_vs_shuffled_pr_auc_delta"] > 0
        row["win_vs_random"] = row["real_vs_random_pr_auc_delta"] is not None and row["real_vs_random_pr_auc_delta"] > 0
        output.append(row)


def delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def score_id(config: dict[str, Any], model: str) -> str:
    return "|".join(
        [
            config["feature_mode"],
            config["split"],
            config["target"],
            str(config["threshold"]),
            model,
        ]
    )


def row_key(row: dict[str, Any], include_seed: bool = True) -> tuple[Any, ...]:
    values = (
        row.get("feature_mode"),
        row.get("split"),
        row.get("target"),
        row.get("threshold"),
        row.get("mask"),
        row.get("balanced_ratio"),
        row.get("model"),
    )
    if include_seed:
        values += (row.get("balanced_seed"),)
    return values


def compare_predictions(left: dict[str, Any], right: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = []
    for key, left_score in left["scores"].items():
        right_score = right["scores"].get(key)
        if right_score is None:
            continue
        metrics = score_diff_metrics(left_score["test_scores"], right_score["test_scores"])
        rows.append(
            {
                "comparison": label,
                "feature_mode": left_score["feature_mode"],
                "split": left_score["split"],
                "target": left_score["target"],
                "threshold": left_score["threshold"],
                "model": left_score["model"],
                "left_dtype": str(left_score["test_scores"].dtype),
                "right_dtype": str(right_score["test_scores"].dtype),
                "left_shape": "x".join(str(item) for item in left_score["test_scores"].shape),
                "right_shape": "x".join(str(item) for item in right_score["test_scores"].shape),
                **metrics,
            }
        )
    return rows


def compare_thresholds(left: dict[str, Any], right: dict[str, Any], label: str) -> list[dict[str, Any]]:
    right_by_key = {row_key(row): row for row in right["thresholds"]}
    right_scores = right["scores"]
    rows = []
    for left_row in left["thresholds"]:
        right_row = right_by_key.get(row_key(left_row))
        if right_row is None:
            continue
        sid = score_id(left_row, left_row["model"])
        left_score = left["scores"].get(sid)
        right_score = right_scores.get(sid)
        flip_count = None
        flip_rate = None
        if left_score is not None and right_score is not None:
            indices = eval_indices_from_row(left_score, left_row)
            if indices.size:
                left_pred = left_score["test_scores"][indices] >= float(left_row["train_threshold"])
                right_pred = right_score["test_scores"][indices] >= float(right_row["train_threshold"])
                flip_count = int(np.sum(left_pred != right_pred))
                flip_rate = float(np.mean(left_pred != right_pred))
        rows.append(
            {
                "comparison": label,
                **{key: left_row.get(key) for key in ("feature_mode", "split", "target", "threshold", "mask", "balanced_ratio", "balanced_seed", "model")},
                "left_train_threshold": left_row.get("train_threshold"),
                "right_train_threshold": right_row.get("train_threshold"),
                "threshold_delta": delta(right_row.get("train_threshold"), left_row.get("train_threshold")),
                "eval_n": left_row.get("eval_n"),
                "threshold_flip_count": flip_count,
                "threshold_flip_rate": flip_rate,
            }
        )
    return rows


def eval_indices_from_row(score: dict[str, Any], row: dict[str, Any]) -> np.ndarray:
    masks = score.get("masks") or rebuild_masks_from_score(score)
    if row.get("mask") == "balanced_event_vs_stable":
        ratio = int(str(row.get("balanced_ratio")).split(":")[1])
        seed = int(row.get("balanced_seed"))
        return balanced_indices(masks, ratio, seed)[0]
    return np.flatnonzero(masks[row["mask"]])


def rebuild_masks_from_score(score: dict[str, Any]) -> dict[str, np.ndarray]:
    # Magnitude is only needed for stable_negative_only; for already computed
    # score entries we cannot reconstruct it from the compact score object.
    # Store a conservative stable mask as all non-events when threshold flips
    # are requested for balanced rows. Exact metric rows are computed earlier.
    y_event = score["test_y"].astype(np.int64)
    event_mask = y_event.astype(bool)
    masks = {
        "all_frames": np.ones(y_event.size, dtype=bool),
        "stable_negative_only": ~event_mask,
        "event_only": event_mask,
    }
    # Pre-event reconstruction from rows/timestamps.
    key_to_index = {
        (str(row["video_id"]), int(round(float(row["time_start_seconds"])))): index
        for index, row in enumerate(score["test_rows"])
    }
    for lead in (1, 2, 3):
        mask = np.zeros(y_event.size, dtype=bool)
        for index, is_event in enumerate(event_mask):
            if not is_event:
                continue
            row = score["test_rows"][index]
            key = (str(row["video_id"]), int(round(float(row["time_start_seconds"]))) - lead)
            pre_index = key_to_index.get(key)
            if pre_index is not None and not event_mask[pre_index]:
                mask[pre_index] = True
        masks[f"pre_event_{lead}s"] = mask
    masks["event_plus_pre_3s"] = masks["event_only"] | masks["pre_event_1s"] | masks["pre_event_2s"] | masks["pre_event_3s"]
    return masks


def compare_metrics(left: dict[str, Any], right: dict[str, Any], label: str, old_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    right_by_key = {row_key(row): row for row in right["metrics"]}
    old_by_key = {old_row_key(row): row for row in old_rows}
    rows = []
    for left_row in left["metrics"]:
        right_row = right_by_key.get(row_key(left_row))
        if right_row is None:
            continue
        old_row = old_by_key.get(old_row_key(left_row))
        record = {
            "comparison": label,
            **{key: left_row.get(key) for key in ("feature_mode", "split", "target", "threshold", "mask", "balanced_ratio", "balanced_seed", "model")},
        }
        for metric in (
            "pr_auc",
            "f1",
            "balanced_accuracy",
            "precision",
            "recall",
            "accuracy",
            "top_1pct_recall",
            "top_5pct_recall",
            "top_10pct_recall",
            "top_event_count_recall",
        ):
            record[f"left_{metric}"] = left_row.get(metric)
            record[f"right_{metric}"] = right_row.get(metric)
            record[f"{metric}_delta"] = delta(right_row.get(metric), left_row.get(metric))
            record[f"old_{metric}"] = old_row.get(metric) if old_row else None
            record[f"old_vs_left_{metric}_delta"] = delta(old_row.get(metric), left_row.get(metric)) if old_row else None
            record[f"old_vs_right_{metric}_delta"] = delta(old_row.get(metric), right_row.get(metric)) if old_row else None
        record["drift_class"] = metric_class(record.get("pr_auc_delta"), record.get("f1_delta"), record.get("recall_delta"))
        rows.append(record)
    return rows


def old_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("feature_mode"),
        row.get("split"),
        row.get("target"),
        row.get("threshold"),
        row.get("mask"),
        row.get("balanced_ratio") or row.get("negative_ratio"),
        row.get("model"),
        row.get("balanced_seed"),
    )


def load_old_rows(prefix: Path) -> list[dict[str, Any]]:
    if not prefix.with_suffix(".json").exists():
        return []
    data = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    rows = []
    rows.extend(data.get("event_masks", []))
    for row in data.get("balanced_sampling", []):
        expanded = dict(row)
        expanded["mask"] = "balanced_event_vs_stable"
        expanded["balanced_ratio"] = row.get("negative_ratio")
        for metric in ("pr_auc", "f1", "balanced_accuracy", "precision", "recall", "accuracy"):
            expanded[metric] = row.get(f"{metric}_mean")
        rows.append(expanded)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
        return f"{float(value):.6f}"
    return str(value)


def summarize_per_video(cpu: dict[str, Any], mps: dict[str, Any]) -> dict[str, Any]:
    mps_by_key = {
        (
            row["feature_mode"],
            row["split"],
            row["target"],
            row["threshold"],
            row["video_id"],
        ): row
        for row in mps["per_video"]
        if row.get("enough_events")
    }
    total = 0
    flips = 0
    for row in cpu["per_video"]:
        if not row.get("enough_events"):
            continue
        other = mps_by_key.get((row["feature_mode"], row["split"], row["target"], row["threshold"], row["video_id"]))
        if other is None:
            continue
        total += 1
        cpu_win = row.get("win_vs_ar") and row.get("win_vs_shuffled") and row.get("win_vs_random")
        mps_win = other.get("win_vs_ar") and other.get("win_vs_shuffled") and other.get("win_vs_random")
        if bool(cpu_win) != bool(mps_win):
            flips += 1
    return {
        "enough_event_video_rows_compared": total,
        "win_loss_flips": flips,
        "win_loss_flip_rate": float(flips / total) if total else None,
    }


def write_markdown(
    path: Path,
    payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> None:
    cpu_mps_pred = [row for row in prediction_rows if row["comparison"] == "cpu1_vs_mps1"]
    max_diff = max((row.get("max_abs_diff") or 0.0 for row in cpu_mps_pred), default=0.0)
    mean_diff = max((row.get("mean_abs_diff") or 0.0 for row in cpu_mps_pred), default=0.0)
    material = [row for row in metric_rows if row["comparison"] == "cpu1_vs_mps1" and row["drift_class"] == "material"]
    threshold_max = max((abs(row.get("threshold_delta") or 0.0) for row in threshold_rows if row["comparison"] == "cpu1_vs_mps1"), default=0.0)
    flip_rate = max((row.get("threshold_flip_rate") or 0.0 for row in threshold_rows if row["comparison"] == "cpu1_vs_mps1"), default=0.0)
    per_video = payload["per_video_summary"]
    if material or (per_video.get("win_loss_flip_rate") or 0.0) > 0.2:
        verdict = "unsafe"
    elif max_diff > 1e-3 or threshold_max > 1e-3 or flip_rate > 0.02:
        verdict = "caution"
    else:
        verdict = "safe"

    lines = [
        "# VEATIC 89 Benchmark Device Consistency Audit",
        "",
        "## SECTION 1: Executive Verdict",
        "",
        f"Classification: **{verdict}**.",
        "",
        f"Max CPU-vs-MPS test-score absolute difference: {fmt(max_diff)}; worst mean absolute difference: {fmt(mean_diff)}.",
        f"Max train-threshold delta: {fmt(threshold_max)}; worst threshold flip rate: {fmt(flip_rate)}.",
        f"Material metric-drift rows: {len(material)}.",
        f"Per-video enough-event win/loss flip rate: {fmt(per_video.get('win_loss_flip_rate'))} ({per_video.get('win_loss_flips')}/{per_video.get('enough_event_video_rows_compared')}).",
        "",
        "## SECTION 2: Benchmark Implementation Path",
        "",
        "- Cached TRIBE/cortical arrays are loaded from `.npz` as NumPy `float32`; no video re-encoding occurs in this audit.",
        "- Feature summaries, masks, thresholds, metrics, shuffled/random controls, and balanced sampling are NumPy-based.",
        "- PCA modes can use PyTorch MPS via `mps_gram` or CPU SVD via `cpu_svd`; MPS Gram still performs the small eigensolve on CPU.",
        "- Ridge fitting can use NumPy pseudo-inverse via `cpu_pinv` or PyTorch MPS normal-equation solve via `mps_solve`.",
        "- Threshold calibration uses NumPy arrays and train predictions only.",
        "- No autocast, float16, or mixed precision path was found; MPS tensors are explicitly `float32`, CPU ridge uses NumPy float64 after standardization.",
        "- Seeds: benchmark scripts use local `np.random.default_rng`; this audit also fixes Python `random`, NumPy global seed, Torch CPU seed, and Torch MPS seed before each pass.",
        "",
        "## SECTION 3: Prediction Score Comparison",
        "",
        "| Comparison | Rows | Max abs diff | Worst mean abs diff | Worst pct >1e-4 | Worst Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in ("cpu1_vs_mps1", "cpu1_vs_cpu2", "mps1_vs_mps2"):
        rows = [row for row in prediction_rows if row["comparison"] == comparison]
        lines.append(
            f"| {comparison} | {len(rows)} | {fmt(max((r.get('max_abs_diff') or 0 for r in rows), default=0))} | "
            f"{fmt(max((r.get('mean_abs_diff') or 0 for r in rows), default=0))} | "
            f"{fmt(max((r.get('pct_gt_1e_4') or 0 for r in rows), default=0))} | "
            f"{fmt(min((r.get('spearman') for r in rows if r.get('spearman') is not None), default=None))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 4: Threshold Comparison",
            "",
            "| Comparison | Rows | Max threshold delta | Worst flip rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison in ("cpu1_vs_mps1", "cpu1_vs_cpu2", "mps1_vs_mps2"):
        rows = [row for row in threshold_rows if row["comparison"] == comparison]
        lines.append(
            f"| {comparison} | {len(rows)} | {fmt(max((abs(r.get('threshold_delta') or 0) for r in rows), default=0))} | "
            f"{fmt(max((r.get('threshold_flip_rate') or 0 for r in rows), default=0))} |"
        )
    lines.extend(
        [
            "",
            "## SECTION 5: Metric Comparison",
            "",
            "| Comparison | Rows | Stable | Minor | Material |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in ("cpu1_vs_mps1", "cpu1_vs_cpu2", "mps1_vs_mps2"):
        rows = [row for row in metric_rows if row["comparison"] == comparison]
        counts = defaultdict(int)
        for row in rows:
            counts[row["drift_class"]] += 1
        lines.append(f"| {comparison} | {len(rows)} | {counts['stable']} | {counts['minor']} | {counts['material']} |")
    lines.extend(
        [
            "",
            "## SECTION 6: Determinism",
            "",
            "CPU and MPS repeatability are summarized by the repeat comparisons above. Any non-zero repeat drift indicates backend numeric nondeterminism or solver sensitivity under identical cached inputs.",
            "",
            "## SECTION 7: Recommendation",
            "",
            f"Recommendation: **{verdict}**.",
            "",
            "Use the CSVs for row-level decisions before making the 124-video run device-consistent. If this report is `caution` or worse, rerun final 89/124 on one backend policy and freeze it.",
            "",
            "## Output Files",
            "",
        ]
    )
    for label, output_path in output_paths.items():
        lines.append(f"- {label}: `{output_path}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_89_complete_20260615.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_89_complete_20260615.report.json")
    parser.add_argument("--cache-dir", default=str(EXTERNAL_ROOT / "benchmarks" / "veatic" / "tribe_cache"))
    parser.add_argument("--old-prefix", default="benchmarks/veatic/veatic_89_event_conditioned_retest_20260616")
    parser.add_argument("--output-prefix", default="benchmarks/veatic/veatic_89_benchmark_device_consistency_20260616")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    ctx = retest.RetestContext(
        Path(args.manifest).expanduser().resolve(),
        Path(args.report).expanduser().resolve(),
        Path(args.cache_dir).expanduser().resolve(),
    )
    old_rows = load_old_rows(Path(args.old_prefix).expanduser().resolve())
    cpu1 = run_device_pass(ctx, kind="cpu", repeat=1, base_seed=args.seed)
    cpu2 = run_device_pass(ctx, kind="cpu", repeat=2, base_seed=args.seed)
    mps1 = run_device_pass(ctx, kind="mps", repeat=1, base_seed=args.seed)
    mps2 = run_device_pass(ctx, kind="mps", repeat=2, base_seed=args.seed)

    prediction_rows = []
    prediction_rows.extend(compare_predictions(cpu1, mps1, "cpu1_vs_mps1"))
    prediction_rows.extend(compare_predictions(cpu1, cpu2, "cpu1_vs_cpu2"))
    prediction_rows.extend(compare_predictions(mps1, mps2, "mps1_vs_mps2"))

    threshold_rows = []
    threshold_rows.extend(compare_thresholds(cpu1, mps1, "cpu1_vs_mps1"))
    threshold_rows.extend(compare_thresholds(cpu1, cpu2, "cpu1_vs_cpu2"))
    threshold_rows.extend(compare_thresholds(mps1, mps2, "mps1_vs_mps2"))

    metric_rows = []
    metric_rows.extend(compare_metrics(cpu1, mps1, "cpu1_vs_mps1", old_rows))
    metric_rows.extend(compare_metrics(cpu1, cpu2, "cpu1_vs_cpu2", old_rows))
    metric_rows.extend(compare_metrics(mps1, mps2, "mps1_vs_mps2", old_rows))

    prefix = Path(args.output_prefix).expanduser().resolve()
    json_path = prefix.with_suffix(".json")
    prediction_path = prefix.with_suffix(".prediction_diff.csv")
    metric_path = prefix.with_suffix(".metric_diff.csv")
    threshold_path = prefix.with_suffix(".threshold_diff.csv")
    md_path = prefix.with_suffix(".md")

    payload = {
        "schema_version": "veatic_89_benchmark_device_consistency_v1",
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "old_prefix": str(Path(args.old_prefix).expanduser().resolve()),
        "implementation_audit": {
            "video_reencoding": False,
            "cached_features_reused": True,
            "threshold_calibration": "NumPy train predictions only",
            "mixed_precision_or_autocast": False,
            "cpu_backend": cpu1["backend"],
            "mps_backend": mps1["backend"],
        },
        "pca_metadata": cpu1["pca_metadata"] + mps1["pca_metadata"],
        "per_video_summary": summarize_per_video(cpu1, mps1),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(prediction_path, prediction_rows)
    write_csv(metric_path, metric_rows)
    write_csv(threshold_path, threshold_rows)
    write_markdown(
        md_path,
        payload,
        prediction_rows,
        threshold_rows,
        metric_rows,
        {
            "JSON": json_path,
            "Prediction diff CSV": prediction_path,
            "Metric diff CSV": metric_path,
            "Threshold diff CSV": threshold_path,
        },
    )
    print(
        json.dumps(
            {
                "markdown": str(md_path),
                "json": str(json_path),
                "prediction_diff_csv": str(prediction_path),
                "metric_diff_csv": str(metric_path),
                "threshold_diff_csv": str(threshold_path),
                "prediction_rows": len(prediction_rows),
                "metric_rows": len(metric_rows),
                "threshold_rows": len(threshold_rows),
                "per_video_summary": payload["per_video_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
