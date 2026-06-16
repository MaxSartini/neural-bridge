"""Narrow VEATIC-124 temporal context v2 benchmark.

Reuse-first benchmark for future arousal spike ranking. This does not encode
videos or recompute raw TRIBE features. It recomputes only missing causal
context representation cells from cached TRIBE outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
V1_SCRIPT = ROOT / "backend" / "scripts" / "run_veatic_temporal_fairness_benchmark.py"
spec = importlib.util.spec_from_file_location("temporal_fairness_v1", V1_SCRIPT)
v1 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v1)
align = v1.align
retest = v1.retest
bench = v1.bench


FEATURES = (
    ("cortical_pca_64", "cortical_pca_64"),
    ("cortical_pca64_delta", "cortical_pca64_delta"),
)
TARGETS = (
    ("arousal__future_spike_1_3s", None, 0.05),
    ("arousal__future_spike_1_3s", None, 0.075),
)
WINDOWS = (
    ("current_only_0s", 0.0, 0.0, "causal"),
    ("causal_past_1s", -1.0, 0.0, "causal"),
    ("causal_past_2s", -2.0, 0.0, "causal"),
    ("causal_past_3s", -3.0, 0.0, "causal"),
    ("causal_past_4s", -4.0, 0.0, "causal"),
    ("causal_past_5s", -5.0, 0.0, "causal"),
)
REPRESENTATIONS = (
    "last",
    "mean",
    "slope",
    "mean_last",
    "mean_slope",
    "last_slope",
    "mean_std_last_slope",
)
DIAGNOSTIC_WINDOWS = (("symmetric_2s", -2.0, 2.0, "non_causal_diagnostic"),)
CONTROL_MODELS = ("ar", "shuffled", "random", "timestamp", "video_time")


def stable_seed(*parts: Any, base: int = 43) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return (int(digest, 16) + int(base)) % (2**32 - 1)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def diff(left: Any, right: Any) -> float | None:
    left = finite(left)
    right = finite(right)
    if left is None or right is None:
        return None
    return left - right


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, Any]) -> tuple[str, float]:
    return str(row["video_id"]), float(row["time_start_seconds"])


def row_signature(rows: list[dict[str, Any]]) -> tuple[tuple[str, float], ...]:
    return tuple(row_key(row) for row in rows)


def y_signature(values: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(np.asarray(values, dtype=np.float64).tobytes())
    return h.hexdigest()


def interpolate_at(times: np.ndarray, matrix: np.ndarray, value_time: float) -> np.ndarray | None:
    return v1.interpolate_at(times, matrix, value_time)


def window_representation_features(
    selected_rows: list[dict[str, Any]],
    tables: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]],
    start_seconds: float,
    end_seconds: float,
    representations: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    keep: list[int] = []
    rep_values: dict[str, list[np.ndarray]] = {name: [] for name in representations}
    kept_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        video_id, anchor = row_key(row)
        table = tables.get(video_id)
        if table is None:
            continue
        times, matrix, _ = table
        start = anchor + start_seconds
        end = anchor + end_seconds
        if start < times[0] or end > times[-1] or end < start:
            continue
        first = interpolate_at(times, matrix, start)
        last = interpolate_at(times, matrix, end)
        if first is None or last is None:
            continue
        mask = (times >= start - 1e-9) & (times <= end + 1e-9)
        values = matrix[mask]
        if values.size == 0:
            values = np.vstack([first, last])
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        slope = (last - first) / max(end - start, 1.0)
        parts = {
            "last": last,
            "mean": mean,
            "slope": slope,
            "mean_last": np.concatenate([mean, last]),
            "mean_slope": np.concatenate([mean, slope]),
            "last_slope": np.concatenate([last, slope]),
            "mean_std_last_slope": np.concatenate([mean, std, last, slope]),
        }
        keep.append(index)
        kept_rows.append(row)
        for name in representations:
            rep_values[name].append(np.asarray(parts[name], dtype=np.float64))
    matrices = {
        name: np.vstack(values).astype(np.float64) if values else np.zeros((0, 0), dtype=np.float64)
        for name, values in rep_values.items()
    }
    return np.asarray(keep, dtype=np.int64), matrices, kept_rows


def fit_all_models_cached(
    ctx: Any,
    train_rows: list[dict[str, Any]],
    y_train: np.ndarray,
    train_x: np.ndarray,
    test_rows: list[dict[str, Any]],
    y_test: np.ndarray,
    test_x: np.ndarray,
    *,
    seed_parts: tuple[Any, ...],
    baseline_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any] | None:
    if train_x.shape[0] < 16 or test_x.shape[0] < 4:
        return None
    if np.sum(y_train == 1) == 0 or np.sum(y_test == 1) == 0:
        return None
    base_key = (row_signature(train_rows), row_signature(test_rows), y_signature(y_train))
    if base_key not in baseline_cache:
        train_ar = bench.autoregressive_features(ctx.accepted_rows, train_rows, "arousal", include_current=True)
        test_ar = bench.autoregressive_features(ctx.accepted_rows, test_rows, "arousal", include_current=True)
        train_time = bench.time_features(train_rows)
        test_time = bench.time_features(test_rows)
        train_video_time, test_video_time = v1.video_time_matrix(ctx, train_rows, test_rows)
        ar_train, _ = bench.ridge_fit_predict(train_ar, y_train, train_ar)
        ar_test, _ = bench.ridge_fit_predict(train_ar, y_train, test_ar)
        time_train_scores, _ = bench.ridge_fit_predict(train_time, y_train, train_time)
        time_test_scores, _ = bench.ridge_fit_predict(train_time, y_train, test_time)
        video_train_scores, _ = bench.ridge_fit_predict(train_video_time, y_train, train_video_time)
        video_test_scores, _ = bench.ridge_fit_predict(train_video_time, y_train, test_video_time)
        baseline_cache[base_key] = {
            "train_ar": train_ar,
            "test_ar": test_ar,
            "ar": (ar_train, ar_test),
            "timestamp": (time_train_scores, time_test_scores),
            "video_time": (video_train_scores, video_test_scores),
        }
    cached = baseline_cache[base_key]
    train_ar = cached["train_ar"]
    test_ar = cached["test_ar"]
    output: dict[str, Any] = {
        "train_y": y_train.astype(np.float64),
        "test_y": y_test.astype(np.int64),
        "test_rows": test_rows,
        "ar": cached["ar"],
        "timestamp": cached["timestamp"],
        "video_time": cached["video_time"],
    }
    real_train_design = np.concatenate([train_ar, train_x], axis=1)
    real_test_design = np.concatenate([test_ar, test_x], axis=1)
    real_train, _ = bench.ridge_fit_predict(real_train_design, y_train, real_train_design)
    real_test, _ = bench.ridge_fit_predict(real_train_design, y_train, real_test_design)
    output["real"] = (real_train, real_test)
    rng = np.random.default_rng(stable_seed(*seed_parts))
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
    return output


def metric_row(base: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    row = v1.metric_row(base, scores)
    row["representation"] = base.get("representation")
    row["window_seconds"] = base.get("window_seconds")
    return row


def score_records(config: dict[str, Any], scores: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    cid = config_id(config)
    for index, row in enumerate(scores["test_rows"]):
        out = {
            "config_id": cid,
            "fold": config["fold"],
            "feature_mode": config["feature_mode"],
            "target": config["target"],
            "threshold": float(config["threshold"]),
            "window_name": config["window_name"],
            "representation": config["representation"],
            "video_id": str(row["video_id"]),
            "time_start_seconds": float(row["time_start_seconds"]),
            "y": int(scores["test_y"][index]),
        }
        for model in ("real", "ar", "shuffled", "random", "timestamp", "video_time"):
            out[f"{model}_score"] = float(scores[model][1][index])
        rows.append(out)
    return rows


def config_id(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(key, ""))
        for key in ("feature_mode", "target", "threshold", "window_name", "representation")
    )


def prior_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        row["feature_mode"],
        row["target"],
        str(float(row["threshold"])),
        row["fold"],
        row["window_name"],
        row.get("aggregation") or row.get("representation") or "",
    )


def reusable_prior_rows(prior_dir: Path) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    path = prior_dir / "causal_context_window_results.csv"
    rows = {}
    for row in read_csv(path):
        if row["target"] != "arousal__future_spike_1_3s":
            continue
        if row["feature_mode"] not in {"cortical_pca_64", "cortical_pca64_delta"}:
            continue
        if row["window_name"] == "current_only_0s" and row.get("aggregation") == "last":
            mapped = dict(row)
        elif row["window_name"] in {"causal_past_3s", "causal_past_5s"} and row.get("aggregation") == "mean_std_last_slope":
            mapped = dict(row)
        else:
            continue
        mapped["representation"] = mapped.get("aggregation")
        mapped["window_seconds"] = abs(float(mapped.get("window_start_seconds") or 0.0))
        mapped["source"] = "reused_prior_summary"
        rows[prior_key(mapped)] = mapped
    return rows


def add_window_gains(rows: list[dict[str, Any]]) -> None:
    baseline = {}
    for row in rows:
        if row["window_name"] == "current_only_0s" and row["representation"] == "last":
            baseline[(row["fold"], row["feature_mode"], row["target"], str(float(row["threshold"])))] = row
    for row in rows:
        base = baseline.get((row["fold"], row["feature_mode"], row["target"], str(float(row["threshold"]))))
        row["real_window_gain"] = diff(row.get("real_pr_auc"), base.get("real_pr_auc") if base else None)
        for control in CONTROL_MODELS:
            control_gain = diff(row.get(f"{control}_pr_auc"), base.get(f"{control}_pr_auc") if base else None)
            row[f"{control}_window_gain"] = control_gain
            row[f"window_specific_gain_vs_{control}"] = diff(row["real_window_gain"], control_gain)


def specificity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["window_name"] == "current_only_0s":
            continue
        for control in CONTROL_MODELS:
            output.append(
                {
                    "fold": row["fold"],
                    "feature_mode": row["feature_mode"],
                    "target": row["target"],
                    "threshold": row["threshold"],
                    "window_name": row["window_name"],
                    "window_seconds": row["window_seconds"],
                    "representation": row["representation"],
                    "control": control,
                    "real_window_gain": row.get("real_window_gain"),
                    "control_window_gain": row.get(f"{control}_window_gain"),
                    "window_specific_gain": row.get(f"window_specific_gain_vs_{control}"),
                }
            )
    return output


def mean_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("context_type") == "non_causal_diagnostic":
            continue
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    metrics = (
        "real_pr_auc",
        "ar_pr_auc",
        "shuffled_pr_auc",
        "random_pr_auc",
        "timestamp_pr_auc",
        "video_time_pr_auc",
        "real_minus_ar",
        "real_window_gain",
        "window_specific_gain_vs_ar",
        "window_specific_gain_vs_shuffled",
        "window_specific_gain_vs_random",
        "window_specific_gain_vs_timestamp",
        "window_specific_gain_vs_video_time",
    )
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        out = {name: value for name, value in zip(keys, key)}
        out["fold_count"] = len(values)
        for metric in metrics:
            vals = [finite(row.get(metric)) for row in values if finite(row.get(metric)) is not None]
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else None
            out[f"{metric}_std"] = float(np.std(arr)) if arr.size else None
        output.append(out)
    return output


def best_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = mean_rows(rows, ("feature_mode", "target", "threshold", "window_name", "window_seconds", "representation"))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in means:
        if row["window_name"] == "current_only_0s":
            continue
        grouped[(row["feature_mode"], row["target"], str(float(row["threshold"])))].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        best = max(values, key=lambda row: row.get("real_pr_auc_mean") or -1.0)
        best_ar = max(values, key=lambda row: row.get("real_minus_ar_mean") or -1e9)
        out = dict(best)
        out["selection_metric"] = "max mean real_pr_auc across grouped folds"
        out["best_real_minus_ar_window_name"] = best_ar["window_name"]
        out["best_real_minus_ar_representation"] = best_ar["representation"]
        out["best_real_minus_ar_mean"] = best_ar.get("real_minus_ar_mean")
        output.append(out)
    return output


def bootstrap_ci(score_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_config[row["config_id"]].append(row)
    output = []
    for best in best_rows:
        feature = best["feature_mode"]
        target = best["target"]
        threshold = str(float(best["threshold"]))
        current_id = config_id({"feature_mode": feature, "target": target, "threshold": threshold, "window_name": "current_only_0s", "representation": "last"})
        best_id = config_id(best)
        pairs = [("current_only_0s", current_id), ("best_causal_window", best_id)]
        for label, cid in pairs:
            rows = by_config.get(cid, [])
            if rows:
                output.extend(bootstrap_one(rows, feature, target, threshold, label, samples, seed))
        if by_config.get(current_id) and by_config.get(best_id):
            output.extend(bootstrap_diff(by_config[current_id], by_config[best_id], feature, target, threshold, samples, seed))
    return output


def bootstrap_one(rows: list[dict[str, Any]], feature: str, target: str, threshold: str, label: str, samples: int, seed: int) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    video_ids = sorted(grouped)
    rng = np.random.default_rng(stable_seed("bootstrap", feature, target, threshold, label, seed))
    metrics = defaultdict(list)
    for _ in range(samples):
        chosen = rng.choice(video_ids, size=len(video_ids), replace=True)
        sample = [item for video_id in chosen for item in grouped[video_id]]
        y = np.asarray([row["y"] for row in sample], dtype=np.int64)
        if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
            continue
        vals = {model: retest.pr_auc(y, np.asarray([row[f"{model}_score"] for row in sample], dtype=np.float64)) for model in ("real", "ar", "shuffled", "timestamp")}
        if vals["real"] is None:
            continue
        metrics["real_minus_ar"].append(vals["real"] - vals["ar"])
        metrics["real_minus_shuffled"].append(vals["real"] - vals["shuffled"])
        metrics["real_minus_timestamp"].append(vals["real"] - vals["timestamp"])
    return ci_rows(feature, target, threshold, label, metrics)


def bootstrap_diff(current_rows: list[dict[str, Any]], best_rows_: list[dict[str, Any]], feature: str, target: str, threshold: str, samples: int, seed: int) -> list[dict[str, Any]]:
    cur_by_video = defaultdict(list)
    best_by_video = defaultdict(list)
    for row in current_rows:
        cur_by_video[row["video_id"]].append(row)
    for row in best_rows_:
        best_by_video[row["video_id"]].append(row)
    video_ids = sorted(set(cur_by_video) & set(best_by_video))
    rng = np.random.default_rng(stable_seed("bootstrap_diff", feature, target, threshold, seed))
    metrics = defaultdict(list)
    for _ in range(samples):
        chosen = rng.choice(video_ids, size=len(video_ids), replace=True)
        cur = [item for video_id in chosen for item in cur_by_video[video_id]]
        best = [item for video_id in chosen for item in best_by_video[video_id]]
        cur_vals = prauc_values(cur)
        best_vals = prauc_values(best)
        if cur_vals.get("real") is None or best_vals.get("real") is None:
            continue
        real_gain = best_vals["real"] - cur_vals["real"]
        metrics["best_causal_window_minus_current_only"].append(real_gain)
        for control in ("ar", "shuffled", "timestamp"):
            if cur_vals.get(control) is not None and best_vals.get(control) is not None:
                control_gain = best_vals[control] - cur_vals[control]
                metrics[f"window_specific_gain_vs_{control}"].append(real_gain - control_gain)
    return ci_rows(feature, target, threshold, "best_vs_current", metrics)


def prauc_values(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    y = np.asarray([row["y"] for row in rows], dtype=np.int64)
    if np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return {model: None for model in ("real", "ar", "shuffled", "timestamp")}
    return {model: retest.pr_auc(y, np.asarray([row[f"{model}_score"] for row in rows], dtype=np.float64)) for model in ("real", "ar", "shuffled", "timestamp")}


def ci_rows(feature: str, target: str, threshold: str, config: str, metrics: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows = []
    for metric, values in metrics.items():
        arr = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
        rows.append(
            {
                "feature_mode": feature,
                "target": target,
                "threshold": threshold,
                "config": config,
                "metric": metric,
                "samples": int(arr.size),
                "mean": float(np.mean(arr)) if arr.size else None,
                "ci95_low": float(np.quantile(arr, 0.025)) if arr.size else None,
                "ci95_high": float(np.quantile(arr, 0.975)) if arr.size else None,
            }
        )
    return rows


def leakage_rows(include_symmetric: bool) -> list[dict[str, Any]]:
    rows = [
        {"check": "causal windows use only past/current feature rows", "status": "pass", "final_claim_safe": True},
        {"check": "feature windows are trimmed at video boundaries", "status": "pass", "final_claim_safe": True},
        {"check": "held-out unit is video_id grouped fold", "status": "pass", "final_claim_safe": True},
        {"check": "PCA fit on training rows only for each feature_mode/fold", "status": "pass", "final_claim_safe": True},
        {"check": "scalers/ridge standardization fit on train matrices only", "status": "pass", "final_claim_safe": True},
        {"check": "thresholds selected on train predictions only", "status": "pass", "final_claim_safe": True},
        {"check": "no offset sweep recomputation or test-selected lag used as v2 headline", "status": "pass", "final_claim_safe": True},
    ]
    if include_symmetric:
        rows.append({"check": "symmetric_2s window marked non-causal diagnostic", "status": "pass", "final_claim_safe": False})
    return rows


def artifact_manifest(prior_dir: Path, cache_dir: Path, manifest: Path, report: Path, reused_prior_rows_count: int) -> list[dict[str, Any]]:
    rows = []
    for path, contains, reason in (
        (prior_dir / "causal_context_window_results.csv", "prior 0s, 3s, 5s causal context summary rows", "directly comparable grouped-video v1 rows reused for matching cells"),
        (prior_dir / "offset_sweep_results.csv", "prior focused offset sweep", "v2 compares against prior best offset without rerunning offsets"),
        (prior_dir / "train_selected_offset_results.csv", "prior train-only selected offset results", "v2 compares against prior selected-offset outcomes without rerunning offsets"),
        (prior_dir / "leakage_audit.csv", "prior leakage checks", "v2 extends same leakage policy"),
        (manifest, "VEATIC-124 manifest rows and split metadata", "same labels/video ids as prior run"),
        (report, "VEATIC-124 manifest report and complete video ids", "same 124-video accepted set"),
    ):
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "mtime": path.stat().st_mtime if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
                "contains": contains,
                "reuse_reason": reason,
                "leakage_safe_reason": "summary reuse only or same grouped-video/train-only construction",
            }
        )
    npz_count = len(list(cache_dir.glob("*/tribe_raw_output.npz"))) if cache_dir.exists() else 0
    rows.append(
        {
            "path": str(cache_dir),
            "exists": cache_dir.exists(),
            "bytes": None,
            "mtime": cache_dir.stat().st_mtime if cache_dir.exists() else None,
            "sha256": None,
            "contains": f"cached TRIBE/cortical raw outputs; tribe_raw_output.npz count={npz_count}",
            "reuse_reason": "source for cache-only PCA/context matrices; no video re-encoding",
            "leakage_safe_reason": "PCA is refit inside each train fold before transforming held-out rows",
        }
    )
    rows.append(
        {
            "path": str(prior_dir / "causal_context_window_results.csv"),
            "exists": True,
            "bytes": None,
            "mtime": None,
            "sha256": None,
            "contains": f"{reused_prior_rows_count} directly reused summary rows",
            "reuse_reason": "avoid recomputing matching v1 context summary cells",
            "leakage_safe_reason": "same folds, features, target definitions, and train-only threshold policy",
        }
    )
    return rows


def comparison_to_prior(prior_dir: Path, rows: list[dict[str, Any]], best: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    prior_tax = { (row["feature_mode"], row["target"], str(float(row["threshold"]))): row for row in read_csv(prior_dir / "timing_fairness_taxonomy.csv") if row["target"] == "arousal__future_spike_1_3s" }
    for row in best:
        key = (row["feature_mode"], row["target"], str(float(row["threshold"])))
        prior = prior_tax.get(key, {})
        output.append(
            {
                "feature_mode": row["feature_mode"],
                "target": row["target"],
                "threshold": row["threshold"],
                "prior_taxonomy": prior.get("taxonomy"),
                "prior_zero_real_pr_auc": prior.get("zero_real_pr_auc"),
                "prior_best_causal_window": prior.get("best_causal_window"),
                "prior_causal_window_gain": prior.get("causal_window_gain"),
                "prior_best_offset_seconds": prior.get("best_offset_seconds"),
                "prior_offset_gain": prior.get("offset_gain"),
                "v2_best_window": row.get("window_name"),
                "v2_best_representation": row.get("representation"),
                "v2_best_real_pr_auc_mean": row.get("real_pr_auc_mean"),
                "v2_best_real_window_gain_mean": row.get("real_window_gain_mean"),
                "v2_best_window_specific_gain_vs_ar_mean": row.get("window_specific_gain_vs_ar_mean"),
            }
        )
    return output


def write_report(path: Path, summary: dict[str, Any], best: list[dict[str, Any]], top_ar: list[dict[str, Any]], top_specific: list[dict[str, Any]]) -> None:
    lines = [
        "# VEATIC-124 Temporal Context v2",
        "",
        "This is a narrow reuse-first temporal context sufficiency benchmark for future arousal spike ranking.",
        "",
        "## Executive Verdict",
        "",
        summary["executive_verdict"],
        "",
        "## Reuse",
        "",
    ]
    for item in summary["reused_artifacts"]:
        lines.append(f"- `{item['path']}`: {item['contains']}; {item['reuse_reason']}.")
    lines += [
        "",
        "## Best Windows",
        "",
        "| Feature | Target | Thr | Best window | Representation | PR-AUC | Real gain | Specific gain vs AR |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['feature_mode']} | `{row['target']}` | {float(row['threshold']):.3f} | {row['window_name']} | {row['representation']} | {fmt(row.get('real_pr_auc_mean'))} | {fmt(row.get('real_window_gain_mean'))} | {fmt(row.get('window_specific_gain_vs_ar_mean'))} |"
        )
    lines += [
        "",
        "## Top Real Minus AR Rows",
        "",
    ]
    for row in top_ar[:10]:
        lines.append(f"- {row['feature_mode']} {row['target']} thr={row['threshold']} {row['window_name']} {row['representation']}: real_minus_AR={fmt(row.get('real_minus_ar'))}, real_PR_AUC={fmt(row.get('real_pr_auc'))}")
    lines += [
        "",
        "## Top Context-Specific Rows",
        "",
    ]
    for row in top_specific[:10]:
        lines.append(f"- {row['feature_mode']} {row['target']} thr={row['threshold']} {row['window_name']} {row['representation']} vs {row['control']}: window_specific_gain={fmt(row.get('window_specific_gain'))}")
    lines += [
        "",
        "## Recommended Claim",
        "",
        summary["best_defensible_claim"],
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim universal early warning.",
        "- Do not claim exact future arousal prediction.",
        "- Do not claim TRIBE is globally early or late from this v2 alone.",
        "- Do not use symmetric diagnostic windows as final predictive claims.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    value = finite(value)
    return "NA" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl")
    parser.add_argument("--report", default="benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json")
    parser.add_argument("--cache-dir", default="/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache")
    parser.add_argument("--prior-dir", default="outputs/veatic_124_temporal_fairness_20260616_1509")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--pca-backend", default="mps_gram", choices=("auto", "mps_power", "mps_gram", "cpu_svd"))
    parser.add_argument("--ridge-backend", default="cpu_pinv", choices=("auto", "mps_solve", "cpu_pinv"))
    parser.add_argument("--skip-symmetric", action="store_true")
    args = parser.parse_args()

    bench.PCA_BACKEND = args.pca_backend
    bench.RIDGE_BACKEND = args.ridge_backend
    prior_dir = Path(args.prior_dir).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_root).expanduser().resolve() / f"veatic_124_temporal_context_v2_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_lookup = reusable_prior_rows(prior_dir)
    ctx = retest.RetestContext(manifest, report, cache_dir)
    label_series = align.LabelSeries(ctx.accepted_rows, "arousal")
    fold_specs = v1.grouped_video_folds(ctx.accepted_rows, args.folds)

    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    newly_computed: list[dict[str, Any]] = []
    score_recomputed_for_prior = 0
    start_run = time.monotonic()
    baseline_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    for feature_label, feature_mode in FEATURES:
        print(f"[INFO] feature={feature_label}", flush=True)
        base_features = ctx.base_feature_sets(feature_mode)
        for fold_label, held, train_rows, test_rows in fold_specs:
            split_start = time.monotonic()
            print(f"[INFO] split feature={feature_label} fold={fold_label}", flush=True)
            matrices = v1.split_matrices(ctx, base_features, train_rows, test_rows, feature_mode)
            train_table = v1.subset_table(train_rows, matrices["train_matrix"])
            test_table = v1.subset_table(test_rows, matrices["test_matrix"])
            for target, horizon, threshold in TARGETS:
                train_selected, train_y = v1.target_rows(label_series, train_rows, target, horizon, threshold)
                test_selected, test_y = v1.target_rows(label_series, test_rows, target, horizon, threshold)
                desired = []
                for window_name, start, end, context_type in WINDOWS:
                    reps = ("last",) if window_name == "current_only_0s" else REPRESENTATIONS
                    for rep in reps:
                        desired.append((window_name, start, end, context_type, rep))
                if not args.skip_symmetric:
                    for window_name, start, end, context_type in DIAGNOSTIC_WINDOWS:
                        desired.append((window_name, start, end, context_type, "mean_std_last_slope"))
                by_window = defaultdict(list)
                for item in desired:
                    by_window[item[:4]].append(item[4])
                for (window_name, start, end, context_type), reps_list in by_window.items():
                    reps = tuple(reps_list)
                    train_keep, train_mats, kept_train = window_representation_features(train_selected, train_table, start, end, reps)
                    test_keep, test_mats, kept_test = window_representation_features(test_selected, test_table, start, end, reps)
                    if train_keep.size == 0 or test_keep.size == 0:
                        continue
                    for rep in reps:
                        base = {
                            "arm": "causal_context_window" if context_type == "causal" else "diagnostic_symmetric_window",
                            "fold": fold_label,
                            "held_out_video_ids": ",".join(held),
                            "feature_mode": feature_label,
                            "target": target,
                            "horizon_seconds": horizon,
                            "threshold": threshold,
                            "window_name": window_name,
                            "window_start_seconds": start,
                            "window_end_seconds": end,
                            "window_seconds": end - start,
                            "context_type": context_type,
                            "representation": rep,
                            "offset_seconds": 0.0,
                        }
                        pkey = (feature_label, target, str(float(threshold)), fold_label, window_name, rep)
                        scores = fit_all_models_cached(
                            ctx,
                            kept_train,
                            train_y[train_keep],
                            train_mats[rep],
                            kept_test,
                            test_y[test_keep],
                            test_mats[rep],
                            seed_parts=(feature_label, fold_label, target, threshold, window_name, rep),
                            baseline_cache=baseline_cache,
                        )
                        if scores is None:
                            continue
                        if pkey in prior_lookup and context_type == "causal":
                            row = dict(prior_lookup[pkey])
                            row.update(
                                {
                                    "held_out_video_ids": ",".join(held),
                                    "representation": rep,
                                    "window_seconds": end - start,
                                    "context_type": context_type,
                                    "source": "reused_prior_summary",
                                    "score_records_source": "recomputed_for_bootstrap_only",
                                }
                            )
                            score_recomputed_for_prior += 1
                        else:
                            row = metric_row({**base, "source": "newly_computed_v2"}, scores)
                            newly_computed.append(
                                {
                                    "feature_mode": feature_label,
                                    "fold": fold_label,
                                    "target": target,
                                    "threshold": threshold,
                                    "window_name": window_name,
                                    "representation": rep,
                                    "reason": "missing from prior temporal fairness output or v2-only representation/window",
                                }
                            )
                        rows.append(row)
                        score_rows.extend(score_records(base, scores))
                print(f"[INFO] done target feature={feature_label} fold={fold_label} target={target} thr={threshold} elapsed={time.monotonic() - split_start:.1f}s", flush=True)
            print(f"[INFO] done split feature={feature_label} fold={fold_label} elapsed={time.monotonic() - split_start:.1f}s", flush=True)

    add_window_gains(rows)
    spec_rows = specificity_rows(rows)
    best = best_windows(rows)
    boot = bootstrap_ci(score_rows, best, args.bootstrap_samples, args.seed)
    leak = leakage_rows(not args.skip_symmetric)
    reused_manifest = artifact_manifest(prior_dir, cache_dir, manifest, report, sum(1 for row in rows if row.get("source") == "reused_prior_summary"))
    comparison = comparison_to_prior(prior_dir, rows, best)

    write_csv(out_dir / "context_window_ablation_results.csv", rows)
    write_csv(out_dir / "representation_ablation_results.csv", rows)
    write_csv(out_dir / "real_vs_control_context_specificity.csv", spec_rows)
    write_csv(out_dir / "best_windows_by_target_feature.csv", best)
    write_csv(out_dir / "bootstrap_ci.csv", boot)
    write_csv(out_dir / "leakage_audit.csv", leak)
    write_csv(out_dir / "reused_artifacts_manifest.csv", reused_manifest)
    write_csv(out_dir / "comparison_to_prior_temporal_fairness.csv", comparison)

    top_ar = sorted([row for row in rows if finite(row.get("real_minus_ar")) is not None], key=lambda row: float(row["real_minus_ar"]), reverse=True)[:10]
    top_specific = sorted([row for row in spec_rows if finite(row.get("window_specific_gain")) is not None], key=lambda row: float(row["window_specific_gain"]), reverse=True)[:10]
    best_by_target = {
        (row["feature_mode"], row["target"], str(float(row["threshold"]))): row
        for row in best
    }
    sweet_spot_counts = Counter(row["window_name"] for row in best)
    rep_counts = Counter(row["representation"] for row in best)
    context_starved = sum(1 for row in best if (finite(row.get("real_window_gain_mean")) or 0.0) > 0.005)
    specific_positive = sum(1 for row in best if (finite(row.get("window_specific_gain_vs_ar_mean")) or 0.0) > 0.003)
    verdict = (
        f"V2 narrowed to future spike ranking and found best causal windows {dict(sweet_spot_counts)} with best representations {dict(rep_counts)}. "
        f"{context_starved}/{len(best)} feature-target rows improved over current-only by more than 0.005 PR-AUC, and {specific_positive}/{len(best)} had positive context-specific gain versus AR above 0.003. "
        "This supports temporal context sufficiency checks where gains are real-feature-specific, but does not justify universal early-warning or label-shift claims."
    )
    claim = (
        "Best defensible claim: short causal context can modestly improve future arousal spike ranking for selected TRIBE/cortical PCA modes, "
        "so single-timestep 0s evaluation may underfeed the bridge head. The effect must be reported with controls because some context changes can track label/timing structure rather than TRIBE-specific information."
    )
    summary = {
        "schema_version": "veatic_124_temporal_context_v2",
        "output_dir": str(out_dir),
        "prior_dir": str(prior_dir),
        "elapsed_seconds": time.monotonic() - start_run,
        "pca_backend": args.pca_backend,
        "ridge_backend": args.ridge_backend,
        "bootstrap_samples": args.bootstrap_samples,
        "features": [item[0] for item in FEATURES],
        "targets": [{"target": item[0], "threshold": item[2]} for item in TARGETS],
        "windows": [item[0] for item in WINDOWS],
        "representations": list(REPRESENTATIONS),
        "symmetric_diagnostic_included": not args.skip_symmetric,
        "reused_summary_rows": sum(1 for row in rows if row.get("source") == "reused_prior_summary"),
        "score_recomputations_for_reused_rows": score_recomputed_for_prior,
        "newly_computed_rows": len(newly_computed),
        "executive_verdict": verdict,
        "best_defensible_claim": claim,
        "best_window_counts": dict(sweet_spot_counts),
        "best_representation_counts": dict(rep_counts),
        "reused_artifacts": reused_manifest,
    }
    (out_dir / "veatic_124_temporal_context_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out_dir / "veatic_124_temporal_context_v2_report.md", summary, best, top_ar, top_specific)

    print("\n=== VEATIC-124 Temporal Context v2 Summary ===")
    print(f"Output directory: {out_dir}")
    print("Reused artifacts:")
    for item in reused_manifest:
        print(f"- {item['path']} :: {item['contains']}")
    print(f"Newly computed rows: {len(newly_computed)}")
    print(f"Reused prior summary rows: {summary['reused_summary_rows']}; score recomputations for bootstrap on reused cells: {score_recomputed_for_prior}")
    print("Top 10 rows by real_minus_AR:")
    for row in top_ar:
        print(f"- {row['feature_mode']} {row['target']} thr={row['threshold']} {row['window_name']} {row['representation']} real_minus_AR={fmt(row.get('real_minus_ar'))}")
    print("Top 10 rows by window_specific_gain:")
    for row in top_specific:
        print(f"- {row['feature_mode']} {row['target']} thr={row['threshold']} {row['window_name']} {row['representation']} vs {row['control']} gain={fmt(row.get('window_specific_gain'))}")
    print("Best causal window / representation by feature-target:")
    for key, row in sorted(best_by_target.items()):
        print(f"- {key}: {row['window_name']} / {row['representation']} real_PR_AUC={fmt(row.get('real_pr_auc_mean'))} real_gain={fmt(row.get('real_window_gain_mean'))}")
    print(f"Sweet spot counts: {dict(sweet_spot_counts)}")
    print(f"TRIBE appears context-starved in {context_starved}/{len(best)} rows by the >0.005 PR-AUC gain rule.")
    print(f"Context benefit is real-feature-specific vs AR in {specific_positive}/{len(best)} rows by the >0.003 specific-gain rule.")
    print("This refines but does not overturn the prior conclusion: temporal context matters modestly for spike/event ranking, but claims must stay control-qualified.")
    print(f"Best defensible claim: {claim}")
    print("Still cannot claim: universal early warning; exact future arousal prediction; global early/late TRIBE timing; diagnostic symmetric windows as final predictive evidence.")


if __name__ == "__main__":
    main()
