"""Run lightweight VEATIC temporal affect baselines.

This is intentionally annotation-only: it validates split difficulty before any
heavy TRIBE extraction is allowed to become the bottleneck or the explanation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TARGETS = ("valence", "arousal")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique) != len(values):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def corr(y_true: np.ndarray, y_pred: np.ndarray, *, spearman: bool = False) -> float | None:
    if len(y_true) < 2:
        return None
    a = rankdata(y_true) if spearman else y_true
    b = rankdata(y_pred) if spearman else y_pred
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "pearson": corr(y_true, y_pred),
        "spearman": corr(y_true, y_pred, spearman=True),
    }


def rows_by_video(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    for video_rows in grouped.values():
        video_rows.sort(key=lambda item: int(item["frame_index"]))
    return dict(grouped)


def target_array(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.asarray([row["targets"][target] for row in rows], dtype=np.float64)


def time_features(rows: list[dict[str, Any]]) -> np.ndarray:
    grouped = rows_by_video(rows)
    max_time = {
        video_id: max(float(row["time_start_seconds"]) for row in video_rows) or 1.0
        for video_id, video_rows in grouped.items()
    }
    features = []
    for row in rows:
        seconds = float(row["time_start_seconds"])
        frac = seconds / max_time[str(row["video_id"])]
        features.append([1.0, seconds, frac, frac * frac, math.sin(math.tau * frac), math.cos(math.tau * frac)])
    return np.asarray(features, dtype=np.float64)


def ridge_predict(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    target: str,
    alpha: float = 1.0,
) -> np.ndarray:
    x_train = time_features(train_rows)
    y_train = target_array(train_rows, target)
    x_test = time_features(test_rows)
    penalty = np.eye(x_train.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x_train.T @ x_train + penalty) @ x_train.T @ y_train
    return x_test @ beta


def mean_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], target: str) -> np.ndarray:
    return np.full(len(test_rows), float(np.mean(target_array(train_rows, target))), dtype=np.float64)


def video_mean_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], target: str) -> np.ndarray:
    train_grouped = rows_by_video(train_rows)
    global_mean = float(np.mean(target_array(train_rows, target)))
    video_means = {
        video_id: float(np.mean(target_array(video_rows, target)))
        for video_id, video_rows in train_grouped.items()
    }
    return np.asarray([video_means.get(str(row["video_id"]), global_mean) for row in test_rows], dtype=np.float64)


def persistence_predict(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], target: str) -> np.ndarray:
    """Predict with the latest known training label from the same video.

    This is a strong autocorrelation baseline for within-video splits. For
    leave-video-out it falls back to the global training mean.
    """
    global_mean = float(np.mean(target_array(train_rows, target)))
    history: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in train_rows:
        history[str(row["video_id"])].append((int(row["frame_index"]), float(row["targets"][target])))
    for values in history.values():
        values.sort()

    preds = []
    for row in test_rows:
        frame = int(row["frame_index"])
        candidates = [value for index, value in history.get(str(row["video_id"]), []) if index < frame]
        preds.append(candidates[-1] if candidates else global_mean)
    return np.asarray(preds, dtype=np.float64)


def eval_fold(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_videos": len({row["video_id"] for row in train_rows}),
        "test_videos": len({row["video_id"] for row in test_rows}),
        "targets": {},
    }
    for target in TARGETS:
        y_true = target_array(test_rows, target)
        predictions = {
            "mean_train": mean_predict(train_rows, test_rows, target),
            "time_ridge": ridge_predict(train_rows, test_rows, target),
            "video_mean": video_mean_predict(train_rows, test_rows, target),
            "persistence_previous_known": persistence_predict(train_rows, test_rows, target),
        }
        result["targets"][target] = {
            name: metrics(y_true, y_pred)
            for name, y_pred in predictions.items()
        }
    return result


def fixed_split(rows: list[dict[str, Any]], split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    train = [row for row in rows if row["splits"][split_name] == "train"]
    test = [row for row in rows if row["splits"][split_name] == "test"]
    gap = sum(1 for row in rows if row["splits"][split_name] == "gap")
    return train, test, gap


def leave_video_out_folds(rows: list[dict[str, Any]], max_folds: int | None) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    grouped = rows_by_video(rows)
    video_ids = sorted(grouped, key=lambda item: int(item) if item.isdigit() else item)
    if max_folds is not None:
        video_ids = video_ids[:max_folds]
    folds = []
    for video_id in video_ids:
        test = grouped[video_id]
        train = [row for other_id, video_rows in grouped.items() if other_id != video_id for row in video_rows]
        folds.append((video_id, train, test))
    return folds


def aggregate_lvo(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "folds": len(fold_results),
        "train_rows_mean": float(np.mean([fold["train_rows"] for fold in fold_results])),
        "test_rows_total": int(sum(fold["test_rows"] for fold in fold_results)),
        "targets": {},
    }
    for target in TARGETS:
        output["targets"][target] = {}
        condition_names = fold_results[0]["targets"][target].keys()
        for condition in condition_names:
            output["targets"][target][condition] = {}
            metric_names = fold_results[0]["targets"][target][condition].keys()
            for metric_name in metric_names:
                values = [
                    fold["targets"][target][condition][metric_name]
                    for fold in fold_results
                    if fold["targets"][target][condition][metric_name] is not None
                ]
                output["targets"][target][condition][metric_name] = (
                    float(np.mean(values)) if values else None
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/veatic/veatic_manifest_1hz.jsonl")
    parser.add_argument("--output", default="benchmarks/veatic/veatic_annotation_baselines.json")
    parser.add_argument("--max-lvo-folds", type=int, default=None)
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    rows = load_manifest(manifest)
    official_train, official_test, official_gap = fixed_split(rows, "official_70_30")
    blocked_train, blocked_test, blocked_gap = fixed_split(rows, "blocked_temporal_gap")

    lvo_fold_results = []
    for video_id, train_rows, test_rows in leave_video_out_folds(rows, args.max_lvo_folds):
        fold = eval_fold(train_rows, test_rows)
        fold["held_out_video_id"] = video_id
        lvo_fold_results.append(fold)

    report = {
        "schema_version": "veatic_annotation_baseline_report_v1",
        "manifest": str(manifest),
        "rows": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "targets": list(TARGETS),
        "modes": {
            "mode_a_official_veatic_70_30": {
                "description": "Official first-70-percent train / last-30-percent test split.",
                "gap_rows": official_gap,
                **eval_fold(official_train, official_test),
            },
            "mode_b_blocked_temporal_gap": {
                "description": "Train early, discard middle gap, test later frames.",
                "gap_rows": blocked_gap,
                **eval_fold(blocked_train, blocked_test),
            },
            "mode_c_leave_video_out": {
                "description": "Hold out whole videos; reports mean metric across held-out-video folds.",
                "max_folds": args.max_lvo_folds,
                "aggregate": aggregate_lvo(lvo_fold_results),
                "folds": lvo_fold_results,
            },
        },
        "interpretation_warning": (
            "These are annotation-only baselines. Neuro features must beat the "
            "time/persistence and non-neuro baselines inside the same split mode "
            "before claiming additive neuro signal."
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "videos": report["videos"]}, indent=2))


if __name__ == "__main__":
    main()
