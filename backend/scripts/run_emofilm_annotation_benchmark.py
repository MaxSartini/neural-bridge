"""Lightweight Emo-FilM annotation benchmark.

This runner operates on window-level annotation manifests. It can run baseline
conditions immediately and will include neuro conditions when a future adapter
adds per-window neuro feature paths to the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def correlation(predicted: np.ndarray, observed: np.ndarray) -> float | None:
    if predicted.size < 2 or np.isclose(np.std(predicted), 0.0) or np.isclose(np.std(observed), 0.0):
        return None
    value = float(np.corrcoef(predicted, observed)[0, 1])
    return value if math.isfinite(value) else None


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranked


def score(predicted: np.ndarray, observed: np.ndarray, film_ids: np.ndarray, times: np.ndarray, seed: int) -> dict[str, Any]:
    errors = predicted - observed
    absolute = np.abs(errors)
    rng = np.random.default_rng(seed)
    bootstrap = [
        float(np.mean(absolute[rng.integers(0, absolute.size, absolute.size)]))
        for _ in range(1000)
    ]
    bootstrap.sort()
    by_film_corr = []
    lagged: dict[str, Any] = {}
    for film_id in sorted(set(film_ids)):
        mask = film_ids == film_id
        if int(mask.sum()) >= 5:
            order = np.argsort(times[mask])
            by_film_corr.append(correlation(predicted[mask][order], observed[mask][order]))
    by_film_corr = [value for value in by_film_corr if value is not None]
    if observed.size >= 10:
        lag_values = {}
        for lag in range(-5, 6):
            if lag < 0:
                lag_values[str(lag)] = correlation(predicted[-lag:], observed[:lag])
            elif lag > 0:
                lag_values[str(lag)] = correlation(predicted[:-lag], observed[lag:])
            else:
                lag_values[str(lag)] = correlation(predicted, observed)
        lagged = {
            "lags_seconds": lag_values,
            "best_lag_seconds": max(
                (int(lag) for lag, value in lag_values.items() if value is not None),
                key=lambda lag: lag_values[str(lag)],
                default=None,
            ),
        }
    threshold = float(np.percentile(observed, 90))
    predicted_threshold = float(np.percentile(predicted, 90))
    observed_peak = observed >= threshold
    predicted_peak = predicted >= predicted_threshold
    tp = int(np.sum(observed_peak & predicted_peak))
    fp = int(np.sum(~observed_peak & predicted_peak))
    fn = int(np.sum(observed_peak & ~predicted_peak))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "n": int(observed.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson": correlation(predicted, observed),
        "spearman": correlation(ranks(predicted), ranks(observed)),
        "time_series_correlation_mean_by_film": float(np.mean(by_film_corr)) if by_film_corr else None,
        "lagged_correlation": lagged,
        "peak_detection_top10": {
            "threshold_observed": threshold,
            "threshold_predicted": predicted_threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "mae_bootstrap_95_ci": [bootstrap[24], bootstrap[974]],
    }


def aggregate(split_scores: list[dict[str, Any]], baseline_maes: list[float]) -> dict[str, Any]:
    keys = ("mae", "rmse", "pearson", "spearman", "time_series_correlation_mean_by_film")
    out: dict[str, Any] = {}
    for key in keys:
        values = [entry[key] for entry in split_scores if entry.get(key) is not None]
        out[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    out["paired_mae_delta_vs_mean_baseline"] = {
        "mean": float(np.mean([entry["mae"] - base for entry, base in zip(split_scores, baseline_maes)])),
        "negative_is_improvement": True,
    }
    out["split_metrics"] = split_scores
    return out


def summarize_target(values: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    film_ids = sorted({row["film_id"] for row in rows})
    global_mean = float(np.mean(values))
    global_std = float(np.std(values))
    if np.isclose(global_std, 0.0):
        skew = None
    else:
        skew = float(np.mean(np.power((values - global_mean) / global_std, 3)))
    per_film = {}
    for film_id in film_ids:
        mask = np.asarray([row["film_id"] == film_id for row in rows], dtype=bool)
        subset = values[mask]
        per_film[film_id] = {
            "mean": float(np.mean(subset)),
            "std": float(np.std(subset)),
            "min": float(np.min(subset)),
            "max": float(np.max(subset)),
            "missing": int(np.sum(~np.isfinite(subset))),
        }
    per_film_means = np.asarray([item["mean"] for item in per_film.values()], dtype=np.float64)
    per_film_stds = np.asarray([item["std"] for item in per_film.values()], dtype=np.float64)
    return {
        "global": {
            "mean": global_mean,
            "std": global_std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "skew": skew,
            "missing": int(np.sum(~np.isfinite(values))),
        },
        "per_film": per_film,
        "zscore_assessment": {
            "global_mean_near_zero": abs(global_mean) < 0.1,
            "global_std_near_one": 0.8 <= global_std <= 1.2,
            "per_film_means_near_zero": bool(np.all(np.abs(per_film_means) < 0.2)),
            "per_film_stds_near_one": bool(np.all((per_film_stds > 0.8) & (per_film_stds < 1.2))),
            "interpretation": (
                "Heuristic only. Derivative aggregate files look z-like but are not guaranteed "
                "to be independently z-scored per film or target."
            ),
        },
    }


def autocorrelation_by_film(values: np.ndarray, rows: list[dict[str, Any]], max_lag: int = 10) -> dict[str, Any]:
    film_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        film_to_indices[row["film_id"]].append(index)
    lag_report = {}
    for lag in range(1, max_lag + 1):
        corrs = []
        for indices in film_to_indices.values():
            ordered = sorted(indices, key=lambda idx: rows[idx]["time_start_seconds"])
            if len(ordered) <= lag:
                continue
            series = values[ordered]
            corr = correlation(series[:-lag], series[lag:])
            if corr is not None:
                corrs.append(corr)
        lag_report[str(lag)] = {
            "mean": float(np.mean(corrs)) if corrs else None,
            "films": len(corrs),
        }
    return lag_report


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int) -> np.ndarray:
    from catboost import CatBoostRegressor

    if (
        train_x.shape[1] == 0
        or np.all(np.std(train_x, axis=0) == 0)
        or np.isclose(np.std(train_y), 0.0)
    ):
        return np.full(test_x.shape[0], np.mean(train_y), dtype=np.float32)
    model = CatBoostRegressor(
        iterations=160,
        depth=4,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    try:
        model.fit(train_x, train_y)
    except Exception as exc:
        if "All train targets are equal" in str(exc):
            return np.full(test_x.shape[0], np.mean(train_y), dtype=np.float32)
        raise
    return np.asarray(model.predict(test_x), dtype=np.float32)


def time_features(rows: list[dict[str, Any]]) -> np.ndarray:
    features = []
    max_time_by_film = defaultdict(float)
    for row in rows:
        max_time_by_film[row["film_id"]] = max(max_time_by_film[row["film_id"]], float(row["time_end_seconds"]))
    for row in rows:
        time_start = float(row["time_start_seconds"])
        duration = max(max_time_by_film[row["film_id"]], 1.0)
        phase = time_start / duration
        features.append(
            [
                time_start,
                phase,
                math.sin(2.0 * math.pi * phase),
                math.cos(2.0 * math.pi * phase),
                float(row["window_index"]),
            ]
        )
    return np.asarray(features, dtype=np.float32)


def video_container_features(rows: list[dict[str, Any]]) -> np.ndarray:
    # Annotation-only local bundle has no film media files. Keep an explicit
    # constant/missing baseline so reports show this condition is not informative.
    return np.zeros((len(rows), 1), dtype=np.float32)


def film_id_features(rows: list[dict[str, Any]]) -> np.ndarray:
    films = sorted({row["film_id"] for row in rows})
    index = {film_id: i for i, film_id in enumerate(films)}
    matrix = np.zeros((len(rows), len(films)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        matrix[row_index, index[row["film_id"]]] = 1.0
    return matrix


def neuro_features(rows: list[dict[str, Any]], key: str) -> np.ndarray | None:
    paths = [row.get("neuro_feature_paths", {}).get(key) for row in rows]
    if not all(paths):
        return None
    vectors = []
    for path in paths:
        with np.load(path) as bundle:
            vectors.append(np.asarray(bundle["features"], dtype=np.float32))
    return np.stack(vectors)


def make_splits(rows: list[dict[str, Any]], mode: str, seed: int, test_fraction: float) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    film_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        film_to_indices[row["film_id"]].append(index)
    splits = []
    if mode == "within_film_temporal":
        for film_id, indices in sorted(film_to_indices.items()):
            indices_array = np.asarray(indices, dtype=np.int64)
            if indices_array.size < 4:
                continue
            test_size = max(1, int(round(indices_array.size * test_fraction)))
            test_local = np.sort(rng.choice(indices_array, size=test_size, replace=False))
            train_local = np.asarray([idx for idx in indices_array if idx not in set(test_local)], dtype=np.int64)
            splits.append((train_local, test_local))
    elif mode == "leave_film_out":
        all_indices = np.arange(len(rows), dtype=np.int64)
        for film_id, indices in sorted(film_to_indices.items()):
            test = np.asarray(indices, dtype=np.int64)
            train = np.asarray([idx for idx in all_indices if idx not in set(test)], dtype=np.int64)
            if train.size and test.size:
                splits.append((train, test))
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    return splits


def run_condition(
    matrix: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
    *,
    shuffle_rows: bool = False,
    shuffle_labels: bool = False,
    gaussian: bool = False,
) -> dict[str, Any]:
    film_ids = np.asarray([row["film_id"] for row in rows])
    times = np.asarray([float(row["time_start_seconds"]) for row in rows], dtype=np.float32)
    scores = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        rng = np.random.default_rng(seed + 10_000 + split_index)
        train_x = matrix[train_idx].copy()
        test_x = matrix[test_idx].copy()
        train_y = y[train_idx].copy()
        if shuffle_rows:
            train_x = train_x[rng.permutation(train_x.shape[0])]
            test_x = test_x[rng.permutation(test_x.shape[0])]
        if gaussian:
            train_x = rng.normal(size=train_x.shape).astype(np.float32)
            test_x = rng.normal(size=test_x.shape).astype(np.float32)
        if shuffle_labels:
            train_y = rng.permutation(train_y)
        pred = fit_predict(train_x, train_y, test_x, seed + split_index)
        scores.append(score(pred, y[test_idx], film_ids[test_idx], times[test_idx], seed + split_index))
    return aggregate(scores, baseline_maes)


def run_direct_prediction_condition(
    predictions: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
) -> dict[str, Any]:
    film_ids = np.asarray([row["film_id"] for row in rows])
    times = np.asarray([float(row["time_start_seconds"]) for row in rows], dtype=np.float32)
    scores = []
    for split_index, (_, test_idx) in enumerate(splits):
        scores.append(score(predictions[test_idx], y[test_idx], film_ids[test_idx], times[test_idx], seed + split_index))
    return aggregate(scores, baseline_maes)


def previous_window_predictions(y: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    predictions = np.empty_like(y, dtype=np.float32)
    film_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        film_to_indices[row["film_id"]].append(index)
    for indices in film_to_indices.values():
        ordered = sorted(indices, key=lambda idx: rows[idx]["time_start_seconds"])
        film_values = y[ordered]
        predictions[ordered[0]] = float(np.mean(film_values))
        for offset, index in enumerate(ordered[1:], start=1):
            predictions[index] = film_values[offset - 1]
    return predictions


def rolling_mean_predictions(y: np.ndarray, rows: list[dict[str, Any]], window: int = 5) -> np.ndarray:
    predictions = np.empty_like(y, dtype=np.float32)
    film_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        film_to_indices[row["film_id"]].append(index)
    for indices in film_to_indices.values():
        ordered = sorted(indices, key=lambda idx: rows[idx]["time_start_seconds"])
        film_values = y[ordered]
        film_mean = float(np.mean(film_values))
        for offset, index in enumerate(ordered):
            if offset == 0:
                predictions[index] = film_mean
            else:
                start = max(0, offset - window)
                predictions[index] = float(np.mean(film_values[start:offset]))
    return predictions


def run_for_mode(rows: list[dict[str, Any]], targets: list[str], mode: str, seed: int, test_fraction: float) -> dict[str, Any]:
    splits = make_splits(rows, mode, seed, test_fraction)
    time_x = time_features(rows)
    video_x = video_container_features(rows)
    film_x = film_id_features(rows)
    available_neuro = {
        "cortical_only": neuro_features(rows, "cortical_only"),
        "subcortical_only": neuro_features(rows, "subcortical_only"),
        "cortical_plus_subcortical": neuro_features(rows, "cortical_plus_subcortical"),
        "compact_cortical": neuro_features(rows, "compact_cortical"),
        "compact_neuro_affect": neuro_features(rows, "compact_neuro_affect"),
    }
    report = {
        "split_mode": mode,
        "split_count": len(splits),
        "targets": {},
        "conditions": {
            "mean_baseline": "train-fold mean target",
            "time_only": "time_start, normalized film phase, sin/cos phase, window_index",
            "video_container_metadata": "unavailable in annotation-only local bundle; constant missing baseline",
            "film_id": "one-hot film identifier; meaningful only for within-film splits, not leave-film-out generalization",
            "previous_window": "uses previous observed annotation within the same film; diagnostic autocorrelation baseline, not valid when prior labels are unavailable",
            "rolling_mean_5": "uses previous observed annotations within the same film; diagnostic autocorrelation baseline",
            "neuro": "skipped unless per-window neuro_feature_paths are present in manifest",
        },
    }
    if not splits:
        report["skipped"] = "Not enough films/windows for this split mode."
        return report
    for target in targets:
        y = np.asarray([float(row["targets"][target]) for row in rows], dtype=np.float32)
        baseline_scores = []
        film_ids = np.asarray([row["film_id"] for row in rows])
        times = np.asarray([float(row["time_start_seconds"]) for row in rows], dtype=np.float32)
        for split_index, (train_idx, test_idx) in enumerate(splits):
            pred = np.full(test_idx.size, np.mean(y[train_idx]), dtype=np.float32)
            baseline_scores.append(score(pred, y[test_idx], film_ids[test_idx], times[test_idx], seed + split_index))
        baseline_maes = [entry["mae"] for entry in baseline_scores]
        target_report: dict[str, Any] = {
            "target_distribution": summarize_target(y, rows),
            "autocorrelation": autocorrelation_by_film(y, rows),
            "mean_baseline": aggregate(baseline_scores, baseline_maes),
            "time_only": run_condition(time_x, y, rows, splits, baseline_maes, seed),
            "video_container_metadata": run_condition(video_x, y, rows, splits, baseline_maes, seed),
            "film_id": run_condition(film_x, y, rows, splits, baseline_maes, seed),
            "previous_window": run_direct_prediction_condition(
                previous_window_predictions(y, rows), y, rows, splits, baseline_maes, seed
            ),
            "rolling_mean_5": run_direct_prediction_condition(
                rolling_mean_predictions(y, rows, window=5), y, rows, splits, baseline_maes, seed
            ),
            "strict_controls": {
                "shuffled_labels_time_only": run_condition(time_x, y, rows, splits, baseline_maes, seed, shuffle_labels=True),
                "random_gaussian_time_feature_count": run_condition(time_x, y, rows, splits, baseline_maes, seed, gaussian=True),
            },
        }
        for name, matrix in available_neuro.items():
            if matrix is None:
                target_report[name] = {"skipped": "No per-window neuro features present in manifest."}
            else:
                target_report[name] = run_condition(matrix, y, rows, splits, baseline_maes, seed)
                target_report["strict_controls"][f"shuffled_{name}"] = run_condition(
                    matrix, y, rows, splits, baseline_maes, seed, shuffle_rows=True
                )
        report["targets"][target] = target_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="benchmarks/emofilm/emofilm_annotation_benchmark_smoke.json")
    parser.add_argument("--targets", default="Anxiety,Fear,Happiness,Sad,Surprise,Calm")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_rows(Path(args.manifest).expanduser().resolve())
    requested_targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    available_targets = set(rows[0]["targets"])
    targets = [target for target in requested_targets if target in available_targets]
    if not targets:
        raise ValueError(f"No requested targets available. requested={requested_targets}, available={sorted(available_targets)}")
    report = {
        "schema_version": "emofilm_annotation_temporal_benchmark_v1",
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "rows": len(rows),
        "films": sorted({row["film_id"] for row in rows}),
        "target_level": "1-second derivative aggregate annotation windows",
        "targets": targets,
        "benchmark_warning": "Annotation-only smoke benchmark; no TRIBE extraction or OpenLAV rows included.",
        "within_film_temporal": run_for_mode(rows, targets, "within_film_temporal", args.seed, args.test_fraction),
        "leave_film_out": run_for_mode(rows, targets, "leave_film_out", args.seed, args.test_fraction),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "targets": targets}, indent=2))


if __name__ == "__main__":
    main()
