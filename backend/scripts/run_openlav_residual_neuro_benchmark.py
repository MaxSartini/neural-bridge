"""Residualized/additive OpenLAV benchmark for neuro-additive value.

This benchmark asks whether neuro features add signal beyond simple media and
extraction-quality confounds. All residualization is performed inside each
outer training fold; outer test labels are never used to fit baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

from run_openlav_benchmark import aggregate_split_scores, build_feature_masks, fit_predict, score  # noqa: E402
from run_openlav_nonneuro_baseline import make_matrix, media_features, quality_features  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_mean(condition: dict[str, Any], key: str) -> Any:
    value = condition.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def summarize_condition(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "mae": metric_mean(condition, "mae"),
        "rmse": metric_mean(condition, "rmse"),
        "pearson": metric_mean(condition, "pearson"),
        "spearman": metric_mean(condition, "spearman"),
        "delta": condition.get("paired_mae_delta_vs_mean_baseline", {}).get("mean"),
    }


def fit_predict_unbounded(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int) -> np.ndarray:
    from catboost import CatBoostRegressor

    if train_x.shape[1] == 0 or np.all(np.std(train_x, axis=0) == 0):
        return np.full(test_x.shape[0], np.mean(train_y), dtype=np.float32)
    model = CatBoostRegressor(
        iterations=220,
        depth=5,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    model.fit(train_x, train_y)
    return np.asarray(model.predict(test_x), dtype=np.float32)


def run_supervised_condition(
    matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
    *,
    shuffle_labels: bool = False,
) -> dict[str, Any]:
    scores = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        train_y = y[train_idx]
        if shuffle_labels:
            train_y = np.random.default_rng(seed + 50_000 + split_index).permutation(train_y)
        predicted = fit_predict(matrix[train_idx], train_y, matrix[test_idx], seed + split_index)
        scores.append(score(predicted, y[test_idx], seed + split_index))
    return aggregate_split_scores(scores, baseline_maes)


def run_gaussian_control(
    feature_count: int,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
) -> dict[str, Any]:
    scores = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        rng = np.random.default_rng(seed + 60_000 + split_index)
        train_x = rng.normal(size=(train_idx.size, feature_count)).astype(np.float32)
        test_x = rng.normal(size=(test_idx.size, feature_count)).astype(np.float32)
        predicted = fit_predict(train_x, y[train_idx], test_x, seed + split_index)
        scores.append(score(predicted, y[test_idx], seed + split_index))
    return aggregate_split_scores(scores, baseline_maes)


def train_oof_baseline(
    nonneuro_x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> np.ndarray:
    from sklearn.model_selection import GroupKFold, KFold

    train_groups = groups[train_idx]
    unique_groups = np.unique(train_groups)
    predictions = np.full(train_idx.size, np.nan, dtype=np.float32)
    if unique_groups.size >= 2:
        n_splits = min(5, unique_groups.size)
        iterator = GroupKFold(n_splits=n_splits).split(nonneuro_x[train_idx], y[train_idx], train_groups)
    else:
        n_splits = min(5, train_idx.size)
        iterator = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(nonneuro_x[train_idx])
    for inner_index, (inner_train_local, inner_val_local) in enumerate(iterator):
        predictions[inner_val_local] = fit_predict(
            nonneuro_x[train_idx][inner_train_local],
            y[train_idx][inner_train_local],
            nonneuro_x[train_idx][inner_val_local],
            seed + inner_index,
        )
    if np.isnan(predictions).any():
        predictions[np.isnan(predictions)] = np.mean(y[train_idx])
    return predictions


def residual_contexts(
    nonneuro_x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> list[dict[str, Any]]:
    contexts = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        train_baseline_oof = train_oof_baseline(nonneuro_x, y, train_idx, groups, seed + split_index)
        test_baseline = fit_predict(nonneuro_x[train_idx], y[train_idx], nonneuro_x[test_idx], seed + split_index)
        nonneuro_score = score(test_baseline, y[test_idx], seed + split_index)
        contexts.append(
            {
                "split_index": split_index,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_residual": y[train_idx] - train_baseline_oof,
                "test_residual": y[test_idx] - test_baseline,
                "test_baseline": test_baseline,
                "nonneuro_score": nonneuro_score,
                "nonneuro_mae": nonneuro_score["mae"],
                "zero_residual_mae": float(np.mean(np.abs(y[test_idx] - test_baseline))),
            }
        )
    return contexts


def residual_error_score(predicted: np.ndarray, observed: np.ndarray, seed: int) -> dict[str, Any]:
    errors = predicted - observed
    absolute = np.abs(errors)
    rng = np.random.default_rng(seed)
    bootstrap = [
        float(np.mean(absolute[rng.integers(0, absolute.size, absolute.size)]))
        for _ in range(1000)
    ]
    bootstrap.sort()
    return {
        "n": int(observed.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson": None if np.isclose(np.std(predicted), 0.0) or np.isclose(np.std(observed), 0.0) else float(np.corrcoef(predicted, observed)[0, 1]),
        "spearman": None,
        "mae_bootstrap_95_ci": [bootstrap[24], bootstrap[974]],
    }


def aggregate_residual(split_scores: list[dict[str, Any]], zero_maes: list[float]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in ("mae", "rmse", "pearson"):
        values = [entry[key] for entry in split_scores if entry.get(key) is not None]
        aggregate[key] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    aggregate["paired_mae_delta_vs_zero_residual_baseline"] = {
        "mean": float(np.mean([entry["mae"] - base for entry, base in zip(split_scores, zero_maes)])),
        "negative_is_improvement": True,
    }
    aggregate["split_metrics"] = split_scores
    return aggregate


def run_residual_condition(
    neuro_x: np.ndarray,
    y: np.ndarray,
    contexts: list[dict[str, Any]],
    seed: int,
    *,
    shuffle_neuro: bool = False,
    gaussian: bool = False,
) -> dict[str, Any]:
    final_scores = []
    residual_scores = []
    for context in contexts:
        split_index = int(context["split_index"])
        train_idx = context["train_idx"]
        test_idx = context["test_idx"]
        train_x = neuro_x[train_idx]
        test_x = neuro_x[test_idx]
        rng = np.random.default_rng(seed + 70_000 + split_index)
        if shuffle_neuro:
            train_x = train_x[rng.permutation(train_x.shape[0])]
            test_x = test_x[rng.permutation(test_x.shape[0])]
        if gaussian:
            train_x = rng.normal(size=train_x.shape).astype(np.float32)
            test_x = rng.normal(size=test_x.shape).astype(np.float32)
        predicted_residual = fit_predict_unbounded(
            train_x,
            context["train_residual"],
            test_x,
            seed + split_index,
        )
        final_prediction = np.clip(context["test_baseline"] + predicted_residual, 0.0, 1.0)
        final_scores.append(score(final_prediction, y[test_idx], seed + split_index))
        residual_scores.append(
            residual_error_score(predicted_residual, context["test_residual"], seed + split_index)
        )
    return {
        "final_prediction": aggregate_split_scores(
            final_scores,
            [float(context["nonneuro_mae"]) for context in contexts],
        ),
        "residual_prediction": aggregate_residual(
            residual_scores,
            [float(context["zero_residual_mae"]) for context in contexts],
        ),
        "nonneuro_baseline": aggregate_split_scores(
            [context["nonneuro_score"] for context in contexts],
            [float(context["nonneuro_mae"]) for context in contexts],
        ),
    }


def stability_summary(condition: dict[str, Any]) -> dict[str, Any]:
    split_metrics = condition.get("final_prediction", {}).get("split_metrics", [])
    deltas = [
        split["mae"] - base["mae"]
        for split, base in zip(split_metrics, condition.get("nonneuro_baseline", {}).get("split_metrics", []))
    ]
    if not deltas:
        return {"folds": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "folds": int(arr.size),
        "improved_folds": int(np.sum(arr < 0)),
        "worse_folds": int(np.sum(arr > 0)),
        "mean_delta": float(np.mean(arr)),
        "min_delta": float(np.min(arr)),
        "max_delta": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--videos-dir", default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos")
    parser.add_argument("--output", default="benchmarks/openlav/openlav_residual_neuro_first50.json")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--run-mode",
        choices=("cortical_fast_default", "full_research", "subcortical_ablation"),
        default="cortical_fast_default",
        help="Default is cortical-only residual/additive benchmarking; subcortical requires explicit research/ablation mode.",
    )
    args = parser.parse_args()

    from sklearn.model_selection import GroupShuffleSplit

    rows = [
        json.loads(line)
        for line in Path(args.manifest).expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    videos_dir = Path(args.videos_dir).expanduser().resolve()
    groups = np.asarray([row["group"] for row in rows])
    splits = list(
        GroupShuffleSplit(
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.seed,
        ).split(np.zeros((len(rows), 1), dtype=np.float32), groups=groups)
    )

    text_features: list[dict[str, float]] = []
    audio_features: list[dict[str, float]] = []
    video_features: list[dict[str, float]] = []
    neuro_vectors: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for row in rows:
        video_path = videos_dir / f"{row['stimulus_id']}.webm"
        video_dict, audio_dict = media_features(video_path)
        video_features.append(video_dict)
        audio_features.append(audio_dict)
        text_features.append(quality_features(Path(row["feature_path"]).parent))
        feature_path = Path(row["feature_path"])
        with np.load(feature_path) as bundle:
            neuro_vectors.append(np.asarray(bundle["calibration_feature_vector"], dtype=np.float32))
        current_names = load_json(feature_path.with_suffix(".json"))["feature_contract"]["feature_names"]
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError(f"Feature contract mismatch in {feature_path}")

    text_names = sorted({name for features in text_features for name in features})
    audio_names = sorted({name for features in audio_features for name in features})
    video_names = sorted({name for features in video_features for name in features})
    text_x = make_matrix(text_features, text_names)
    audio_x = make_matrix(audio_features, audio_names)
    video_x = make_matrix(video_features, video_names)
    tav_quality_x = np.concatenate([text_x, audio_x, video_x], axis=1)

    neuro = np.stack(neuro_vectors)
    masks = build_feature_masks(feature_names or [])
    include_subcortical = args.run_mode in {"full_research", "subcortical_ablation"}
    neuro_conditions = {
        "cortical_only": neuro[:, masks["all_cortical"]],
        "compact_cortical": neuro[:, masks["cortical_salience"]],
    }
    if include_subcortical:
        neuro_conditions.update(
            {
                "subcortical_only": neuro[:, masks["all_subcortical"]],
                "cortical_plus_subcortical": neuro[:, masks["all_cortical"] | masks["all_subcortical"]],
                "compact_neuro_affect": neuro[:, masks["compact_neuro_affect"]],
            }
        )
    nonneuro_conditions = {
        "video_container": video_x,
        "text_audio_video_quality": tav_quality_x,
        "combined_handcrafted_non_neuro": tav_quality_x,
    }
    additive_base = nonneuro_conditions["combined_handcrafted_non_neuro"]

    report: dict[str, Any] = {
        "schema_version": "openlav_residual_additive_benchmark_v1",
        "run_mode": args.run_mode,
        "subcortical_enabled": include_subcortical,
        "subcortical_policy": (
            "Disabled by default. Subcortical remains available for explicit full_research/subcortical_ablation, "
            "but current evidence does not justify default compute."
            if not include_subcortical
            else "Explicit research/ablation mode with subcortical enabled."
        ),
        "rows": len(rows),
        "groups": int(np.unique(groups).size),
        "split": {
            "method": "repeated_source_url_group_shuffle_holdout",
            "seed": args.seed,
            "n_splits": args.n_splits,
            "test_size": args.test_size,
        },
        "contract": {
            "residualization": (
                "For each outer split, non-neuro test predictions are fit on outer-train only. "
                "Train residuals use inner grouped OOF non-neuro predictions within outer-train only."
            ),
            "no_full_dataset_fit": True,
            "controls": [
                "nonneuro_plus_shuffled_cortical",
                "nonneuro_plus_shuffled_subcortical",
                "nonneuro_plus_shuffled_all_neuro",
                "random_gaussian_features",
                "shuffled_labels",
            ],
        },
        "feature_counts": {
            "video_container": int(video_x.shape[1]),
            "text_audio_video_quality": int(tav_quality_x.shape[1]),
            "combined_handcrafted_non_neuro": int(tav_quality_x.shape[1]),
            **{name: int(matrix.shape[1]) for name, matrix in neuro_conditions.items()},
        },
        "targets": {},
    }

    for axis in sorted(rows[0]["targets"]):
        y = np.asarray([row["targets"][axis] for row in rows], dtype=np.float32)
        baseline_scores = []
        for split_index, (train_idx, test_idx) in enumerate(splits):
            baseline_scores.append(
                score(
                    np.full(test_idx.size, np.mean(y[train_idx])),
                    y[test_idx],
                    args.seed + split_index,
                )
            )
        mean_maes = [entry["mae"] for entry in baseline_scores]

        target: dict[str, Any] = {
            "mean_baseline": aggregate_split_scores(baseline_scores, mean_maes),
            "nonneuro_baselines": {},
            "neuro_only": {},
            "additive_models": {},
            "residualized_models": {},
            "strict_controls": {},
            "stability": {},
        }
        for name, matrix in nonneuro_conditions.items():
            target["nonneuro_baselines"][name] = run_supervised_condition(
                matrix, y, splits, mean_maes, args.seed
            )
        for name, matrix in neuro_conditions.items():
            target["neuro_only"][name] = run_supervised_condition(
                matrix, y, splits, mean_maes, args.seed
            )
        for name, matrix in neuro_conditions.items():
            additive_matrix = np.concatenate([additive_base, matrix], axis=1)
            target["additive_models"][f"nonneuro_plus_{name}"] = run_supervised_condition(
                additive_matrix, y, splits, mean_maes, args.seed
            )
        target["strict_controls"]["shuffled_labels_full_neuro"] = run_supervised_condition(
            neuro_conditions["cortical_plus_subcortical"] if include_subcortical else neuro_conditions["cortical_only"],
            y,
            splits,
            mean_maes,
            args.seed,
            shuffle_labels=True,
        )
        target["strict_controls"]["random_gaussian_full_neuro"] = run_gaussian_control(
            (
                neuro_conditions["cortical_plus_subcortical"]
                if include_subcortical
                else neuro_conditions["cortical_only"]
            ).shape[1],
            y,
            splits,
            mean_maes,
            args.seed,
        )

        residual_context_by_nonneuro = {
            name: residual_contexts(matrix, y, groups, splits, args.seed)
            for name, matrix in nonneuro_conditions.items()
        }
        residual_base = "combined_handcrafted_non_neuro"
        contexts = residual_context_by_nonneuro[residual_base]
        for name, matrix in neuro_conditions.items():
            condition = run_residual_condition(matrix, y, contexts, args.seed)
            target["residualized_models"][f"{residual_base}_residual_plus_{name}"] = condition
            target["stability"][f"{residual_base}_residual_plus_{name}"] = stability_summary(condition)
        target["strict_controls"]["residual_plus_shuffled_cortical"] = run_residual_condition(
            neuro_conditions["cortical_only"], y, contexts, args.seed, shuffle_neuro=True
        )
        if include_subcortical:
            target["strict_controls"]["residual_plus_shuffled_subcortical"] = run_residual_condition(
                neuro_conditions["subcortical_only"], y, contexts, args.seed, shuffle_neuro=True
            )
            target["strict_controls"]["residual_plus_shuffled_all_neuro"] = run_residual_condition(
                neuro_conditions["cortical_plus_subcortical"], y, contexts, args.seed, shuffle_neuro=True
            )
            target["strict_controls"]["residual_plus_random_gaussian_all_neuro"] = run_residual_condition(
                neuro_conditions["cortical_plus_subcortical"], y, contexts, args.seed, gaussian=True
            )
        else:
            target["strict_controls"]["residual_plus_random_gaussian_cortical"] = run_residual_condition(
                neuro_conditions["cortical_only"], y, contexts, args.seed, gaussian=True
            )
        for control_name, condition in target["strict_controls"].items():
            if control_name.startswith("residual_plus_"):
                target["stability"][control_name] = stability_summary(condition)
        report["targets"][axis] = target

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
