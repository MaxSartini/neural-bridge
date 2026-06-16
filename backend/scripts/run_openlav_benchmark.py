"""Run grouped OpenLAV neuro-feature ablations on cached calibration IRs."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))


def correlation(predicted: np.ndarray, observed: np.ndarray) -> float | None:
    if (
        predicted.size < 2
        or np.isclose(np.std(predicted), 0.0)
        or np.isclose(np.std(observed), 0.0)
    ):
        return None
    value = float(np.corrcoef(predicted, observed)[0, 1])
    return value if math.isfinite(value) else None


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return output


def score(predicted: np.ndarray, observed: np.ndarray, seed: int) -> dict:
    errors = predicted - observed
    absolute = np.abs(errors)
    rng = random.Random(seed)
    bootstrap = [
        float(np.mean([absolute[rng.randrange(absolute.size)] for _ in range(absolute.size)]))
        for _ in range(1000)
    ]
    bootstrap.sort()
    total_variance = float(np.sum(np.square(observed - np.mean(observed))))
    residual_variance = float(np.sum(np.square(errors)))
    return {
        "n": int(observed.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "brier": float(np.mean(np.square(errors))),
        "pearson": correlation(predicted, observed),
        "spearman": correlation(ranks(predicted), ranks(observed)),
        "r2": 1.0 - residual_variance / total_variance if total_variance else None,
        "mae_bootstrap_95_ci": [bootstrap[24], bootstrap[974]],
        "prediction_summary": {
            "mean": float(np.mean(predicted)),
            "std": float(np.std(predicted)),
            "min": float(np.min(predicted)),
            "max": float(np.max(predicted)),
        },
    }


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int) -> np.ndarray:
    from catboost import CatBoostRegressor

    if train_x.shape[1] == 0 or np.all(np.std(train_x, axis=0) == 0):
        return np.full(test_x.shape[0], np.mean(train_y), dtype=np.float32)
    model = CatBoostRegressor(
        iterations=300,
        depth=5,
        learning_rate=0.04,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), 0.0, 1.0)


def fit_predict_regularized_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Small-sample guardrail baseline for high-dimensional neuro features."""
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    del seed
    if train_x.shape[1] == 0 or np.all(np.std(train_x, axis=0) == 0):
        return np.full(test_x.shape[0], np.mean(train_y), dtype=np.float32)
    model = make_pipeline(
        VarianceThreshold(threshold=1e-12),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 4, 16)),
    )
    model.fit(train_x, train_y)
    return np.clip(model.predict(test_x), 0.0, 1.0)


def aggregate_split_scores(split_scores: list[dict], baseline_maes: list[float]) -> dict:
    metrics = ("mae", "rmse", "brier", "pearson", "spearman", "r2")
    aggregate = {}
    for metric in metrics:
        values = [entry[metric] for entry in split_scores if entry.get(metric) is not None]
        aggregate[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    aggregate["paired_mae_delta_vs_mean_baseline"] = {
        "mean": float(np.mean([entry["mae"] - baseline for entry, baseline in zip(split_scores, baseline_maes)])),
        "negative_is_improvement": True,
    }
    aggregate["split_metrics"] = split_scores
    return aggregate


def split_local_control(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
    mode: str,
) -> dict:
    """Controls generated after train/test split to avoid row-identity leakage."""
    condition_scores = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        rng = np.random.default_rng(seed + 10_000 + split_index)
        train_x = feature_matrix[train_idx].copy()
        test_x = feature_matrix[test_idx].copy()
        if mode == "shuffle_rows":
            train_x = train_x[rng.permutation(train_x.shape[0])]
            test_x = test_x[rng.permutation(test_x.shape[0])]
        elif mode == "gaussian":
            train_x = rng.normal(size=train_x.shape).astype(np.float32)
            test_x = rng.normal(size=test_x.shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown split-local control mode: {mode}")
        predicted = fit_predict(train_x, y[train_idx], test_x, seed + split_index)
        condition_scores.append(score(predicted, y[test_idx], seed + split_index))
    return aggregate_split_scores(condition_scores, baseline_maes)


def cortical_subcortical_component_controls(
    cortical_matrix: np.ndarray,
    subcortical_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
) -> dict:
    """2x2 component controls for locating signal in cortical vs subcortical blocks."""
    outputs: dict[str, list[dict]] = {
        "real_cortical_real_subcortical": [],
        "real_cortical_shuffled_subcortical": [],
        "shuffled_cortical_real_subcortical": [],
        "shuffled_cortical_shuffled_subcortical": [],
        "random_gaussian_cortical_subcortical": [],
    }
    for split_index, (train_idx, test_idx) in enumerate(splits):
        rng = np.random.default_rng(seed + 30_000 + split_index)
        train_c = cortical_matrix[train_idx]
        test_c = cortical_matrix[test_idx]
        train_s = subcortical_matrix[train_idx]
        test_s = subcortical_matrix[test_idx]

        def shuffled_rows(matrix: np.ndarray) -> np.ndarray:
            return matrix[rng.permutation(matrix.shape[0])]

        def add_prediction(name: str, current_train_x: np.ndarray, current_test_x: np.ndarray) -> None:
            predicted = fit_predict(
                current_train_x,
                y[train_idx],
                current_test_x,
                seed + split_index,
            )
            outputs[name].append(score(predicted, y[test_idx], seed + split_index))

        add_prediction(
            "real_cortical_real_subcortical",
            np.concatenate([train_c, train_s], axis=1),
            np.concatenate([test_c, test_s], axis=1),
        )
        add_prediction(
            "real_cortical_shuffled_subcortical",
            np.concatenate([train_c, shuffled_rows(train_s)], axis=1),
            np.concatenate([test_c, shuffled_rows(test_s)], axis=1),
        )
        add_prediction(
            "shuffled_cortical_real_subcortical",
            np.concatenate([shuffled_rows(train_c), train_s], axis=1),
            np.concatenate([shuffled_rows(test_c), test_s], axis=1),
        )
        add_prediction(
            "shuffled_cortical_shuffled_subcortical",
            np.concatenate([shuffled_rows(train_c), shuffled_rows(train_s)], axis=1),
            np.concatenate([shuffled_rows(test_c), shuffled_rows(test_s)], axis=1),
        )
        add_prediction(
            "random_gaussian_cortical_subcortical",
            rng.normal(size=(train_idx.size, train_c.shape[1] + train_s.shape[1])).astype(np.float32),
            rng.normal(size=(test_idx.size, train_c.shape[1] + train_s.shape[1])).astype(np.float32),
        )
    return {
        name: aggregate_split_scores(split_scores, baseline_maes)
        for name, split_scores in outputs.items()
    }


def feature_family(name: str) -> str:
    if name.startswith("missingness::"):
        return "missingness"
    if name.startswith("quality::"):
        return "quality"
    if name.startswith("global::"):
        return "global"
    if name.startswith("subcortical::"):
        return "subcortical"
    if name.startswith("cortical::"):
        return "cortical"
    return "other"


def permutation_importance_report(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    top_n: int = 20,
) -> dict:
    """Held-out permutation importance using MAE increase after test-column shuffle."""
    if feature_matrix.shape[1] == 0:
        return {"skipped": "No matching features."}
    rng = np.random.default_rng(seed + 20_000)
    deltas = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    split_count = 0
    for split_index, (train_idx, test_idx) in enumerate(splits):
        from catboost import CatBoostRegressor

        train_x = feature_matrix[train_idx]
        test_x = feature_matrix[test_idx]
        if train_x.shape[1] == 0 or np.all(np.std(train_x, axis=0) == 0):
            continue
        model = CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=seed + split_index,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        model.fit(train_x, y[train_idx])
        baseline_pred = np.clip(model.predict(test_x), 0.0, 1.0)
        baseline_mae = float(np.mean(np.abs(baseline_pred - y[test_idx])))
        for feature_index in range(feature_matrix.shape[1]):
            permuted = test_x.copy()
            permuted[:, feature_index] = permuted[
                rng.permutation(permuted.shape[0]), feature_index
            ]
            permuted_pred = np.clip(model.predict(permuted), 0.0, 1.0)
            permuted_mae = float(np.mean(np.abs(permuted_pred - y[test_idx])))
            deltas[feature_index] += permuted_mae - baseline_mae
        split_count += 1
    if split_count == 0:
        return {"skipped": "No non-constant features after splitting."}
    deltas /= split_count
    order = np.argsort(-deltas)
    top_features = [
        {
            "feature": feature_names[index],
            "family": feature_family(feature_names[index]),
            "mean_mae_increase_when_permuted": float(deltas[index]),
        }
        for index in order[:top_n]
    ]
    family_totals: dict[str, float] = {}
    family_positive_totals: dict[str, float] = {}
    for name, delta in zip(feature_names, deltas):
        family = feature_family(name)
        family_totals[family] = family_totals.get(family, 0.0) + float(delta)
        family_positive_totals[family] = family_positive_totals.get(family, 0.0) + max(float(delta), 0.0)
    nuisance_positive = sum(
        family_positive_totals.get(family, 0.0)
        for family in ("missingness", "quality", "global")
    )
    neuro_positive = sum(
        family_positive_totals.get(family, 0.0)
        for family in ("cortical", "subcortical")
    )
    return {
        "method": "heldout_test_column_permutation_mae_delta",
        "split_count": split_count,
        "top_features": top_features,
        "family_mae_delta_sum": family_totals,
        "family_positive_mae_delta_sum": family_positive_totals,
        "nuisance_positive_importance": nuisance_positive,
        "neuro_positive_importance": neuro_positive,
        "missingness_or_duration_dominates": nuisance_positive > neuro_positive,
    }


def model_feature_importance_report(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    top_n: int = 20,
) -> dict:
    """CatBoost training-fold feature importance averaged across grouped splits."""
    if feature_matrix.shape[1] == 0:
        return {"skipped": "No matching features."}
    from catboost import CatBoostRegressor

    importances = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    split_count = 0
    for split_index, (train_idx, _) in enumerate(splits):
        train_x = feature_matrix[train_idx]
        if train_x.shape[1] == 0 or np.all(np.std(train_x, axis=0) == 0):
            continue
        model = CatBoostRegressor(
            iterations=300,
            depth=5,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=seed + split_index,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
        model.fit(train_x, y[train_idx])
        importances += np.asarray(model.get_feature_importance(), dtype=np.float64)
        split_count += 1
    if split_count == 0:
        return {"skipped": "No non-constant features after splitting."}
    importances /= split_count
    order = np.argsort(-importances)
    top_features = [
        {
            "feature": feature_names[index],
            "family": feature_family(feature_names[index]),
            "mean_catboost_importance": float(importances[index]),
        }
        for index in order[:top_n]
    ]
    family_totals: dict[str, float] = {}
    for name, value in zip(feature_names, importances):
        family = feature_family(name)
        family_totals[family] = family_totals.get(family, 0.0) + float(value)
    nuisance_total = sum(family_totals.get(family, 0.0) for family in ("missingness", "quality", "global"))
    neuro_total = sum(family_totals.get(family, 0.0) for family in ("cortical", "subcortical"))
    return {
        "method": "catboost_training_fold_prediction_values_change_mean",
        "split_count": split_count,
        "top_features": top_features,
        "family_importance_sum": family_totals,
        "nuisance_importance": nuisance_total,
        "neuro_importance": neuro_total,
        "missingness_or_duration_dominates": nuisance_total > neuro_total,
    }


def control_sanity_result(condition: dict, baseline: dict) -> dict:
    delta = condition.get("paired_mae_delta_vs_mean_baseline", {}).get("mean")
    baseline_delta = baseline.get("paired_mae_delta_vs_mean_baseline", {}).get("mean", 0.0)
    if delta is None:
        return {"status": "unknown", "reason": "condition did not produce paired MAE"}
    return {
        "status": "pass" if delta >= baseline_delta - 0.005 else "fail",
        "paired_mae_delta_vs_mean_baseline": delta,
        "expected": "shuffled/neutral controls should not materially beat the mean baseline",
    }


def effect_result(condition: dict) -> dict:
    delta = condition.get("paired_mae_delta_vs_mean_baseline", {}).get("mean")
    if delta is None:
        return {"status": "unknown", "reason": "condition did not produce paired MAE"}
    return {
        "status": "improves_over_mean_baseline" if delta < 0.0 else "does_not_improve_over_mean_baseline",
        "paired_mae_delta_vs_mean_baseline": delta,
        "negative_is_improvement": True,
    }


def feature_mask_summary(mask: np.ndarray, rows_count: int, rationale: str) -> dict:
    count = int(np.sum(mask))
    return {
        "count": count,
        "feature_to_row_ratio": float(count / rows_count),
        "rationale": rationale,
    }


def build_feature_masks(names: list[str]) -> dict[str, np.ndarray]:
    """Pre-registered interpretable masks to avoid small-n high-dimensional overfit."""
    names_array = np.asarray(names)

    def starts_with(prefixes: tuple[str, ...]) -> np.ndarray:
        return np.asarray([name.startswith(prefixes) for name in names_array])

    def contains_any(tokens: tuple[str, ...]) -> np.ndarray:
        lower_tokens = tuple(token.lower() for token in tokens)
        return np.asarray([
            any(token in name.lower() for token in lower_tokens)
            for name in names_array
        ])

    all_subcortical = starts_with(("subcortical::",))
    all_cortical = starts_with(("cortical::",))
    global_quality = starts_with(("global::", "quality::", "missingness::"))
    affective_subcortical = all_subcortical & contains_any((
        "amygdala",
        "accumbens",
        "pallidum",
        "hippocampus",
        "thalamus",
    ))
    core_subcortical = all_subcortical & contains_any((
        "amygdala",
        "accumbens",
        "pallidum",
    ))
    cortical_salience = all_cortical & contains_any((
        "g_and_s_cingul-ant",
        "g_and_s_cingul-mid-ant",
        "g_insular_short",
        "s_circular_insula_ant",
        "s_circular_insula_inf",
        "s_circular_insula_sup",
        "g_front_inf-orbital",
        "g_orbital",
        "s_orbital_lateral",
        "s_orbital_med-olfact",
        "s_orbital-h_shaped",
        "s_suborbital",
    ))
    core_cortical = all_cortical & contains_any((
        "g_and_s_cingul-ant",
        "g_and_s_cingul-mid-ant",
        "g_insular_short",
        "s_circular_insula_ant",
    ))
    return {
        "all_cortical": all_cortical,
        "all_subcortical": all_subcortical,
        "global_quality": global_quality,
        "affective_subcortical": affective_subcortical,
        "core_subcortical": core_subcortical,
        "cortical_salience": cortical_salience,
        "core_cortical": core_cortical,
        "compact_neuro_affect": global_quality | affective_subcortical | cortical_salience,
        "ultra_compact_neuro": global_quality | core_subcortical | core_cortical,
        "full_neuro": np.ones(names_array.shape[0], dtype=bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", default="benchmarks/openlav/openlav_benchmark.json")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--minimum-rows", type=int, default=40)
    parser.add_argument(
        "--run-mode",
        choices=("cortical_fast_default", "full_research", "subcortical_ablation"),
        default="cortical_fast_default",
        help="Default is cortical-only compact benchmarking; subcortical requires explicit research/ablation mode.",
    )
    args = parser.parse_args()

    from sklearn.model_selection import GroupShuffleSplit

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < args.minimum_rows:
        raise ValueError(
            f"OpenLAV benchmark requires at least {args.minimum_rows} cached stimuli, "
            f"got {len(rows)}"
        )
    model_contract_hashes = {
        row.get("cache_contract", {}).get("model_contract_hash") for row in rows
    }
    feature_name_hashes = {
        row.get("cache_contract", {}).get("feature_names_hash") for row in rows
    }
    if None in model_contract_hashes or len(model_contract_hashes) != 1:
        raise ValueError(f"Mixed or missing model contracts: {sorted(str(x) for x in model_contract_hashes)}")
    if None in feature_name_hashes or len(feature_name_hashes) != 1:
        raise ValueError(f"Mixed or missing feature contracts: {sorted(str(x) for x in feature_name_hashes)}")
    vectors = []
    names = None
    for row in rows:
        feature_path = Path(row["feature_path"])
        with np.load(feature_path) as bundle:
            vectors.append(np.asarray(bundle["calibration_feature_vector"], dtype=np.float32))
        current = json.loads(feature_path.with_suffix(".json").read_text(encoding="utf-8"))[
            "feature_contract"
        ]["feature_names"]
        if names is None:
            names = current
        elif names != current:
            raise ValueError(f"Feature contract mismatch in {feature_path}")
    x = np.stack(vectors)
    groups = np.asarray([row["group"] for row in rows])
    splits = list(
        GroupShuffleSplit(
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.seed,
        ).split(x, groups=groups)
    )
    masks = build_feature_masks(names)
    cortical = masks["all_cortical"]
    subcortical = masks["all_subcortical"]
    include_subcortical = args.run_mode in {"full_research", "subcortical_ablation"}
    permutations = {
        "shuffled_cortical": np.random.default_rng(args.seed).permutation(x.shape[0]),
        "shuffled_subcortical": np.random.default_rng(args.seed + 1).permutation(x.shape[0]),
    }
    conditions = {
        "cortical_only": x[:, cortical],
        "compact_global_quality": x[:, masks["global_quality"]],
        "compact_cortical_salience": x[:, masks["cortical_salience"]],
        "neutral_neuro": np.zeros_like(x),
        "shuffled_cortical": x.copy(),
    }
    if include_subcortical:
        conditions.update(
            {
                "subcortical_only": x[:, subcortical],
                "cortical_plus_subcortical_calibrated": x,
                "compact_subcortical_affective": x[:, masks["affective_subcortical"]],
                "compact_neuro_affect": x[:, masks["compact_neuro_affect"]],
                "ultra_compact_neuro": x[:, masks["ultra_compact_neuro"]],
                "shuffled_subcortical": x.copy(),
            }
        )
    conditions["shuffled_cortical"][:, cortical] = x[permutations["shuffled_cortical"]][:, cortical]
    if include_subcortical:
        conditions["shuffled_subcortical"][:, subcortical] = x[permutations["shuffled_subcortical"]][:, subcortical]
    condition_feature_names = {
        "cortical_only": list(np.asarray(names)[cortical]),
        "compact_global_quality": list(np.asarray(names)[masks["global_quality"]]),
        "compact_cortical_salience": list(np.asarray(names)[masks["cortical_salience"]]),
    }
    if include_subcortical:
        condition_feature_names.update(
            {
                "subcortical_only": list(np.asarray(names)[subcortical]),
                "cortical_plus_subcortical_calibrated": names,
                "compact_subcortical_affective": list(np.asarray(names)[masks["affective_subcortical"]]),
                "compact_neuro_affect": list(np.asarray(names)[masks["compact_neuro_affect"]]),
                "ultra_compact_neuro": list(np.asarray(names)[masks["ultra_compact_neuro"]]),
            }
        )
    strict_control_sources = {
        "split_local_shuffled_cortical": conditions["cortical_only"],
        "split_local_shuffled_compact_cortical_salience": conditions["compact_cortical_salience"],
        "random_gaussian_cortical": conditions["cortical_only"],
        "random_gaussian_compact_cortical_salience": conditions["compact_cortical_salience"],
    }
    if include_subcortical:
        strict_control_sources.update(
            {
                "split_local_shuffled_full_neuro": conditions["cortical_plus_subcortical_calibrated"],
                "split_local_shuffled_compact_neuro_affect": conditions["compact_neuro_affect"],
                "split_local_shuffled_ultra_compact_neuro": conditions["ultra_compact_neuro"],
                "split_local_shuffled_subcortical_affective": conditions["compact_subcortical_affective"],
                "random_gaussian_full_neuro": conditions["cortical_plus_subcortical_calibrated"],
                "random_gaussian_compact_neuro_affect": conditions["compact_neuro_affect"],
                "random_gaussian_ultra_compact_neuro": conditions["ultra_compact_neuro"],
                "random_gaussian_subcortical_affective": conditions["compact_subcortical_affective"],
            }
        )
    importance_conditions = {
        "cortical_only": conditions["cortical_only"],
        "compact_cortical_salience": conditions["compact_cortical_salience"],
        "compact_global_quality": conditions["compact_global_quality"],
    }
    if include_subcortical:
        importance_conditions.update(
            {
                "cortical_plus_subcortical_calibrated": conditions["cortical_plus_subcortical_calibrated"],
                "compact_neuro_affect": conditions["compact_neuro_affect"],
                "ultra_compact_neuro": conditions["ultra_compact_neuro"],
                "compact_subcortical_affective": conditions["compact_subcortical_affective"],
            }
        )
    report = {
        "schema_version": "openlav_grouped_ablation_v3",
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
        "contract_audit": {
            "single_model_contract": True,
            "model_contract_hash": next(iter(model_contract_hashes)),
            "single_feature_contract": True,
            "feature_names_hash": next(iter(feature_name_hashes)),
            "all_rows_have_cache_keys": all(
                bool(row.get("cache_contract", {}).get("cache_key")) for row in rows
            ),
            "all_rows_have_stimulus_hashes": all(
                bool(row.get("cache_contract", {}).get("stimulus_sha256")) for row in rows
            ),
        },
        "leakage_audit": {
            "group_overlap_by_split": [
                sorted(set(groups[train_idx]) & set(groups[test_idx]))
                for train_idx, test_idx in splits
            ],
            "duplicate_feature_vectors": int(
                x.shape[0] - np.unique(x, axis=0).shape[0]
            ),
        },
        "sanity_checks": {},
        "component_effects": {},
        "feature_contract": {
            "total": int(x.shape[1]),
            "cortical": int(np.sum(cortical)),
            "subcortical": int(np.sum(subcortical)),
            "global_quality_missingness": int(np.sum(masks["global_quality"])),
            "feature_to_row_ratio": float(x.shape[1] / len(rows)),
            "small_n_recommendation": (
                "Use ultra_compact_neuro or compact_neuro_affect as headline "
                "conditions until rows materially exceed feature count. Treat the "
                "854-feature full condition as exploratory on small samples."
            ),
        },
        "feature_sets": {
            "cortical_only": feature_mask_summary(
                cortical,
                len(rows),
                "All cortical parcel statistics; high-dimensional exploratory condition.",
            ),
            "compact_global_quality": feature_mask_summary(
                masks["global_quality"],
                len(rows),
                "Global BOLD summaries, missingness flags, and extraction quality controls.",
            ),
            "compact_cortical_salience": feature_mask_summary(
                masks["cortical_salience"],
                len(rows),
                "Anterior/mid cingulate, insula, and orbital cortical salience/valuation parcels.",
            ),
        },
        "model_families": {
            "primary": "catboost_regressor",
            "guardrail": "variance_threshold_standardized_ridge_cv",
            "interpretation": (
                "If CatBoost improves but the regularized linear guardrail does not, "
                "treat the lift as provisional until larger grouped holdouts confirm it."
            ),
        },
        "control_audit": {
            "legacy_shuffled_features": (
                "shuffled_cortical and shuffled_subcortical are global row "
                "permutations created before split; they preserve all non-shuffled "
                "feature families from the original row."
            ),
            "strict_split_local_controls": (
                "split_local_shuffled_* and random_gaussian_* are generated inside "
                "each train/test split, after the split is known. They destroy "
                "row/stimulus identity separately in train and test."
            ),
            "feature_selection_scaling_imputation": (
                "No labels are used for feature selection or imputation. Ridge scaling "
                "and variance filtering are fit inside train folds by sklearn Pipeline. "
                "CatBoost uses raw features and labels only in training folds."
            ),
            "shuffled_subcortical_metadata": (
                "legacy shuffled_subcortical keeps original cortical/global/quality/"
                "missingness features and only shuffles subcortical columns. Use "
                "split_local_shuffled_subcortical_affective and random_gaussian_* "
                "for stricter fake-neuro controls."
            ),
            "random_gaussian_controls": True,
            "permutation_importance": True,
            "component_controls": (
                "real/shuffled cortical crossed with real/shuffled subcortical "
                "uses only cortical and subcortical feature columns, excluding "
                "global, quality, and missingness features."
            ),
        },
        "targets": {},
        "unavailable_conditions": {
            "semantic_baseline": "Requires a separately frozen semantic feature model.",
            "llm_only_baseline": "Requires a separately executed deterministic LLM benchmark.",
            "persona_only": "OpenLAV contains participant ratings, not validated Neural Bridge personas.",
            "neuro_plus_persona": "Requires a separately validated participant interaction model.",
        },
    }
    if include_subcortical:
        report["feature_sets"].update(
            {
                "subcortical_only": feature_mask_summary(
                    subcortical,
                    len(rows),
                    "All subcortical parcel statistics; high-dimensional exploratory condition.",
                ),
                "cortical_plus_subcortical_calibrated": feature_mask_summary(
                    masks["full_neuro"],
                    len(rows),
                    "Full calibrated neuro IR; not recommended as headline evidence at small n.",
                ),
                "compact_subcortical_affective": feature_mask_summary(
                    masks["affective_subcortical"],
                    len(rows),
                    "Bilateral amygdala, accumbens, pallidum, hippocampus, and thalamus statistics.",
                ),
                "compact_neuro_affect": feature_mask_summary(
                    masks["compact_neuro_affect"],
                    len(rows),
                    "Union of global quality controls, affective subcortex, and cortical salience parcels.",
                ),
                "ultra_compact_neuro": feature_mask_summary(
                    masks["ultra_compact_neuro"],
                    len(rows),
                    "Small-n headline mask: global controls plus core amygdala/accumbens/pallidum and anterior cingulate/insula parcels.",
                ),
            }
        )
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
        baseline_maes = [entry["mae"] for entry in baseline_scores]
        axis_report = {
            "mean_baseline": aggregate_split_scores(baseline_scores, baseline_maes)
        }
        for condition, feature_matrix in conditions.items():
            if feature_matrix.shape[1] == 0:
                axis_report[condition] = {"skipped": "No matching features in IR contract."}
                continue
            condition_scores = []
            for split_index, (train_idx, test_idx) in enumerate(splits):
                predicted = fit_predict(
                    feature_matrix[train_idx],
                    y[train_idx],
                    feature_matrix[test_idx],
                    args.seed + split_index,
                )
                condition_scores.append(
                    score(predicted, y[test_idx], args.seed + split_index)
                )
            axis_report[condition] = aggregate_split_scores(condition_scores, baseline_maes)
        linear_report = {}
        for condition, feature_matrix in conditions.items():
            if feature_matrix.shape[1] == 0:
                linear_report[condition] = {"skipped": "No matching features in IR contract."}
                continue
            condition_scores = []
            for split_index, (train_idx, test_idx) in enumerate(splits):
                predicted = fit_predict_regularized_linear(
                    feature_matrix[train_idx],
                    y[train_idx],
                    feature_matrix[test_idx],
                    args.seed + split_index,
                )
                condition_scores.append(
                    score(predicted, y[test_idx], args.seed + split_index)
                )
            linear_report[condition] = aggregate_split_scores(condition_scores, baseline_maes)
        axis_report["regularized_linear_controls"] = linear_report
        strict_controls = {}
        for condition, feature_matrix in strict_control_sources.items():
            mode = "gaussian" if condition.startswith("random_gaussian") else "shuffle_rows"
            strict_controls[condition] = split_local_control(
                feature_matrix,
                y,
                splits,
                baseline_maes,
                args.seed,
                mode,
            )
        axis_report["strict_split_local_controls"] = strict_controls
        component_controls = {}
        if include_subcortical:
            component_controls = cortical_subcortical_component_controls(
                conditions["cortical_only"],
                conditions["subcortical_only"],
                y,
                splits,
                baseline_maes,
                args.seed,
            )
            axis_report["cortical_subcortical_component_controls"] = component_controls
        axis_report["permutation_importance"] = {
            condition: permutation_importance_report(
                feature_matrix,
                condition_feature_names[condition],
                y,
                splits,
                args.seed,
            )
            for condition, feature_matrix in importance_conditions.items()
        }
        axis_report["feature_importance"] = {
            condition: model_feature_importance_report(
                feature_matrix,
                condition_feature_names[condition],
                y,
                splits,
                args.seed,
            )
            for condition, feature_matrix in importance_conditions.items()
        }
        shuffled_label_scores = []
        for split_index, (train_idx, test_idx) in enumerate(splits):
            shuffled_train_y = np.random.default_rng(args.seed + split_index).permutation(
                y[train_idx]
            )
            predicted = fit_predict(
                x[train_idx],
                shuffled_train_y,
                x[test_idx],
                args.seed + split_index,
            )
            shuffled_label_scores.append(
                score(predicted, y[test_idx], args.seed + split_index)
            )
        axis_report["shuffled_labels"] = aggregate_split_scores(
            shuffled_label_scores, baseline_maes
        )
        report["targets"][axis] = axis_report
        sanity_checks = {
            "neutral_neuro": control_sanity_result(axis_report["neutral_neuro"], axis_report["mean_baseline"]),
            "shuffled_cortical": control_sanity_result(axis_report["shuffled_cortical"], axis_report["mean_baseline"]),
            "shuffled_labels": control_sanity_result(axis_report["shuffled_labels"], axis_report["mean_baseline"]),
        }
        for control_name, control_result in strict_controls.items():
            sanity_checks[control_name] = control_sanity_result(control_result, axis_report["mean_baseline"])
        if include_subcortical:
            sanity_checks.update(
                {
                    "shuffled_subcortical": control_sanity_result(axis_report["shuffled_subcortical"], axis_report["mean_baseline"]),
                    "real_cortical_shuffled_subcortical": control_sanity_result(component_controls["real_cortical_shuffled_subcortical"], axis_report["mean_baseline"]),
                    "shuffled_cortical_real_subcortical": control_sanity_result(component_controls["shuffled_cortical_real_subcortical"], axis_report["mean_baseline"]),
                    "shuffled_cortical_shuffled_subcortical": control_sanity_result(component_controls["shuffled_cortical_shuffled_subcortical"], axis_report["mean_baseline"]),
                    "random_gaussian_cortical_subcortical": control_sanity_result(component_controls["random_gaussian_cortical_subcortical"], axis_report["mean_baseline"]),
                }
            )
        report["sanity_checks"][axis] = sanity_checks
        component_effects = {
            "cortical_only": effect_result(axis_report["cortical_only"]),
            "compact_cortical_salience": effect_result(axis_report["compact_cortical_salience"]),
        }
        if include_subcortical:
            component_effects.update(
                {
                    "real_cortical_real_subcortical": effect_result(component_controls["real_cortical_real_subcortical"]),
                    "subcortical_only": effect_result(axis_report["subcortical_only"]),
                    "compact_subcortical_affective": effect_result(axis_report["compact_subcortical_affective"]),
                    "compact_neuro_affect": effect_result(axis_report["compact_neuro_affect"]),
                    "ultra_compact_neuro": effect_result(axis_report["ultra_compact_neuro"]),
                }
            )
        report["component_effects"][axis] = component_effects
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    leakage_report = {
        "schema_version": "openlav_leakage_audit_v1",
        "source_benchmark": str(output),
        "contract_audit": report["contract_audit"],
        "leakage_audit": report["leakage_audit"],
        "sanity_checks": report["sanity_checks"],
        "interpretation": (
            "Any failed shuffled or neutral control means benchmark lift is not "
            "interpretable as neuro signal until leakage, split bias, or sample-size "
            "effects are resolved."
        ),
    }
    output.with_suffix(".leakage.json").write_text(
        json.dumps(leakage_report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
