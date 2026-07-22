"""VEATIC-only event discovery preregistration.

The manifest is built from substrate metadata and feature-row identities only.  It cannot
inspect label values, so target, metric, and promotion choices are frozen before any
benchmark-test labels are opened.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .contracts import FeatureRows, LabelRows, SubstrateIdentity, TargetSpec
from .evidence import digest_json
from .protocol import assert_row_alignment, future_target_values, target_support_mask


@dataclass(frozen=True)
class EventTargetHypothesis:
    """One separately confirmable VEATIC event definition."""

    name: str
    label: Literal["arousal", "valence"]
    start_seconds: float
    stop_seconds: float
    transform: Literal["positive", "absolute"]
    train_quantile: float
    role: Literal["primary", "diagnostic"] = "primary"

    def horizon_rows(self, row_hz: float) -> tuple[int, ...]:
        if not 0.0 < self.start_seconds <= self.stop_seconds:
            raise ValueError("target seconds must be positive and increasing")
        if not 0.0 < self.train_quantile < 1.0:
            raise ValueError("target quantile must be between zero and one")
        start = math.ceil(self.start_seconds * row_hz)
        stop = math.floor(self.stop_seconds * row_hz)
        if start <= 0 or stop < start:
            raise ValueError("target interval has no rows at the canonical cadence")
        return tuple(range(start, stop + 1))


TARGET_QUANTILE_GRID = (0.95, 0.925, 0.90, 0.875, 0.85, 0.80)
_SPLIT_SEED = 20_260_722
_BENCHMARK_TRAIN_FRACTION = 0.70


def _quantiles(values: np.ndarray) -> dict[str, float]:
    points = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        str(point): float(value)
        for point, value in zip(points, np.quantile(values, points), strict=True)
    }


def _profile(identity: SubstrateIdentity, features: FeatureRows) -> dict[str, Any]:
    identity.validate()
    features.validate()
    videos = features.video_id.astype(str)
    if set(videos) != set(identity.video_ids) or len(videos) != identity.row_count:
        raise ValueError("metadata profile requires the complete canonical substrate")
    rows_per_video = np.asarray([np.sum(videos == video) for video in identity.video_ids])
    eligible_per_video = np.asarray(
        [np.sum((videos == video) & features.quality_eligible) for video in identity.video_ids]
    )
    durations = np.asarray(
        [
            float(np.max(features.time_seconds[videos == video]) + 1.0 / identity.row_hz)
            for video in identity.video_ids
        ]
    )
    return {
        "video_count": len(identity.video_ids),
        "row_count": identity.row_count,
        "row_hz": identity.row_hz,
        "eligible_rows": int(features.quality_eligible.sum()),
        "excluded_rows": int((~features.quality_eligible).sum()),
        "rows_per_video_quantiles": _quantiles(rows_per_video),
        "eligible_rows_per_video_quantiles": _quantiles(eligible_per_video),
        "duration_seconds_quantiles": _quantiles(durations),
    }


def _official_temporal_split(
    identity: SubstrateIdentity,
    features: FeatureRows,
) -> dict[str, Any]:
    """Recalculate VEATIC's first-70/last-30 protocol over usable rows in every video."""

    videos = features.video_id.astype(str)
    boundaries = []
    train_rows = 0
    test_rows = 0
    for video in identity.video_ids:
        eligible_rows = np.sort(
            features.row_index[(videos == video) & features.quality_eligible]
        )
        cutoff = int(len(eligible_rows) * _BENCHMARK_TRAIN_FRACTION)
        if cutoff <= 0 or cutoff >= len(eligible_rows):
            raise ValueError(f"video {video} cannot support an eligible 70/30 split")
        train_rows += cutoff
        test_rows += len(eligible_rows) - cutoff
        boundaries.append(
            {
                "video_id": video,
                "eligible_rows": len(eligible_rows),
                "benchmark_train_rows": cutoff,
                "benchmark_test_rows": len(eligible_rows) - cutoff,
                "last_train_row_index": int(eligible_rows[cutoff - 1]),
                "first_test_row_index": int(eligible_rows[cutoff]),
            }
        )

    def hashed(video: str) -> str:
        return digest_json({"seed": _SPLIT_SEED, "video_id": video})

    ordered_videos = sorted(identity.video_ids, key=hashed)
    split = {
        "seed": _SPLIT_SEED,
        "protocol": "veatic_first_70_percent_last_30_percent_per_video",
        "quality_rule": "exclude_black_or_high_duplicate_before_calculating_each_video_split",
        "requested_train_fraction": _BENCHMARK_TRAIN_FRACTION,
        "actual_train_fraction": train_rows / (train_rows + test_rows),
        "video_ids": list(identity.video_ids),
        "benchmark_train_rows": train_rows,
        "benchmark_test_rows": test_rows,
        "inner_grouped_video_folds": [ordered_videos[index::5] for index in range(5)],
        "benchmark_reporting_video_slices": [ordered_videos[index::5] for index in range(5)],
        "per_video_boundaries": boundaries,
    }
    split["split_sha256"] = digest_json(split)
    return split


def benchmark_partition_mask(
    features: FeatureRows,
    split: Mapping[str, Any],
    partition: Literal["train", "test"],
) -> np.ndarray:
    """Select usable rows owned by one preregistered temporal benchmark partition."""

    features.validate()
    if partition not in {"train", "test"}:
        raise ValueError(f"unsupported benchmark partition: {partition}")
    boundary_by_video = {
        str(row["video_id"]): row for row in split["per_video_boundaries"]
    }
    videos = features.video_id.astype(str)
    mask = np.zeros(len(videos), dtype=bool)
    for video in np.unique(videos):
        boundary = boundary_by_video.get(video)
        if boundary is None:
            raise ValueError(f"split has no boundary for video {video}")
        owned = videos == video
        if partition == "train":
            owned &= features.row_index <= int(boundary["last_train_row_index"])
        else:
            owned &= features.row_index >= int(boundary["first_test_row_index"])
        mask |= owned & features.quality_eligible
    return mask


def build_event_preregistration(
    identity: SubstrateIdentity,
    features: FeatureRows,
) -> dict[str, Any]:
    """Freeze the claim-capable event search without accepting label values."""

    profile = _profile(identity, features)
    split = _official_temporal_split(identity, features)
    maximum_horizon_seconds = min(
        10.0,
        math.floor(profile["duration_seconds_quantiles"]["0.1"] * 0.2 * identity.row_hz)
        / identity.row_hz,
    )

    manifest: dict[str, Any] = {
        "schema": "veatic21_event_preregistration_v12",
        "programme": "veatic-2.1",
        "claim": "within-video future-tail ranking of arousal-increase events",
        "label_values_accessed": False,
        "substrate": {
            "profile": profile,
            "row_plan_sha256": identity.row_plan_sha256,
            "source_tree_sha256": identity.source_tree_sha256,
            "encoder_model_sha256": identity.encoder_model_sha256,
            "vjepa_artifact_id": identity.vjepa_artifact_id,
            "vjepa_sha256_tree": identity.vjepa_sha256_tree,
            "tribe_artifact_id": identity.tribe_artifact_id,
            "tribe_sha256_tree": identity.tribe_sha256_tree,
        },
        "label_access": {
            "benchmark_train_labels": "eligible_first_70_percent_rows_from_all_124_videos",
            "benchmark_test_labels": (
                "eligible_last_30_percent_rows_sealed_until_predictions_and_models_are_frozen"
            ),
            "production_refit": "fresh_pca_and_head_from_scratch_on_all_124_after_benchmark",
            "production_refit_is_claim_evidence": False,
        },
        "split": split,
        "target_calibration": {
            "source": "benchmark_train_arousal_labels_only",
            "cadence_seconds": 1.0 / identity.row_hz,
            "maximum_horizon_seconds": maximum_horizon_seconds,
            "movement_curve": "median_absolute_change_at_each_canonical_lag",
            "primary_horizon_stops": (
                "earliest_lags_reaching_25_50_75_percent_of_benchmark_train_movement_scale"
            ),
            "window_grid": (
                "all_start_stop_pairs_across_first_row_25_50_75_percent_movement_lags_"
                "and_maximum_supported_lag"
            ),
            "transform": "positive_future_maximum_increase",
            "quantile_grid": list(TARGET_QUANTILE_GRID),
            "quantile_rule": (
                "retain_every_rate_supported_in_all_inner_video_folds_then_select_by_"
                "supervised_incremental_validation"
            ),
            "threshold_refit_scope": "current_training_partition_only",
            "benchmark_test_labels_used_for_calibration": False,
            "final_target_policy": "freeze_one_train_cv_winner_before_benchmark_test",
        },
        "evidence_protocols": {
            "official_temporal_benchmark": {
                "train_fraction_requested": _BENCHMARK_TRAIN_FRACTION,
                "train_videos": len(identity.video_ids),
                "test_videos": len(identity.video_ids),
                "train_rows": split["benchmark_train_rows"],
                "test_rows": split["benchmark_test_rows"],
                "inner_grouped_video_folds": 5,
                "test_reporting_video_slices": 5,
                "promotion_scope": "primary",
                "test_labels_closed_until_prediction_seal": True,
                "claim_boundary": "same_video_future_tail_not_unseen_video_generalization",
            },
            "production_refit": {
                "video_count": len(identity.video_ids),
                "configuration_frozen_from_benchmark": True,
                "refit_pca_scalers_thresholds_and_head_from_scratch": True,
                "produces_benchmark_metrics": False,
            },
            "blocked_temporal": {
                "train_fraction": 0.60,
                "gap_fraction": 0.10,
                "test_fraction": 0.30,
                "promotion_scope": "separate_stricter_temporal_sensitivity",
                "never_pooled_with_official_temporal_benchmark": True,
            },
        },
        "representations": {
            "encoder_stack": "vjepa_2p1_inside_tribe_v2",
            "canonical_neural_bridge_input": "tribe_cortical",
            "cached_vjepa_role": "expensive_reusable_intermediate_for_tribe_v2",
            "internal_ablation_views": ["vjepa_temporal_mean", "tribe_grouped_mean"],
            "independent_source_fusion": False,
            "raw_current_row": "required_linear_diagnostic_not_presumed_winner",
            "pca": {
                "benchmark_fit_scope": "eligible_first_70_percent_rows_across_all_124_videos",
                "production_refit_scope": (
                    "all_eligible_rows_across_all_124_after_benchmark_configuration_freeze"
                ),
                "variance_targets": [0.80, 0.90, 0.95, 0.99],
                "fixed_width_candidates": [64, 128, 256, 512],
                "maximum_components": 512,
                "basis_fit": "fit_one_512_component_basis_per_exact_fold_and_source",
                "scaler": "featurewise_standard_scaler_fit_on_exact_training_rows",
                "solver": "deterministic_incremental_pca",
                "batch_rows": "2_x_maximum_components_balanced_without_short_final_batch",
                "actual_width": "smallest_train_fitted_width_reaching_variance_target",
                "candidate_width_union": (
                    "fixed_widths_plus_smallest_available_prefix_reaching_each_variance_target"
                ),
                "winner": "veatic_inner_validation_not_explained_variance_alone",
                "reuse": "fit_maximum_once_per_source_fold_then_reuse_prefixes",
                "cache_reuse_across": ["targets", "quantiles", "heads", "seeds"],
                "cache_key": (
                    "substrate_quality_split_fold_train_rows_source_transform_scaler_"
                    "pca_solver_code"
                ),
                "invalidation": "any_cache_key_change_requires_a_fresh_fit",
            },
            "supervised_projection": {
                "candidate_status": "unselected_until_veatic_inner_validation",
                "candidate": "end_to_end_learned_bottleneck_from_tribe_cortical",
                "fit_scope": "exact_training_fold_with_labels",
                "validation_role": "compare_against_fixed_pca_on_identical_folds_and_seeds",
                "benchmark_test_labels_used_for_fit": False,
                "pca_is_baseline_not_ceiling": True,
            },
            "causal_context_seconds": [0.5, 1.0, 2.0, 3.0, 5.0],
            "temporal_forms": [
                "current",
                "causal_mean",
                "current_minus_causal_mean",
                "causal_first_difference",
                "ordered_sequence",
            ],
            "future_features_forbidden": True,
            "video_boundary_crossing_forbidden": True,
        },
        "heads": {
            "candidate_status": "all_unselected_until_veatic_inner_validation",
            "linear_internal_ablation": ["current", "causal_delta"],
            "label_assisted_discovery": [
                "frozen_ar_plus_causal_temporal_residual",
                "frozen_ar_plus_gated_multiscale_temporal_residual",
            ],
            "video_only_baselines": [
                "current_row_mlp_control",
                "causal_temporal_convolution",
                "gated_multiscale_temporal_convolution",
            ],
            "training_outputs": ["train_thresholded_arousal_spike_logits"],
            "ar_contract": {
                "ar_fit_scope": "same_training_partition",
                "reuse": "fit_once_per_target_and_fold_then_freeze_for_every_matched_lane",
                "role": "comparator_and_label_assisted_residual_candidate_only",
                "forbidden_at_client_inference": True,
                "safe_residual_candidate": {
                    "form": "frozen_ar_plus_bounded_learned_video_residual",
                    "gate_and_bound_values": "calculate_and_select_from_veatic_only",
                    "checkpoint_rule": (
                        "per_fold_seed_use_residual_only_if_inner_validation_delta_vs_ar_is_"
                        "positive_otherwise_emit_unchanged_frozen_ar"
                    ),
                    "fallback_scope": "whole_fold_seed_prediction_not_row_outcome_oracle",
                    "benchmark_labels_used_for_fallback": False,
                    "fallback_is_not_residual_win_evidence": True,
                },
            },
            "spike_contract": {
                "again_hypothesis": "frozen_ar_predictions_feed_gated_temporal_residual",
                "veatic_candidate_requires_fresh_training": True,
                "veatic_evidence_selects_or_rejects": True,
                "objective": (
                    "crack_label_assisted_incremental_video_value_over_fresh_veatic_ar_in_"
                    "inner_validation"
                ),
                "video_only_does_not_have_to_beat_ar_during_spike_discovery": True,
                "sealed_benchmark_timing": (
                    "open_future_tails_once_only_after_the_spike_winner_is_frozen"
                ),
            },
            "transfer_boundary": (
                "no_again_head_weights_dimensions_gates_inputs_losses_or_numeric_configuration"
            ),
        },
        "training": {
            "comparison_seed_panel": [20_260_722, 20_260_723, 20_260_724],
            "comparison_seed_policy": (
                "every_candidate_uses_identical_folds_initialization_ids_and_sampler_orders"
            ),
            "stability_seed_panel": list(range(20_260_801, 20_260_810)),
            "stability_seed_policy": (
                "fixed_before_training_opened_after_shortlist_before_controls_and_winner_freeze"
            ),
            "freshness_policy": "refit_weights_and_pca_without_changing_declared_seed_ids",
            "checkpoint_ensembles": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            "ensemble_rule": "equal_weight_average_no_member_or_weight_selection",
            "numeric_configuration_source": "veatic_2p1_only",
            "numeric_configuration_policy": (
                "freeze_each_experiment_matrix_only_after_the_preceding_veatic_gate_completes"
            ),
            "again_numeric_configuration_reuse": False,
            "checkpoint_metric": "inner_average_precision_skill_delta_vs_frozen_ar",
            "checkpoint_eligibility": "every_completed_validation_from_epoch_1",
            "checkpoint_selection": "best_frozen_stage_metric_earliest_checkpoint_breaks_exact_tie",
            "minimum_epochs_before_termination": 50,
            "last_checkpoint_preference": False,
            "termination": (
                "after_epoch_50_continue_while_validation_can_improve_until_veatic_calibrated_"
                "plateau_and_optimizer_convergence_with_generous_runaway_compute_safety_"
                "ceiling_only_models_may_run_400_plus_epochs"
            ),
            "benchmark_labels_for_checkpointing": False,
            "nonconvergence": "candidate_failure",
        },
        "controls": [
            "target_specific_frozen_ar",
            "within_video_sequence_shuffled",
            "matched_random",
            "causal_prefix_video_mean",
            "diagnostics_only",
            "within_video_circular_label_permutation",
        ],
        "metrics": {
            "primary": "pooled_average_precision_skill",
            "reported": [
                "pooled_pr_auc",
                "analytic_chance_pr_auc",
                "pooled_average_precision_skill",
                "per_video_pr_auc_defined_only",
                "top_1_5_10_percent_event_recall",
                "brier_score",
            ],
            "uncertainty": "paired_video_cluster_bootstrap_95_percent_ci",
            "zero_event_videos": "retained_in_pooled_metric_per_video_metric_undefined",
        },
        "selection": {
            "stage_order": [
                "target_and_fresh_ar_discovery",
                "representation_and_pca_experiments",
                "model_and_training_experiments",
                "fixed_fold_and_seed_confirmation",
                "matched_controls_and_no_harm",
                "winner_selection_and_freeze",
                "single_sealed_benchmark_confirmation",
            ],
            "primary_key": "mean_inner_average_precision_skill_delta_vs_frozen_ar",
            "tie_break": [
                "lower_confidence_bound_descending",
                "median_delta_descending",
                "parameter_count_ascending",
                "candidate_name_ascending",
            ],
            "benchmark_test_scores_used_for_selection": False,
        },
        "promotion_gates": {
            "all_integrity_and_leakage_gates_pass": True,
            "paired_cluster_bootstrap_lower_ci_vs_frozen_ar_above_zero": True,
            "paired_cluster_bootstrap_lower_ci_vs_strongest_matched_control_above_zero": True,
            "positive_benchmark_test_slice_delta_minimum": "4_of_5",
            "seed_group_sign_test": "one_sided_exact_p_below_0p05",
            "single_spike_target_frozen_before_benchmark_test": True,
            "spike_configuration_frozen_before_benchmark_test": True,
            "ensemble_uplift_median_above_zero": True,
            "secondary_slice_no_harm_margin": (
                "freeze_from_veatic_train_video_cluster_uncertainty_before_benchmark"
            ),
            "blocked_temporal_is_separate": True,
            "exact_value_or_continuous_claim": False,
        },
        "failure_rules": [
            "fail_nonconverged_candidate",
            "fail_candidate_missing_any_paired_comparison_seed",
            "fail_any_train_test_fit_overlap",
            "fail_any_future_feature_or_video_boundary_crossing",
            "fail_if_outer_labels_open_before_prediction_seal",
            "fail_any_per_row_switch_using_observed_target_or_error",
            "fail_if_real_lane_does_not_beat_frozen_ar_and_every_matched_control",
            "fail_if_effect_is_fold_or_seed_concentrated",
            "do_not_repair_or_retune_after_benchmark_test_results",
        ],
        "again_boundary": {
            "allowed": "hypotheses_failure_modes_and_semantically_neutral_rigor_only",
            "forbidden": (
                "trained_heads_checkpoints_pca_scalers_thresholds_targets_windows_gates_or_scores"
            ),
        },
    }
    manifest["preregistration_sha256"] = digest_json(manifest)
    return manifest


def _movement_curve(
    features: FeatureRows,
    labels: LabelRows,
    max_lag_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    videos = labels.video_id.astype(str)
    medians = []
    for lag in range(1, max_lag_rows + 1):
        changes = []
        for video in np.unique(videos):
            mask = videos == video
            order = np.argsort(labels.row_index[mask])
            row_index = labels.row_index[mask][order]
            arousal = labels.arousal[mask][order].astype(np.float64)
            lookup = {int(row): value for row, value in zip(row_index, arousal, strict=True)}
            changes.extend(
                abs(lookup[int(row) + lag] - value)
                for row, value in zip(row_index, arousal, strict=True)
                if (int(row) + lag) in lookup
            )
        medians.append(float(np.median(changes)))
    return np.arange(1, max_lag_rows + 1, dtype=np.int64), np.asarray(medians)


def _movement_milestones(lags: np.ndarray, movement: np.ndarray) -> tuple[int, int, int]:
    maximum = float(np.max(movement))
    if maximum <= 0.0:
        raise ValueError("benchmark-train arousal labels contain no temporal movement")
    chosen: list[int] = []
    for fraction in (0.25, 0.50, 0.75):
        ranked = np.argsort(np.abs(movement / maximum - fraction), kind="stable")
        choice = next(int(lags[index]) for index in ranked if int(lags[index]) not in chosen)
        chosen.append(choice)
    ordered = sorted(chosen)
    return ordered[0], ordered[1], ordered[2]


def _supported_event_quantiles(
    values: np.ndarray,
    support: np.ndarray,
    video_id: np.ndarray,
    benchmark_train_folds: list[list[str]],
) -> list[tuple[float, dict[str, Any]]]:
    owned = values[support]
    supported = []
    for quantile in TARGET_QUANTILE_GRID:
        threshold = float(np.quantile(owned, quantile, method="linear"))
        event = np.isfinite(values) & (values >= threshold)
        fold_support = []
        valid = True
        for fold in benchmark_train_folds:
            fold_mask = support & np.isin(video_id.astype(str), fold)
            positive_videos = len(set(video_id[fold_mask & event].astype(str)))
            minimum_videos = max(2, math.ceil(len(fold) * 0.25))
            fold_valid = bool(event[fold_mask].any() and (~event[fold_mask]).any())
            fold_valid = fold_valid and positive_videos >= minimum_videos
            valid &= fold_valid
            fold_support.append(
                {
                    "rows": int(fold_mask.sum()),
                    "positive_rows": int(event[fold_mask].sum()),
                    "positive_videos": positive_videos,
                    "minimum_positive_videos": minimum_videos,
                    "pass": fold_valid,
                }
            )
        if valid:
            supported.append(
                (
                    quantile,
                    {
                        "benchmark_train_threshold_diagnostic_only": threshold,
                        "benchmark_train_prevalence": float(np.mean(event[support])),
                        "positive_videos": len(set(video_id[support & event].astype(str))),
                        "fold_support": fold_support,
                    },
                )
            )
    if not supported:
        raise ValueError("no event quantile has sufficient grouped-video support")
    return supported


def calibrate_event_preregistration(
    preregistration: dict[str, Any],
    features: FeatureRows,
    labels: LabelRows,
) -> dict[str, Any]:
    """Use every benchmark-train label while proving benchmark-test labels stay closed."""

    expected_digest = preregistration.get("preregistration_sha256")
    unsigned = dict(preregistration)
    unsigned.pop("preregistration_sha256", None)
    if expected_digest != digest_json(unsigned):
        raise ValueError("preregistration digest does not match its payload")
    if preregistration.get("label_values_accessed") is not False:
        raise ValueError("calibration requires an unopened preregistration")
    features.validate()
    labels.validate()
    assert_row_alignment(features, labels)
    split = preregistration["split"]
    expected_videos = set(split["video_ids"])
    observed = set(labels.video_id.astype(str))
    if (
        observed != expected_videos
        or len(labels.video_id) != int(split["benchmark_train_rows"])
        or not features.quality_eligible.all()
        or not benchmark_partition_mask(features, split, "train").all()
    ):
        raise ValueError("calibration labels must be exactly the eligible benchmark-train rows")

    row_hz = float(preregistration["substrate"]["profile"]["row_hz"])
    maximum_seconds = float(preregistration["target_calibration"]["maximum_horizon_seconds"])
    max_lag_rows = int(math.floor(maximum_seconds * row_hz))
    lags, movement = _movement_curve(features, labels, max_lag_rows)
    short, medium, long = _movement_milestones(lags, movement)
    anchors = tuple(dict.fromkeys((1, short, medium, long, int(lags[-1]))))
    unique_windows = tuple(
        (start, stop)
        for start_index, start in enumerate(anchors)
        for stop in anchors[start_index:]
        if start <= stop
    )
    folds = split["inner_grouped_video_folds"]
    targets = []
    for start, stop in unique_windows:
        temporary = TargetSpec(
            name="calibration",
            label="arousal",
            horizon_rows=tuple(range(start, stop + 1)),
            quantile=0.90,
            transform="positive",
        )
        values = future_target_values(labels, temporary)
        support = features.quality_eligible & target_support_mask(features, temporary)
        start_seconds, stop_seconds = start / row_hz, stop / row_hz
        for quantile, diagnostics in _supported_event_quantiles(
            values, support, labels.video_id, folds
        ):
            hypothesis = EventTargetHypothesis(
                name=(
                    f"arousal_positive_max_{start_seconds:g}_{stop_seconds:g}s_"
                    f"train_q{int(round(quantile * 1000)):03d}"
                ).replace(".", "p"),
                label="arousal",
                start_seconds=start_seconds,
                stop_seconds=stop_seconds,
                transform="positive",
                train_quantile=quantile,
            )
            target = asdict(hypothesis)
            target["horizon_rows"] = list(hypothesis.horizon_rows(row_hz))
            target["benchmark_train_support"] = diagnostics
            targets.append(target)

    calibration = {
        "schema": "veatic21_event_calibration_v12",
        "preregistration_sha256": expected_digest,
        "benchmark_train_video_count": len(expected_videos),
        "benchmark_test_video_count": len(expected_videos),
        "benchmark_train_row_count": int(split["benchmark_train_rows"]),
        "benchmark_test_row_count": int(split["benchmark_test_rows"]),
        "benchmark_test_labels_accessed": False,
        "label_summary": {
            "arousal_min": float(np.min(labels.arousal)),
            "arousal_max": float(np.max(labels.arousal)),
            "arousal_mean": float(np.mean(labels.arousal)),
            "arousal_std": float(np.std(labels.arousal)),
        },
        "movement_curve": {
            "lag_seconds": (lags / row_hz).tolist(),
            "median_absolute_change": movement.tolist(),
            "milestone_rows": [short, medium, long],
            "window_anchor_rows": list(anchors),
        },
        "target_hypotheses": targets,
        "target_selection": (
            "supervised_inner_grouped_video_cv_on_first_70_percent_then_freeze_before_last_30_percent"
        ),
        "threshold_values_are_not_reused": True,
        "next_fit_scope": "eligible_first_70_percent_rows_grouped_by_video",
    }
    calibration["calibration_sha256"] = digest_json(calibration)
    return calibration


__all__ = [
    "EventTargetHypothesis",
    "benchmark_partition_mask",
    "build_event_preregistration",
    "calibrate_event_preregistration",
]
