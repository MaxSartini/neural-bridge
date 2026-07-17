"""Immutable execution contract for the VEATIC 2.1 end-state programme.

VEATIC 2.1 is a full retraining on the new dense substrate.  Historical
VEATIC and AGAIN results provide bounded architectural priors only: no fitted
PCA, autoregressive score, event threshold, normalizer, checkpoint, or metric
is reusable as a VEATIC 2.1 baseline.  All 124 videos participate in five-fold
grouped-video cross-validation; there is no internal reserved-video panel.

The confirmation row accounting deliberately mirrors the strongest AGAIN
protocols.  Privileged continuous and event confirmations each score nine
members plus three fixed three-member ensembles for every target/fold/lane.
The zero-label confirmation scores three members plus their fixed ensemble.
Metrics do not multiply the scored-row count.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "veatic21_endstate_contract_v1"
DATASET_NAME = "veatic21"
VIDEO_COUNT = 124
RESERVED_VIDEO_COUNT = 0
ROW_FREQUENCY_HZ = 2.0
GROUPED_FOLDS = (1, 2, 3, 4, 5)
EVENT_QUANTILE = 0.90


@dataclass(frozen=True)
class PrimaryTarget:
    """One continuous target and its train-only-q90 event counterpart."""

    name: str
    event_name: str
    response_axis: str
    interpretation: str


PRIMARY_TARGETS = (
    PrimaryTarget(
        name="future_arousal_max_delta_rows_4_10",
        event_name="future_arousal_max_delta_rows_4_10_train_q90",
        response_axis="arousal",
        interpretation="future_positive_arousal_movement",
    ),
    PrimaryTarget(
        name="future_valence_rise_magnitude_rows_4_10",
        event_name="future_valence_rise_magnitude_rows_4_10_train_q90",
        response_axis="valence",
        interpretation="future_positive_valence_movement",
    ),
    PrimaryTarget(
        name="future_valence_drop_magnitude_rows_4_10",
        event_name="future_valence_drop_magnitude_rows_4_10_train_q90",
        response_axis="valence",
        interpretation="future_negative_valence_movement_magnitude",
    ),
    PrimaryTarget(
        name="future_valence_max_abs_movement_rows_4_10",
        event_name="future_valence_max_abs_movement_rows_4_10_train_q90",
        response_axis="valence",
        interpretation="future_bidirectional_valence_movement_magnitude",
    ),
)


@dataclass(frozen=True)
class DiscoveryRecipe:
    """One prespecified recipe in the small, auditable development grid."""

    name: str
    feature_family: str
    pca_width: int
    head: str
    causal_rows: int
    prior: str


# This is an enumeration, not a hyperparameter optimiser.  It combines the
# promoted AGAIN temporal/PCA prior with the old-VEATIC delta/PCA64 prior and
# two explicit mechanism ablations.  Adding a recipe changes the contract.
DISCOVERY_RECIPES = (
    DiscoveryRecipe(
        name="temporal_mean_2s_pca256_short_conv",
        feature_family="temporal_mean_2s",
        pca_width=256,
        head="short_temporal_conv_residual",
        causal_rows=5,
        prior="again_promoted",
    ),
    DiscoveryRecipe(
        name="temporal_mean_2s_pca256_flat_mlp",
        feature_family="temporal_mean_2s",
        pca_width=256,
        head="flat_mlp_residual",
        causal_rows=5,
        prior="again_head_ablation",
    ),
    DiscoveryRecipe(
        name="temporal_mean_2s_pca64_short_conv",
        feature_family="temporal_mean_2s",
        pca_width=64,
        head="short_temporal_conv_residual",
        causal_rows=5,
        prior="cross_prior_width_ablation",
    ),
    DiscoveryRecipe(
        name="delta_pca64_short_conv",
        feature_family="delta",
        pca_width=64,
        head="short_temporal_conv_residual",
        causal_rows=5,
        prior="old_veatic_delta_pca64",
    ),
    DiscoveryRecipe(
        name="delta_pca256_short_conv",
        feature_family="delta",
        pca_width=256,
        head="short_temporal_conv_residual",
        causal_rows=5,
        prior="cross_prior_width_ablation",
    ),
    DiscoveryRecipe(
        name="current_pca256_current_row_mlp",
        feature_family="current",
        pca_width=256,
        head="current_row_mlp",
        causal_rows=1,
        prior="current_row_mechanism_ablation",
    ),
)


DISCOVERY_SEEDS = (20260716, 20260717, 20260718)
PRIVILEGED_CONFIRMATION_SEEDS = tuple(range(20260801, 20260810))
PRIVILEGED_CONFIRMATION_GROUPS = tuple(
    tuple(PRIVILEGED_CONFIRMATION_SEEDS[index : index + 3])
    for index in range(0, 9, 3)
)
ZERO_LABEL_CONFIRMATION_SEEDS = (20260810, 20260811, 20260812)
ZERO_LABEL_CONFIRMATION_GROUPS = (ZERO_LABEL_CONFIRMATION_SEEDS,)


# The canonical privileged seven: one real bridge, its target-specific frozen
# AR baseline, and five matched same-capacity/control-policy lanes.
PRIVILEGED_LANES = (
    "real_residual",
    "frozen_ar_only",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
)
PRIVILEGED_PRIMARY_CONTROLS = (
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
)


# Exactly seven rows are retained in the zero-label matrix.  The first six are
# genuine response-free inference lanes.  The privileged teacher is opened
# only after predictions are sealed, is descriptive, and is never a promotion
# control or deployable input path.
ZERO_LABEL_LANES = (
    "video_supervised_temporal",
    "video_supervised_current_row",
    "diagnostics_only_supervised_temporal",
    "no_video_supervised_temporal",
    "sequence_shuffled_supervised_temporal",
    "label_permutation_supervised_temporal",
    "privileged_teacher_ceiling",
)
ZERO_LABEL_RESPONSE_FREE_LANES = ZERO_LABEL_LANES[:-1]
ZERO_LABEL_FALSE_SIGNAL_CONTROLS = (
    "diagnostics_only_supervised_temporal",
    "no_video_supervised_temporal",
    "sequence_shuffled_supervised_temporal",
    "label_permutation_supervised_temporal",
)
ZERO_LABEL_DESCRIPTIVE_LANES = ("privileged_teacher_ceiling",)


@dataclass(frozen=True)
class RecomputationContract:
    """Items that must be learned afresh from VEATIC 2.1 rows."""

    fold_safe_pca_per_outer_fold: bool
    target_specific_ar_per_fold_and_seed: bool
    train_only_standardization_per_fold: bool
    train_only_event_threshold_per_target_and_fold: bool
    train_only_inner_model_selection: bool
    historical_veatic_fitted_artifacts_reusable: bool
    again_fitted_artifacts_reusable: bool
    old_veatic_schema_authoritative: bool


RECOMPUTATION = RecomputationContract(
    fold_safe_pca_per_outer_fold=True,
    target_specific_ar_per_fold_and_seed=True,
    train_only_standardization_per_fold=True,
    train_only_event_threshold_per_target_and_fold=True,
    train_only_inner_model_selection=True,
    historical_veatic_fitted_artifacts_reusable=False,
    again_fitted_artifacts_reusable=False,
    old_veatic_schema_authoritative=False,
)


@dataclass(frozen=True)
class GateThresholds:
    """Canonical higher-is-better, fail-closed promotion thresholds."""

    continuous_spearman_delta_vs_ar: float = 0.002
    continuous_spearman_delta_vs_best_control: float = 0.002
    continuous_top5_delta_vs_ar: float = 0.001
    continuous_top5_delta_vs_best_control: float = 0.001
    continuous_positive_fold_triples: int = 12
    event_pr_auc_delta_vs_ar: float = 0.005
    event_pr_auc_delta_vs_best_control: float = 0.005
    event_positive_fold_triples: int = 15
    fold_triples: int = 15
    positive_fold_means: int = 5
    paired_median_must_be_positive: bool = True
    max_single_fold_triple_positive_contribution: float = 0.25
    continuous_ensemble_spearman_uplift: float = 0.001
    continuous_ensemble_top5_uplift_must_be_positive: bool = True
    event_ensemble_uplift: float = 0.001
    event_ensemble_positive_fold_triples: int = 12
    label_permutation_minus_ar_max: float = 0.001
    zero_label_aggregate_delta_must_be_positive: bool = True
    zero_label_tier1_directional_fold_wins: int = 3
    zero_label_tier2_directional_fold_wins: int = 4
    zero_label_first30_directional_fold_wins: int = 4
    zero_label_bootstrap_lower95_must_be_positive: bool = True
    valence_direction_fold_wins: int = 4


GATES = GateThresholds()


@dataclass(frozen=True)
class ScoredRowCounts:
    """Exact rows in the locked end-state confirmation matrix."""

    privileged_member_per_endpoint: int
    privileged_ensemble_per_endpoint: int
    continuous_total: int
    binary_total: int
    zero_label_member: int
    zero_label_ensemble: int
    zero_label_total: int
    grand_total: int


def expected_scored_rows() -> ScoredRowCounts:
    targets = len(PRIMARY_TARGETS)
    folds = len(GROUPED_FOLDS)
    privileged_lanes = len(PRIVILEGED_LANES)
    zero_lanes = len(ZERO_LABEL_LANES)
    privileged_members = (
        targets * folds * len(PRIVILEGED_CONFIRMATION_SEEDS) * privileged_lanes
    )
    privileged_ensembles = (
        targets * folds * len(PRIVILEGED_CONFIRMATION_GROUPS) * privileged_lanes
    )
    privileged_total = privileged_members + privileged_ensembles
    zero_members = targets * folds * len(ZERO_LABEL_CONFIRMATION_SEEDS) * zero_lanes
    zero_ensembles = targets * folds * len(ZERO_LABEL_CONFIRMATION_GROUPS) * zero_lanes
    zero_total = zero_members + zero_ensembles
    return ScoredRowCounts(
        privileged_member_per_endpoint=privileged_members,
        privileged_ensemble_per_endpoint=privileged_ensembles,
        continuous_total=privileged_total,
        binary_total=privileged_total,
        zero_label_member=zero_members,
        zero_label_ensemble=zero_ensembles,
        zero_label_total=zero_total,
        grand_total=privileged_total * 2 + zero_total,
    )


SCORED_ROWS = expected_scored_rows()


@dataclass(frozen=True)
class FinalExportRequirements:
    """Requirements for refitting the selected VEATIC 2.1 model on all videos."""

    video_count: int
    reserve_count: int
    selection_frozen_before_all_video_refit: bool
    refit_pca_on_all_videos: bool
    refit_normalizers_on_all_videos: bool
    refit_event_thresholds_on_all_videos: bool
    refit_target_specific_ar_on_all_videos: bool
    export_all_four_targets: bool
    export_privileged_three_checkpoint_groups: bool
    export_zero_label_three_checkpoint_ensemble: bool
    zero_label_export_has_response_inputs: bool
    zero_label_starts_at_row_zero: bool
    source_caches_are_immutable: bool
    artifact_checksums_and_provenance_required: bool
    failed_seed_or_fold_deletion_allowed: bool
    post_confirmation_weight_search_allowed: bool


FINAL_EXPORT = FinalExportRequirements(
    video_count=VIDEO_COUNT,
    reserve_count=RESERVED_VIDEO_COUNT,
    selection_frozen_before_all_video_refit=True,
    refit_pca_on_all_videos=True,
    refit_normalizers_on_all_videos=True,
    refit_event_thresholds_on_all_videos=True,
    refit_target_specific_ar_on_all_videos=True,
    export_all_four_targets=True,
    export_privileged_three_checkpoint_groups=True,
    export_zero_label_three_checkpoint_ensemble=True,
    zero_label_export_has_response_inputs=False,
    zero_label_starts_at_row_zero=True,
    source_caches_are_immutable=True,
    artifact_checksums_and_provenance_required=True,
    failed_seed_or_fold_deletion_allowed=False,
    post_confirmation_weight_search_allowed=False,
)


@dataclass(frozen=True)
class EndStateContract:
    schema_version: str
    dataset: str
    video_count: int
    reserve_count: int
    row_frequency_hz: float
    grouped_folds: tuple[int, ...]
    targets: tuple[PrimaryTarget, ...]
    discovery_recipes: tuple[DiscoveryRecipe, ...]
    discovery_seeds: tuple[int, ...]
    privileged_confirmation_seeds: tuple[int, ...]
    privileged_confirmation_groups: tuple[tuple[int, ...], ...]
    zero_label_confirmation_seeds: tuple[int, ...]
    zero_label_confirmation_groups: tuple[tuple[int, ...], ...]
    privileged_lanes: tuple[str, ...]
    zero_label_lanes: tuple[str, ...]
    recomputation: RecomputationContract
    gates: GateThresholds
    scored_rows: ScoredRowCounts
    final_export: FinalExportRequirements
    optuna_allowed: bool
    direct_ar_raw_feature_concatenation_allowed: bool
    failed_trial4_path_allowed: bool


END_STATE_CONTRACT = EndStateContract(
    schema_version=SCHEMA_VERSION,
    dataset=DATASET_NAME,
    video_count=VIDEO_COUNT,
    reserve_count=RESERVED_VIDEO_COUNT,
    row_frequency_hz=ROW_FREQUENCY_HZ,
    grouped_folds=GROUPED_FOLDS,
    targets=PRIMARY_TARGETS,
    discovery_recipes=DISCOVERY_RECIPES,
    discovery_seeds=DISCOVERY_SEEDS,
    privileged_confirmation_seeds=PRIVILEGED_CONFIRMATION_SEEDS,
    privileged_confirmation_groups=PRIVILEGED_CONFIRMATION_GROUPS,
    zero_label_confirmation_seeds=ZERO_LABEL_CONFIRMATION_SEEDS,
    zero_label_confirmation_groups=ZERO_LABEL_CONFIRMATION_GROUPS,
    privileged_lanes=PRIVILEGED_LANES,
    zero_label_lanes=ZERO_LABEL_LANES,
    recomputation=RECOMPUTATION,
    gates=GATES,
    scored_rows=SCORED_ROWS,
    final_export=FINAL_EXPORT,
    optuna_allowed=False,
    direct_ar_raw_feature_concatenation_allowed=False,
    failed_trial4_path_allowed=False,
)


def _flatten(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(seed for group in groups for seed in group)


def validate_endstate_contract(contract: EndStateContract = END_STATE_CONTRACT) -> None:
    """Fail closed if an edit broadens or weakens the locked programme."""

    if contract.video_count != 124 or contract.reserve_count != 0:
        raise ValueError("VEATIC 2.1 must use all 124 videos with no internal reserve")
    if contract.grouped_folds != (1, 2, 3, 4, 5):
        raise ValueError("VEATIC 2.1 requires exactly five grouped-video folds")
    if len(contract.targets) != 4 or len({target.name for target in contract.targets}) != 4:
        raise ValueError("VEATIC 2.1 requires four unique primary targets")
    if len({target.event_name for target in contract.targets}) != 4:
        raise ValueError("Every primary target needs a unique binary event endpoint")
    if {target.response_axis for target in contract.targets} != {"arousal", "valence"}:
        raise ValueError("The target contract must cover native arousal and valence")

    if contract.discovery_recipes != DISCOVERY_RECIPES or len(contract.discovery_recipes) != 6:
        raise ValueError("Discovery is limited to the six prespecified fixed recipes")
    if contract.optuna_allowed:
        raise ValueError("Optuna is not part of the bounded VEATIC 2.1 discovery")
    if contract.direct_ar_raw_feature_concatenation_allowed:
        raise ValueError("The failed direct AR-plus-raw concatenation path is forbidden")
    if contract.failed_trial4_path_allowed:
        raise ValueError("The failed Trial-4 path is forbidden")

    seed_sets = (
        set(contract.discovery_seeds),
        set(contract.privileged_confirmation_seeds),
        set(contract.zero_label_confirmation_seeds),
    )
    if any(not seeds for seeds in seed_sets):
        raise ValueError("Every stage needs at least one seed")
    if any(seed_sets[left] & seed_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Discovery, privileged confirmation, and zero-label seeds must be disjoint")
    if len(contract.privileged_confirmation_seeds) != 9:
        raise ValueError("Privileged confirmation requires exactly nine seeds")
    if tuple(map(len, contract.privileged_confirmation_groups)) != (3, 3, 3):
        raise ValueError("Privileged seeds must form three fixed triples")
    if _flatten(contract.privileged_confirmation_groups) != contract.privileged_confirmation_seeds:
        raise ValueError("Privileged triples must partition the nine seeds exactly once")
    if len(contract.zero_label_confirmation_seeds) != 3:
        raise ValueError("Zero-label confirmation requires exactly three seeds")
    if contract.zero_label_confirmation_groups != (contract.zero_label_confirmation_seeds,):
        raise ValueError("Zero-label confirmation requires one fixed three-seed ensemble")

    if len(contract.privileged_lanes) != 7 or len(set(contract.privileged_lanes)) != 7:
        raise ValueError("Privileged confirmation requires exactly seven unique lanes")
    if len(contract.zero_label_lanes) != 7 or len(set(contract.zero_label_lanes)) != 7:
        raise ValueError("Zero-label confirmation requires exactly seven unique lanes")
    if "privileged_teacher_ceiling" not in contract.zero_label_lanes:
        raise ValueError("The descriptive privileged teacher ceiling must remain in the matrix")
    if "privileged_teacher_ceiling" in ZERO_LABEL_FALSE_SIGNAL_CONTROLS:
        raise ValueError("The privileged teacher ceiling cannot be a promotion control")

    recomputation = contract.recomputation
    required_fresh = (
        recomputation.fold_safe_pca_per_outer_fold,
        recomputation.target_specific_ar_per_fold_and_seed,
        recomputation.train_only_standardization_per_fold,
        recomputation.train_only_event_threshold_per_target_and_fold,
        recomputation.train_only_inner_model_selection,
    )
    forbidden_reuse = (
        recomputation.historical_veatic_fitted_artifacts_reusable,
        recomputation.again_fitted_artifacts_reusable,
        recomputation.old_veatic_schema_authoritative,
    )
    if not all(required_fresh) or any(forbidden_reuse):
        raise ValueError("Every VEATIC 2.1 PCA, AR, threshold, and fitted baseline must be fresh")

    targets = len(contract.targets)
    folds = len(contract.grouped_folds)
    privileged_member = targets * folds * 9 * 7
    privileged_ensemble = targets * folds * 3 * 7
    zero_member = targets * folds * 3 * 7
    zero_ensemble = targets * folds * 1 * 7
    expected = ScoredRowCounts(
        privileged_member_per_endpoint=privileged_member,
        privileged_ensemble_per_endpoint=privileged_ensemble,
        continuous_total=privileged_member + privileged_ensemble,
        binary_total=privileged_member + privileged_ensemble,
        zero_label_member=zero_member,
        zero_label_ensemble=zero_ensemble,
        zero_label_total=zero_member + zero_ensemble,
        grand_total=2 * (privileged_member + privileged_ensemble)
        + zero_member
        + zero_ensemble,
    )
    if contract.scored_rows != expected or (
        expected.continuous_total,
        expected.binary_total,
        expected.zero_label_total,
        expected.grand_total,
    ) != (1680, 1680, 560, 3920):
        raise ValueError("Locked scored-row accounting must be 1680 + 1680 + 560 = 3920")

    export = contract.final_export
    if export.video_count != 124 or export.reserve_count != 0:
        raise ValueError("Final export must refit on all 124 videos")
    required_export_flags = (
        export.selection_frozen_before_all_video_refit,
        export.refit_pca_on_all_videos,
        export.refit_normalizers_on_all_videos,
        export.refit_event_thresholds_on_all_videos,
        export.refit_target_specific_ar_on_all_videos,
        export.export_all_four_targets,
        export.export_privileged_three_checkpoint_groups,
        export.export_zero_label_three_checkpoint_ensemble,
        export.zero_label_starts_at_row_zero,
        export.source_caches_are_immutable,
        export.artifact_checksums_and_provenance_required,
    )
    forbidden_export_flags = (
        export.zero_label_export_has_response_inputs,
        export.failed_seed_or_fold_deletion_allowed,
        export.post_confirmation_weight_search_allowed,
    )
    if not all(required_export_flags) or any(forbidden_export_flags):
        raise ValueError("Final all-124 export requirements have been weakened")


def contract_manifest(contract: EndStateContract = END_STATE_CONTRACT) -> dict[str, Any]:
    """Return a JSON-safe manifest with a stable digest."""

    validate_endstate_contract(contract)
    payload = asdict(contract)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **payload,
        "contract_digest": hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest(),
    }


validate_endstate_contract()


__all__ = [
    "DATASET_NAME",
    "DISCOVERY_RECIPES",
    "DISCOVERY_SEEDS",
    "END_STATE_CONTRACT",
    "EVENT_QUANTILE",
    "FINAL_EXPORT",
    "GATES",
    "GROUPED_FOLDS",
    "PRIVILEGED_CONFIRMATION_GROUPS",
    "PRIVILEGED_CONFIRMATION_SEEDS",
    "PRIVILEGED_LANES",
    "PRIMARY_TARGETS",
    "RECOMPUTATION",
    "RESERVED_VIDEO_COUNT",
    "ROW_FREQUENCY_HZ",
    "SCHEMA_VERSION",
    "SCORED_ROWS",
    "VIDEO_COUNT",
    "ZERO_LABEL_CONFIRMATION_GROUPS",
    "ZERO_LABEL_CONFIRMATION_SEEDS",
    "ZERO_LABEL_DESCRIPTIVE_LANES",
    "ZERO_LABEL_FALSE_SIGNAL_CONTROLS",
    "ZERO_LABEL_LANES",
    "ZERO_LABEL_RESPONSE_FREE_LANES",
    "DiscoveryRecipe",
    "EndStateContract",
    "FinalExportRequirements",
    "GateThresholds",
    "PrimaryTarget",
    "RecomputationContract",
    "ScoredRowCounts",
    "contract_manifest",
    "expected_scored_rows",
    "validate_endstate_contract",
]
