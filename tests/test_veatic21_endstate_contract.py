from __future__ import annotations

from dataclasses import replace

import pytest

from backend.scripts import veatic21_endstate_contract as contract


def test_all_124_videos_enter_five_grouped_folds_with_no_reserve() -> None:
    locked = contract.END_STATE_CONTRACT
    assert locked.video_count == 124
    assert locked.reserve_count == 0
    assert locked.grouped_folds == (1, 2, 3, 4, 5)
    assert len(locked.targets) == 4
    assert {target.response_axis for target in locked.targets} == {"arousal", "valence"}


def test_discovery_is_the_exact_six_recipe_bounded_grid() -> None:
    recipes = contract.DISCOVERY_RECIPES
    assert len(recipes) == 6
    assert [
        (recipe.feature_family, recipe.pca_width, recipe.head, recipe.causal_rows)
        for recipe in recipes
    ] == [
        ("temporal_mean_2s", 256, "short_temporal_conv_residual", 5),
        ("temporal_mean_2s", 256, "flat_mlp_residual", 5),
        ("temporal_mean_2s", 64, "short_temporal_conv_residual", 5),
        ("delta", 64, "short_temporal_conv_residual", 5),
        ("delta", 256, "short_temporal_conv_residual", 5),
        ("current", 256, "current_row_mlp", 1),
    ]
    assert any(recipe.prior == "again_promoted" for recipe in recipes)
    assert any(recipe.prior == "old_veatic_delta_pca64" for recipe in recipes)
    assert contract.END_STATE_CONTRACT.optuna_allowed is False
    assert contract.END_STATE_CONTRACT.direct_ar_raw_feature_concatenation_allowed is False
    assert contract.END_STATE_CONTRACT.failed_trial4_path_allowed is False


def test_stage_seeds_are_disjoint_and_confirmation_groups_are_fixed_triples() -> None:
    discovery = set(contract.DISCOVERY_SEEDS)
    privileged = set(contract.PRIVILEGED_CONFIRMATION_SEEDS)
    zero_label = set(contract.ZERO_LABEL_CONFIRMATION_SEEDS)
    assert not discovery & privileged
    assert not discovery & zero_label
    assert not privileged & zero_label
    assert len(privileged) == 9
    assert tuple(map(len, contract.PRIVILEGED_CONFIRMATION_GROUPS)) == (3, 3, 3)
    assert tuple(
        seed for group in contract.PRIVILEGED_CONFIRMATION_GROUPS for seed in group
    ) == contract.PRIVILEGED_CONFIRMATION_SEEDS
    assert contract.ZERO_LABEL_CONFIRMATION_GROUPS == (
        contract.ZERO_LABEL_CONFIRMATION_SEEDS,
    )


def test_exact_seven_privileged_and_seven_zero_label_matrix_lanes() -> None:
    assert len(contract.PRIVILEGED_LANES) == len(set(contract.PRIVILEGED_LANES)) == 7
    assert len(contract.ZERO_LABEL_LANES) == len(set(contract.ZERO_LABEL_LANES)) == 7
    assert len(contract.ZERO_LABEL_RESPONSE_FREE_LANES) == 6
    assert contract.ZERO_LABEL_DESCRIPTIVE_LANES == ("privileged_teacher_ceiling",)
    assert "privileged_teacher_ceiling" not in contract.ZERO_LABEL_FALSE_SIGNAL_CONTROLS
    assert set(contract.ZERO_LABEL_FALSE_SIGNAL_CONTROLS) < set(
        contract.ZERO_LABEL_RESPONSE_FREE_LANES
    )


def test_scored_row_accounting_is_exactly_3920() -> None:
    rows = contract.SCORED_ROWS
    assert rows.privileged_member_per_endpoint == 1260
    assert rows.privileged_ensemble_per_endpoint == 420
    assert rows.continuous_total == 1680
    assert rows.binary_total == 1680
    assert rows.zero_label_member == 420
    assert rows.zero_label_ensemble == 140
    assert rows.zero_label_total == 560
    assert rows.grand_total == 3920


def test_all_new_baselines_and_preprocessing_are_recomputed() -> None:
    fresh = contract.RECOMPUTATION
    assert fresh.fold_safe_pca_per_outer_fold is True
    assert fresh.target_specific_ar_per_fold_and_seed is True
    assert fresh.train_only_standardization_per_fold is True
    assert fresh.train_only_event_threshold_per_target_and_fold is True
    assert fresh.train_only_inner_model_selection is True
    assert fresh.historical_veatic_fitted_artifacts_reusable is False
    assert fresh.again_fitted_artifacts_reusable is False
    assert fresh.old_veatic_schema_authoritative is False


def test_gate_thresholds_match_the_canonical_end_state_protocol() -> None:
    gates = contract.GATES
    assert gates.continuous_spearman_delta_vs_ar == 0.002
    assert gates.continuous_spearman_delta_vs_best_control == 0.002
    assert gates.continuous_top5_delta_vs_ar == 0.001
    assert gates.continuous_top5_delta_vs_best_control == 0.001
    assert gates.continuous_positive_fold_triples == 12
    assert gates.event_pr_auc_delta_vs_ar == 0.005
    assert gates.event_pr_auc_delta_vs_best_control == 0.005
    assert gates.event_positive_fold_triples == gates.fold_triples == 15
    assert gates.positive_fold_means == 5
    assert gates.max_single_fold_triple_positive_contribution == 0.25
    assert gates.zero_label_tier1_directional_fold_wins == 3
    assert gates.zero_label_tier2_directional_fold_wins == 4
    assert gates.zero_label_first30_directional_fold_wins == 4
    assert gates.valence_direction_fold_wins == 4


def test_final_export_refits_selected_models_on_all_124() -> None:
    export = contract.FINAL_EXPORT
    assert export.video_count == 124
    assert export.reserve_count == 0
    assert export.selection_frozen_before_all_video_refit is True
    assert export.refit_pca_on_all_videos is True
    assert export.refit_normalizers_on_all_videos is True
    assert export.refit_event_thresholds_on_all_videos is True
    assert export.refit_target_specific_ar_on_all_videos is True
    assert export.export_all_four_targets is True
    assert export.export_privileged_three_checkpoint_groups is True
    assert export.export_zero_label_three_checkpoint_ensemble is True
    assert export.zero_label_export_has_response_inputs is False
    assert export.zero_label_starts_at_row_zero is True
    assert export.source_caches_are_immutable is True
    assert export.artifact_checksums_and_provenance_required is True
    assert export.failed_seed_or_fold_deletion_allowed is False
    assert export.post_confirmation_weight_search_allowed is False


def test_contract_manifest_is_stable_and_json_safe() -> None:
    first = contract.contract_manifest()
    second = contract.contract_manifest()
    assert first == second
    assert len(first["contract_digest"]) == 32
    assert first["scored_rows"]["grand_total"] == 3920
    assert first["final_export"]["video_count"] == 124


@pytest.mark.parametrize(
    "field,value",
    (
        ("optuna_allowed", True),
        ("direct_ar_raw_feature_concatenation_allowed", True),
        ("failed_trial4_path_allowed", True),
    ),
)
def test_failed_again_paths_are_rejected(field: str, value: bool) -> None:
    edited = replace(contract.END_STATE_CONTRACT, **{field: value})
    with pytest.raises(ValueError):
        contract.validate_endstate_contract(edited)


def test_validation_rejects_reserves_seed_overlap_duplicate_lanes_and_wrong_counts() -> None:
    with pytest.raises(ValueError, match="no internal reserve"):
        contract.validate_endstate_contract(
            replace(contract.END_STATE_CONTRACT, reserve_count=25)
        )

    with pytest.raises(ValueError, match="must be disjoint"):
        contract.validate_endstate_contract(
            replace(
                contract.END_STATE_CONTRACT,
                zero_label_confirmation_seeds=contract.DISCOVERY_SEEDS,
                zero_label_confirmation_groups=(contract.DISCOVERY_SEEDS,),
            )
        )

    duplicate_lanes = contract.PRIVILEGED_LANES[:-1] + (
        contract.PRIVILEGED_LANES[0],
    )
    with pytest.raises(ValueError, match="seven unique lanes"):
        contract.validate_endstate_contract(
            replace(contract.END_STATE_CONTRACT, privileged_lanes=duplicate_lanes)
        )

    wrong_rows = replace(contract.SCORED_ROWS, grand_total=3919)
    with pytest.raises(ValueError, match="3920"):
        contract.validate_endstate_contract(
            replace(contract.END_STATE_CONTRACT, scored_rows=wrong_rows)
        )


def test_validation_rejects_artifact_reuse_and_weakened_export() -> None:
    reused = replace(
        contract.RECOMPUTATION,
        historical_veatic_fitted_artifacts_reusable=True,
    )
    with pytest.raises(ValueError, match="must be fresh"):
        contract.validate_endstate_contract(
            replace(contract.END_STATE_CONTRACT, recomputation=reused)
        )

    leaking_export = replace(
        contract.FINAL_EXPORT,
        zero_label_export_has_response_inputs=True,
    )
    with pytest.raises(ValueError, match="weakened"):
        contract.validate_endstate_contract(
            replace(contract.END_STATE_CONTRACT, final_export=leaking_export)
        )
