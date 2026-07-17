from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import random

import pytest

from backend.scripts import veatic21_discovery as discovery
from backend.scripts import veatic21_endstate_contract as endstate


def _row_counts() -> dict[str, int]:
    return {str(video): 80 + (video * 37) % 131 for video in range(1, 125)}


@pytest.fixture(scope="module")
def plan() -> discovery.NestedDiscoveryPlan:
    return discovery.build_nested_discovery_plan(_row_counts())


def _winner_map(plan: discovery.NestedDiscoveryPlan) -> dict[tuple[str, str], str]:
    names = tuple(recipe.name for recipe in plan.recipes)
    return {
        (target, protocol): names[(target_index + 2 * protocol_index) % len(names)]
        for target_index, target in enumerate(plan.targets)
        for protocol_index, protocol in enumerate(plan.protocols)
    }


def _full_score_rows(
    plan: discovery.NestedDiscoveryPlan,
    *,
    winners: dict[tuple[str, str], str] | None = None,
    exact_tie: bool = False,
) -> tuple[discovery.DiscoveryScoreRow, ...]:
    winners = _winner_map(plan) if winners is None else winners
    rows: list[discovery.DiscoveryScoreRow] = []
    for target in plan.targets:
        for protocol in plan.protocols:
            winner = winners[(target, protocol)]
            for outer in plan.outer_folds:
                for recipe in plan.recipes:
                    if exact_tie:
                        recipe_quality = 0.2
                    else:
                        recipe_quality = 0.55 if recipe.name == winner else 0.10
                    for inner in outer.inner_folds:
                        for seed_index, seed in enumerate(plan.discovery_seeds):
                            # Shared nuisance terms cannot change the recipe ranking.
                            nuisance = (
                                outer.outer_fold * 1e-4
                                + inner.fold * 1e-5
                                + seed_index * 1e-6
                            )
                            if protocol in (
                                discovery.PRIVILEGED_CONTINUOUS,
                                discovery.ZERO_LABEL_CONTINUOUS,
                            ):
                                metrics = {
                                    discovery.SPEARMAN: recipe_quality + nuisance,
                                    discovery.TOP5_LIFT: recipe_quality / 3.0 + nuisance,
                                }
                            else:
                                metrics = {
                                    discovery.TRAIN_Q90_PR_AUC: recipe_quality + nuisance
                                }
                            rows.append(
                                discovery.make_discovery_score_row(
                                    plan,
                                    target=target,
                                    protocol=protocol,
                                    outer_fold=outer.outer_fold,
                                    recipe=recipe.name,
                                    inner_fold=inner.fold,
                                    seed=seed,
                                    metrics=metrics,
                                )
                            )
    return tuple(rows)


@pytest.fixture(scope="module")
def score_rows(
    plan: discovery.NestedDiscoveryPlan,
) -> tuple[discovery.DiscoveryScoreRow, ...]:
    return _full_score_rows(plan)


@pytest.fixture(scope="module")
def artifact(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
) -> discovery.DiscoverySelectionArtifact:
    return discovery.select_nested_recipes(plan, score_rows)


def test_default_plan_uses_exact_endstate_recipes_targets_and_disjoint_seeds(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    assert tuple(recipe.name for recipe in plan.recipes) == tuple(
        recipe.name for recipe in endstate.DISCOVERY_RECIPES
    )
    assert plan.targets == tuple(target.name for target in endstate.PRIMARY_TARGETS)
    assert plan.discovery_seeds == endstate.DISCOVERY_SEEDS
    assert not set(plan.discovery_seeds) & set(plan.confirmation_seeds)
    assert plan.protocols == discovery.SUPPORTED_PROTOCOLS
    assert discovery.normalize_recipes(plan.recipes) == plan.recipes


def test_all_124_videos_enter_five_outer_folds_and_inner_cv_is_train_only(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    audit = discovery.audit_nested_ownership(plan)
    assert audit.passed
    assert all(audit.check_map().values())
    assert plan.video_count == 124
    assert len(plan.outer_folds) == 5
    outer_test = [video for fold in plan.outer_folds for video in fold.test_videos]
    assert len(outer_test) == len(set(outer_test)) == 124
    for outer in plan.outer_folds:
        assert len(outer.inner_folds) == 3
        assert not set(outer.train_videos) & set(outer.test_videos)
        inner_validation = [
            video for fold in outer.inner_folds for video in fold.validation_videos
        ]
        assert set(inner_validation) == set(outer.train_videos)
        assert len(inner_validation) == len(set(inner_validation))
        assert not set(inner_validation) & set(outer.test_videos)


def test_nested_row_balancing_and_plan_digest_are_deterministic(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    repeated = discovery.build_nested_discovery_plan(_row_counts())
    assert repeated == plan
    discovery.validate_plan_digest(plan)
    max_video_rows = max(_row_counts().values())
    outer_loads = [fold.test_rows for fold in plan.outer_folds]
    assert max(outer_loads) - min(outer_loads) <= max_video_rows
    for outer in plan.outer_folds:
        inner_loads = [fold.validation_rows for fold in outer.inner_folds]
        assert max(inner_loads) - min(inner_loads) <= max_video_rows


def test_seed_overlap_fails_before_any_scores_can_be_selected() -> None:
    with pytest.raises(discovery.DiscoveryContractError, match="must be disjoint"):
        discovery.build_nested_discovery_plan(
            _row_counts(),
            discovery_seeds=(101, 102, 103),
            confirmation_seeds=(103, 201, 202),
        )


def test_generic_recipe_mappings_are_available_only_when_explicitly_unlocked() -> None:
    recipes = discovery.normalize_recipes(
        (
            {
                "name": "small",
                "feature_family": "current",
                "pca_width": 64,
                "head": "mlp",
                "causal_rows": 1,
            },
            {
                "name": "large",
                "feature_family": "temporal",
                "pca_width": 256,
                "head": "conv",
                "causal_rows": 5,
            },
        ),
        enforce_canonical=False,
    )
    assert [recipe.name for recipe in recipes] == ["small", "large"]
    assert [recipe.complexity_score for recipe in recipes] == [64, 1280]
    with pytest.raises(discovery.DiscoveryContractError, match="exact bounded"):
        discovery.normalize_recipes(recipes, enforce_canonical=True)


def test_complete_matrix_is_exact_recipe_by_inner_fold_by_seed(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
) -> None:
    audit, normalized = discovery.audit_score_matrix(plan, score_rows)
    expected = (
        len(plan.targets)
        * len(plan.protocols)
        * len(plan.outer_folds)
        * len(plan.recipes)
        * plan.inner_fold_count
        * len(plan.discovery_seeds)
    )
    assert audit.passed
    assert audit.expected_rows == audit.observed_rows == len(normalized) == expected
    assert expected == 4 * 4 * 5 * 6 * 3 * 3


def test_incomplete_or_duplicate_matrix_fails_closed(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
) -> None:
    with pytest.raises(discovery.DiscoveryContractError, match="Incomplete or invalid"):
        discovery.select_nested_recipes(plan, score_rows[:-1])
    with pytest.raises(discovery.DiscoveryContractError, match="Incomplete or invalid"):
        discovery.select_nested_recipes(plan, score_rows + (score_rows[0],))


def test_score_ownership_rejects_outer_test_leakage(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
) -> None:
    row = score_rows[0]
    leaked_video = plan.outer(row.outer_fold).test_videos[0]
    leaked = replace(row, fit_videos=row.fit_videos + (leaked_video,))
    contaminated = (leaked,) + score_rows[1:]
    audit, _ = discovery.audit_score_matrix(plan, contaminated)
    assert not audit.passed
    assert row.key in audit.ownership_failures
    with pytest.raises(discovery.DiscoveryContractError, match="ownership"):
        discovery.select_nested_recipes(plan, contaminated)


def test_outer_test_scores_have_no_accepted_input_path(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
    artifact: discovery.DiscoverySelectionArtifact,
) -> None:
    row = score_rows[0]
    with pytest.raises(discovery.DiscoveryContractError, match="Outer-test"):
        discovery.make_discovery_score_row(
            plan,
            target=row.target,
            protocol=row.protocol,
            outer_fold=row.outer_fold,
            recipe=row.recipe,
            inner_fold=row.inner_fold,
            seed=row.seed,
            metrics={
                discovery.SPEARMAN: 0.1,
                discovery.TOP5_LIFT: 0.1,
                "outer_test_spearman": 999.0,
            },
        )
    with pytest.raises(discovery.DiscoveryContractError, match="Outer-test"):
        discovery.make_discovery_score_row(
            plan,
            target=row.target,
            protocol=row.protocol,
            outer_fold=row.outer_fold,
            recipe=row.recipe,
            inner_fold=row.inner_fold,
            seed=row.seed,
            metrics={
                discovery.SPEARMAN: 0.1,
                discovery.TOP5_LIFT: 0.1,
                "outer_score": 999.0,
            },
        )

    mapping = {
        "target": row.target,
        "protocol": row.protocol,
        "outer_fold": row.outer_fold,
        "recipe": row.recipe,
        "inner_fold": row.inner_fold,
        "seed": row.seed,
        "score_scope": row.score_scope,
        "fit_videos": row.fit_videos,
        "validation_videos": row.validation_videos,
        "ownership_digest": row.ownership_digest,
        "metrics": row.metric_map(),
        "outer_test_score": 1_000_000.0,
    }
    with pytest.raises(discovery.DiscoveryContractError, match="no input path"):
        discovery.audit_score_matrix(plan, (mapping,) + score_rows[1:])
    assert artifact.outer_test_scores_used is False
    assert all(not selection.outer_test_scores_used for selection in artifact.selections)


def test_target_and_protocol_selections_are_independent(
    plan: discovery.NestedDiscoveryPlan,
    artifact: discovery.DiscoverySelectionArtifact,
) -> None:
    winners = _winner_map(plan)
    selected = {
        (target, protocol): artifact.selection(target, protocol, 1).selected_recipe
        for target in plan.targets
        for protocol in plan.protocols
    }
    assert selected == winners
    assert len(set(selected.values())) > 1


def test_protocol_rules_prioritize_continuous_and_binary_primary_metrics(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    target = plan.targets[0]
    continuous = discovery.PRIVILEGED_CONTINUOUS
    binary = discovery.PRIVILEGED_BINARY
    first, second = plan.recipes[:2]
    winners = _winner_map(plan)
    winners[(target, continuous)] = first.name
    winners[(target, binary)] = second.name
    rows = list(_full_score_rows(plan, winners=winners))

    # Make the losing continuous recipe's top-5 metric enormous.  Spearman is
    # the prespecified first key and therefore still determines the choice.
    for index, row in enumerate(rows):
        if row.target == target and row.protocol == continuous and row.recipe == second.name:
            metrics = row.metric_map()
            metrics[discovery.TOP5_LIFT] = 100.0
            rows[index] = replace(row, metrics=tuple(
                discovery.MetricValue(name, value)
                for name, value in sorted(metrics.items())
            ))
    selected = discovery.select_nested_recipes(plan, rows)
    assert selected.selection(target, continuous, 1).selected_recipe == first.name
    assert selected.selection(target, binary, 1).selected_recipe == second.name


def test_exact_metric_ties_use_complexity_then_frozen_recipe_order(
    plan: discovery.NestedDiscoveryPlan,
) -> None:
    tied = discovery.select_nested_recipes(plan, _full_score_rows(plan, exact_tie=True))
    simplest = min(plan.recipes, key=lambda recipe: (recipe.complexity_score, recipe.order))
    assert all(
        selection.selected_recipe == simplest.name for selection in tied.selections
    )

    # Keep the two PCA64 temporal recipes tied and make all other recipes worse;
    # their equal complexity is resolved by canonical order.
    pca64 = [recipe for recipe in plan.recipes if recipe.complexity_score == 320]
    assert len(pca64) == 2
    winners = _winner_map(plan)
    rows = list(_full_score_rows(plan, winners=winners))
    for index, row in enumerate(rows):
        quality = 0.7 if row.recipe in {recipe.name for recipe in pca64} else 0.1
        if row.protocol in (
            discovery.PRIVILEGED_CONTINUOUS,
            discovery.ZERO_LABEL_CONTINUOUS,
        ):
            metrics = {
                discovery.SPEARMAN: quality,
                discovery.TOP5_LIFT: quality,
            }
        else:
            metrics = {discovery.TRAIN_Q90_PR_AUC: quality}
        rows[index] = replace(
            row,
            metrics=tuple(
                discovery.MetricValue(name, value)
                for name, value in sorted(metrics.items())
            ),
        )
    ordered_tie = discovery.select_nested_recipes(plan, rows)
    assert all(
        selection.selected_recipe == pca64[0].name
        for selection in ordered_tie.selections
    )


def test_selection_and_artifact_digests_are_order_independent_and_immutable(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
    artifact: discovery.DiscoverySelectionArtifact,
) -> None:
    shuffled = list(score_rows)
    random.Random(20260716).shuffle(shuffled)
    repeated = discovery.select_nested_recipes(plan, shuffled)
    assert repeated == artifact
    discovery.verify_selection_artifact(artifact, plan=plan, score_rows=score_rows)
    assert len(artifact.artifact_digest) == len(plan.digest) == 64
    with pytest.raises(FrozenInstanceError):
        artifact.outer_test_scores_used = True  # type: ignore[misc]
    corrupted = replace(artifact, score_rows=artifact.score_rows - 1)
    with pytest.raises(discovery.DiscoveryContractError, match="digest mismatch"):
        discovery.verify_selection_artifact(corrupted)


def test_inner_scope_and_required_protocol_metrics_fail_closed(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: tuple[discovery.DiscoveryScoreRow, ...],
) -> None:
    wrong_scope = replace(score_rows[0], score_scope="outer_test")
    audit, _ = discovery.audit_score_matrix(plan, (wrong_scope,) + score_rows[1:])
    assert not audit.passed
    assert wrong_scope.key in audit.scope_failures

    missing_metric = replace(
        score_rows[0],
        metrics=(discovery.MetricValue(discovery.SPEARMAN, 0.1),),
    )
    audit, _ = discovery.audit_score_matrix(plan, (missing_metric,) + score_rows[1:])
    assert not audit.passed
    assert missing_metric.key in audit.metric_failures
