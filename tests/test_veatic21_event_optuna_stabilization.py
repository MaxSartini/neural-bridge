from __future__ import annotations

from backend.scripts import run_veatic21_event_optuna_stabilization as optuna_run


def test_schedule_locks_current_training_semantics_and_fresh_showdown() -> None:
    schedule = optuna_run.dry_schedule()
    assert schedule["trial_count"] == 50
    assert schedule["search_member_fits"] == 750
    assert schedule["showdown_residual_member_fits"] == 150
    assert schedule["checkpoint_selection_from_epoch"] == 1
    assert schedule["early_stop_not_before_epoch"] == 50
    assert set(optuna_run.SEARCH_SEEDS).isdisjoint(optuna_run.SHOWDOWN_SEEDS)
    assert len(optuna_run.SEARCH_PANELS) == 5
    assert len(optuna_run.HELDOUT_PANELS) == 10
    assert set(optuna_run.SEARCH_PANELS) | set(optuna_run.HELDOUT_PANELS) == set(
        optuna_run.ALL_PANELS
    )


def test_robust_objective_prefers_broad_stable_panel_gains() -> None:
    stable = [
        {
            "ensemble_delta_vs_ar": 0.01,
            "member_mean_delta_vs_ar": 0.006,
            "ensemble_uplift": 0.004,
        }
        for _ in range(5)
    ]
    unstable = [
        {
            "ensemble_delta_vs_ar": delta,
            "member_mean_delta_vs_ar": 0.006,
            "ensemble_uplift": 0.004,
        }
        for delta in (-0.02, 0.005, 0.015, 0.02, 0.03)
    ]
    stable_value, stable_metrics = optuna_run.robust_objective(stable)
    unstable_value, _ = optuna_run.robust_objective(unstable)
    assert stable_metrics["win_rate"] == 1.0
    assert stable_value > unstable_value


def test_showdown_audit_and_gate_use_paired_tuned_original_and_ar() -> None:
    members = []
    ensembles = []
    for outer, inner in optuna_run.ALL_PANELS:
        for lane, score in (("tuned", 0.315), ("original", 0.305)):
            for seed in optuna_run.SHOWDOWN_SEEDS:
                members.append(
                    {
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "lane": lane,
                        "seed": seed,
                        "pr_auc": score,
                        "ar_pr_auc": 0.300,
                        "delta_vs_ar": score - 0.300,
                        "ar_prediction_digest": f"ar-{outer}-{inner}-{seed}",
                        "heldout_label_digest": f"label-{outer}-{inner}",
                        "undefined_per_video_pr_auc_score_filled": False,
                        "zero_event_videos_excluded_from_pooled_negatives": False,
                        "outer_test_scores_used": False,
                    }
                )
            ensembles.append(
                {
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "lane": lane,
                    "pr_auc": score + 0.001,
                    "ar_pr_auc": 0.300,
                    "delta_vs_ar": score + 0.001 - 0.300,
                    "ensemble_uplift_over_member_mean": 0.001,
                    "heldout_label_digest": f"label-{outer}-{inner}",
                    "undefined_per_video_pr_auc_score_filled": False,
                    "zero_event_videos_excluded_from_pooled_negatives": False,
                    "outer_test_scores_used": False,
                }
            )
    audit = optuna_run.audit_showdown(members, ensembles)
    summary = optuna_run.summarize_showdown(members, ensembles)
    assert audit["passed"] is True
    assert summary["passed"] is True
    assert summary["primary_heldout_10"]["tuned_vs_original"]["wins"] == 10
    assert summary["primary_heldout_10"]["tuned_vs_ar"]["member_wins"] == 50
