from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust
from backend.scripts import run_again_dense_2hz_phase6_trial4_fresh_seed_validation as fresh
from backend.scripts import run_again_dense_2hz_phase6_trial4_blocked_15seed as blocked15


def test_seed_roles_are_disjoint_and_fixed() -> None:
    assert robust.DEVELOPMENT_SEEDS == (20260625, 20260626, 20260627, 20260628, 20260629)
    assert robust.VALIDATION_SEEDS == (20260630, 20260631, 20260632, 20260633, 20260634)
    assert set(robust.DEVELOPMENT_SEEDS).isdisjoint(robust.VALIDATION_SEEDS)


def test_robust_objective_penalizes_one_bad_seed() -> None:
    stable = robust.robust_objective([0.010, 0.010, 0.010, 0.010, 0.010])
    outlier = robust.robust_objective([0.020, 0.020, 0.020, 0.020, -0.010])
    assert stable > outlier


def test_validation_gate_requires_four_paired_wins_and_robust_gain() -> None:
    original = {seed: 0.010 for seed in robust.VALIDATION_SEEDS}
    candidate = {seed: 0.012 for seed in robust.VALIDATION_SEEDS}
    gate = robust.validation_gate(candidate, original)
    assert gate["stage_a_pass"] is True
    candidate[robust.VALIDATION_SEEDS[-1]] = 0.005
    gate = robust.validation_gate(candidate, original)
    assert gate["paired_wins"] == 4


def test_dry_run_forbids_heldout_and_grouped_scores() -> None:
    script = Path(robust.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["trials"] == 24
    assert payload["heldout_scores_read"] is False
    assert payload["grouped_scores_read"] is False
    assert payload["original_enqueued_trial_zero"] is True
    assert payload["accelerator"] == "mlx"


def test_trial4_fresh_seed_rescue_is_locked_and_inner_only() -> None:
    assert fresh.FRESH_SEEDS == (20260635, 20260636, 20260637, 20260638, 20260639)
    assert fresh.TRIAL4_PARAMS["hidden"] == 96
    assert fresh.TRIAL4_PARAMS["max_epochs"] == 60
    script = Path(fresh.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["heldout_scores_read"] is False
    assert payload["grouped_scores_read"] is False
    assert payload["seed_20260627_deleted"] is False


def test_predesignated_stress_seed_is_retained_but_cannot_alone_veto() -> None:
    rows = []
    for seed in blocked15.SEEDS:
        values = {
            "frozen_ar_only": 0.20,
            "original_real_residual": 0.21,
            "candidate_real_residual": 0.22 if seed != 20260627 else 0.10,
            "candidate_shuffled_pca_residual": 0.205,
            "candidate_random_pca_residual": 0.205,
            "candidate_label_permutation_residual": 0.20,
            "candidate_train_only_video_mean_residual": 0.205,
            "candidate_diagnostics_only_residual": 0.20,
        }
        rows.extend({"seed": seed, "lane": lane, "pr_auc": value} for lane, value in values.items())
    verdict, seed_df = blocked15.compute_verdict(pd.DataFrame(rows), True)
    assert len(seed_df) == 15
    assert verdict["checks"]["seed_20260627_retained"] is True
    assert verdict["stage_b_pass"] is True
