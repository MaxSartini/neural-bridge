from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.scripts import run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_diagnostic as phase7


def test_scope_is_locked_fresh_and_control_complete() -> None:
    assert phase7.SEEDS == tuple(range(20260684, 20260693))
    assert phase7.GROUPS == tuple(tuple(phase7.SEEDS[i:i + 3]) for i in range(0, 9, 3))
    assert len({seed for group in phase7.GROUPS for seed in group}) == 9
    assert phase7.CONTROLS == (
        "frozen_ar_only",
        "real_residual",
        "shuffled_pca_residual",
        "random_pca_residual",
        "label_permutation_residual",
        "train_only_video_mean_residual",
        "diagnostics_only_residual",
    )
    assert phase7.EXPECTED_ROWS == 84


def test_average_scores_requires_three_aligned_members() -> None:
    item = {
        key: np.arange(4, dtype=np.float32)
        for key in ("train_score", "train_reg", "test_score", "test_reg")
    }
    out = phase7.average_scores([item, item, item])
    np.testing.assert_array_equal(out["test_reg"], item["test_reg"])
    try:
        phase7.average_scores([item, item])
    except ValueError:
        pass
    else:
        raise AssertionError("two-member ensemble was accepted")


def test_dry_run_forbids_search_and_cpu_fallback() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(phase7.__file__).resolve()), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["heldout_hyperparameter_search"] is False
    assert payload["heldout_weight_search"] is False
    assert payload["member_selection"] is False
    assert payload["optuna_used"] is False
    assert payload["accelerator"] == "mlx_gpu_mps"
    assert payload["no_cpu_fallback"] is True
    assert payload["ar_selection"] == "inner_only_top5_lift_then_spearman_then_top10_lift"


def synthetic_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_by_lane = {
        "frozen_ar_only": (0.200, 0.110, 0.120, 0.100, 0.130, 0.140, 0.010, 0.050),
        "real_residual": (0.215, 0.130, 0.140, 0.120, 0.120, 0.130, 0.008, 0.040),
        "shuffled_pca_residual": (0.201, 0.111, 0.121, 0.101, 0.129, 0.139, 0.011, 0.051),
        "random_pca_residual": (0.202, 0.112, 0.122, 0.102, 0.128, 0.138, 0.012, 0.052),
        "label_permutation_residual": (0.198, 0.108, 0.118, 0.098, 0.132, 0.142, 0.013, 0.053),
        "train_only_video_mean_residual": (0.203, 0.113, 0.123, 0.103, 0.127, 0.137, 0.014, 0.054),
        "diagnostics_only_residual": (0.204, 0.114, 0.124, 0.104, 0.126, 0.136, 0.015, 0.055),
    }
    for group, seeds in enumerate(phase7.GROUPS, 1):
        for seed in seeds:
            for lane, values in base_by_lane.items():
                spearman, top1, top5, top10, mae, rmse, bias, peak = values
                rows.append(
                    {
                        "row_type": "member",
                        "group": group,
                        "seed": seed,
                        "lane": lane,
                        "continuous_spearman": spearman - (0.004 if lane == "real_residual" else 0.0),
                        "continuous_pearson": spearman,
                        "continuous_mae": mae,
                        "continuous_rmse": rmse,
                        "continuous_bias": bias,
                        "peak_underprediction": peak,
                        "top_1pct_continuous_lift": top1,
                        "top_5pct_continuous_lift": top5 - (0.004 if lane == "real_residual" else 0.0),
                        "top_10pct_continuous_lift": top10,
                    }
                )
        for lane, values in base_by_lane.items():
            spearman, top1, top5, top10, mae, rmse, bias, peak = values
            rows.append(
                {
                    "row_type": "ensemble",
                    "group": group,
                    "seed": 0,
                    "lane": lane,
                    "continuous_spearman": spearman + group * 0.001,
                    "continuous_pearson": spearman + group * 0.001,
                    "continuous_mae": mae,
                    "continuous_rmse": rmse,
                    "continuous_bias": bias,
                    "peak_underprediction": peak,
                    "top_1pct_continuous_lift": top1 + group * 0.001,
                    "top_5pct_continuous_lift": top5 + group * 0.001,
                    "top_10pct_continuous_lift": top10 + group * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_verdict_separates_ranking_from_exact_value_and_fails_closed() -> None:
    rows = synthetic_rows()
    result, groups = phase7.compute_verdict(rows, audit_pass=True)
    assert len(groups) == 3
    assert result["ranking_lift_diagnostic_pass"] is True
    assert result["exact_value_candidate_pass"] is True
    failed, _ = phase7.compute_verdict(rows.iloc[:-1], audit_pass=True)
    assert failed["ranking_lift_diagnostic_pass"] is False
    assert "exact_scope" in failed["failed_ranking_gates"]
    failed_audit, _ = phase7.compute_verdict(rows, audit_pass=False)
    assert failed_audit["ranking_lift_diagnostic_pass"] is False
    assert "audit_pass" in failed_audit["failed_ranking_gates"]
