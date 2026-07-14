from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from backend.scripts import run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm as confirm


def test_scope_is_fresh_fixed_and_control_complete() -> None:
    assert confirm.SEEDS == tuple(range(20260693, 20260708))
    assert confirm.GROUPS == tuple(tuple(confirm.SEEDS[i:i + 3]) for i in range(0, 15, 3))
    assert len({seed for group in confirm.GROUPS for seed in group}) == 15
    assert confirm.EXPECTED_ROWS == 140


def test_dry_run_locks_ranking_only_mlx_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(confirm.__file__).resolve()), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["rows"] == 140
    assert payload["confirmation_scope"] == "blocked_continuous_ranking_lift_only"
    assert payload["exact_value_promotion_forbidden"] is True
    assert payload["accelerator"] == "mlx_gpu_mps"
    assert payload["no_cpu_fallback"] is True
    assert payload["heldout_hyperparameter_search"] is False


def synthetic_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lane_values = {
        "frozen_ar_only": (0.200, 0.110, 0.120, 0.100, 0.120, 0.140),
        "real_residual": (0.216, 0.130, 0.142, 0.121, 0.121, 0.141),
        "shuffled_pca_residual": (0.202, 0.112, 0.122, 0.102, 0.122, 0.142),
        "random_pca_residual": (0.203, 0.113, 0.123, 0.103, 0.123, 0.143),
        "label_permutation_residual": (0.198, 0.108, 0.118, 0.098, 0.124, 0.144),
        "train_only_video_mean_residual": (0.204, 0.114, 0.124, 0.104, 0.124, 0.144),
        "diagnostics_only_residual": (0.205, 0.115, 0.125, 0.105, 0.125, 0.145),
    }
    for group, seeds in enumerate(confirm.GROUPS, 1):
        for seed in seeds:
            for lane, values in lane_values.items():
                spearman, top1, top5, top10, mae, rmse = values
                rows.append(
                    {
                        "row_type": "member",
                        "group": group,
                        "seed": seed,
                        "lane": lane,
                        "continuous_spearman": spearman
                        - (0.005 if lane == "real_residual" else 0)
                        + (((seed % 3) - 1) * 0.01 if lane == "real_residual" else 0),
                        "continuous_pearson": spearman,
                        "continuous_mae": mae,
                        "continuous_rmse": rmse,
                        "continuous_bias": 0.01,
                        "peak_underprediction": 0.05,
                        "top_1pct_continuous_lift": top1,
                        "top_5pct_continuous_lift": top5 - (0.004 if lane == "real_residual" else 0),
                        "top_10pct_continuous_lift": top10,
                    }
                )
        for lane, values in lane_values.items():
            spearman, top1, top5, top10, mae, rmse = values
            offset = (group - 3) * 0.0002
            rows.append(
                {
                    "row_type": "ensemble",
                    "group": group,
                    "seed": 0,
                    "lane": lane,
                    "continuous_spearman": spearman + offset,
                    "continuous_pearson": spearman + offset,
                    "continuous_mae": mae,
                    "continuous_rmse": rmse,
                    "continuous_bias": 0.01,
                    "peak_underprediction": 0.05,
                    "top_1pct_continuous_lift": top1 + offset,
                    "top_5pct_continuous_lift": top5 + offset,
                    "top_10pct_continuous_lift": top10 + offset,
                }
            )
    return pd.DataFrame(rows)


def test_verdict_passes_ranking_but_never_promotes_exact_values() -> None:
    result, group_frame = confirm.compute_verdict(synthetic_rows(), audit_pass=True)
    assert len(group_frame) == 5
    assert result["blocked_continuous_ranking_lift_confirmation_pass"] is True
    assert result["blocked_continuous_ranking_lift_proven"] is True
    assert result["exact_continuous_value_forecasting_proven"] is False
    assert result["grouped_continuous_confirmation_authorized"] is True


def test_verdict_fails_closed_on_scope_or_audit() -> None:
    rows = synthetic_rows()
    bad_scope, _ = confirm.compute_verdict(rows.iloc[:-1], audit_pass=True)
    assert bad_scope["blocked_continuous_ranking_lift_confirmation_pass"] is False
    assert "exact_scope" in bad_scope["failed_gates"]
    bad_audit, _ = confirm.compute_verdict(rows, audit_pass=False)
    assert bad_audit["blocked_continuous_ranking_lift_confirmation_pass"] is False
    assert "audit_pass" in bad_audit["failed_gates"]
