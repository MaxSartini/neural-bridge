from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from backend.scripts import run_again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped as grouped


def test_scope_is_fixed_fresh_and_420() -> None:
    assert grouped.SEEDS == tuple(range(20260708, 20260717))
    assert grouped.GROUPS == tuple(tuple(grouped.SEEDS[i:i + 3]) for i in range(0, 9, 3))
    assert grouped.EXPECTED_MEMBER_ROWS == 315
    assert grouped.EXPECTED_ENSEMBLE_ROWS == 105
    assert grouped.EXPECTED_ROWS == 420


def test_dry_run_preserves_blocked_verdict_and_requires_mlx() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(grouped.__file__).resolve()), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["rows"] == 420
    assert payload["blocked_verdict_relabelled"] is False
    assert payload["authorization"].startswith("explicit_user_authorization")
    assert payload["accelerator"] == "mlx_gpu_mps"
    assert payload["no_cpu_fallback"] is True
    assert payload["no_pca_refit"] is True


def synthetic_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = {
        "frozen_ar_only": (0.20, 0.10, 0.11, 0.09),
        "real_residual": (0.22, 0.13, 0.14, 0.12),
        "shuffled_pca_residual": (0.202, 0.102, 0.112, 0.092),
        "random_pca_residual": (0.203, 0.103, 0.113, 0.093),
        "label_permutation_residual": (0.198, 0.098, 0.108, 0.088),
        "train_only_video_mean_residual": (0.204, 0.104, 0.114, 0.094),
        "diagnostics_only_residual": (0.205, 0.105, 0.115, 0.095),
    }
    for fold in grouped.FOLDS:
        for group_id, seeds in enumerate(grouped.GROUPS, 1):
            for seed in seeds:
                for lane, (spearman, top1, top5, top10) in values.items():
                    member_offset = (((seed % 3) - 1) * 0.01) if lane == "real_residual" else 0
                    rows.append({"row_type": "member", "fold": fold, "group": group_id, "seed": seed, "lane": lane, "continuous_spearman": spearman - (0.005 if lane == "real_residual" else 0) + member_offset, "top_1pct_continuous_lift": top1, "top_5pct_continuous_lift": top5 - (0.004 if lane == "real_residual" else 0), "top_10pct_continuous_lift": top10, "continuous_mae": 0.12, "continuous_rmse": 0.15})
            offset = (fold + group_id - 4) * 0.0001
            for lane, (spearman, top1, top5, top10) in values.items():
                rows.append({"row_type": "ensemble", "fold": fold, "group": group_id, "seed": 0, "lane": lane, "continuous_spearman": spearman + offset, "top_1pct_continuous_lift": top1 + offset, "top_5pct_continuous_lift": top5 + offset, "top_10pct_continuous_lift": top10 + offset, "continuous_mae": 0.12, "continuous_rmse": 0.15})
    return pd.DataFrame(rows)


def test_verdict_passes_complete_grouped_ranking_but_not_exact_values() -> None:
    result, deltas = grouped.compute_verdict(synthetic_rows(), audit_pass=True)
    assert len(deltas) == 15
    assert result["grouped_continuous_ranking_lift_pass"] is True
    assert result["exact_continuous_value_forecasting_proven"] is False


def test_verdict_fails_closed_on_scope_or_audit() -> None:
    rows = synthetic_rows()
    bad_scope, _ = grouped.compute_verdict(rows.iloc[:-1], audit_pass=True)
    assert bad_scope["grouped_continuous_ranking_lift_pass"] is False
    assert "exact_scope" in bad_scope["failed_gates"]
    bad_audit, _ = grouped.compute_verdict(rows, audit_pass=False)
    assert bad_audit["grouped_continuous_ranking_lift_pass"] is False
    assert "audit_pass" in bad_audit["failed_gates"]
