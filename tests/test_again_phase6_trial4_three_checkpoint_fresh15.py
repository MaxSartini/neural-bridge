from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backend.scripts import run_again_dense_2hz_phase6_trial4_three_checkpoint_fresh15 as ens


def test_scope_is_disjoint_fixed_and_complete() -> None:
    assert ens.SEEDS == tuple(range(20260645, 20260660))
    assert ens.GROUPS == tuple(tuple(ens.SEEDS[i:i + 3]) for i in range(0, 15, 3))
    assert len({s for g in ens.GROUPS for s in g}) == 15
    assert ens.EXPECTED_ROWS == 60


def test_average_scores_requires_three_aligned_members() -> None:
    item = {k: np.arange(4, dtype=np.float32) for k in ("train_score", "train_reg", "test_score", "test_reg")}
    out = ens.average_scores([item, item, item])
    np.testing.assert_array_equal(out["test_score"], item["test_score"])
    try:
        ens.average_scores([item, item])
    except ValueError:
        pass
    else:
        raise AssertionError("two-member ensemble was accepted")


def test_dry_run_forbids_member_selection_and_weight_search() -> None:
    completed = subprocess.run([sys.executable, str(Path(ens.__file__).resolve()), "--dry-run"], check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["member_selection"] is False
    assert payload["heldout_weight_search"] is False
    assert payload["accelerator"] == "mlx"


def test_verdict_fails_closed_on_bad_audit() -> None:
    rows = []
    for group, seeds in enumerate(ens.GROUPS, 1):
        for seed in seeds:
            rows.extend([
                {"row_type": "member", "group": group, "seed": seed, "lane": "ar_member", "pr_auc": .25},
                {"row_type": "member", "group": group, "seed": seed, "lane": "original_member", "pr_auc": .26},
                {"row_type": "member", "group": group, "seed": seed, "lane": "trial4_member", "pr_auc": .261 + (seed % 3) * .002},
            ])
        rows.extend([
            {"row_type": "ensemble", "group": group, "seed": 0, "lane": "ar_checkpoint_ensemble", "pr_auc": .25},
            {"row_type": "ensemble", "group": group, "seed": 0, "lane": "original_checkpoint_ensemble", "pr_auc": .26},
            {"row_type": "ensemble", "group": group, "seed": 0, "lane": "trial4_checkpoint_ensemble", "pr_auc": .264},
        ])
    result, _ = ens.compute_verdict(pd.DataFrame(rows), audit_pass=False)
    assert result["pilot_pass"] is False
    assert "audit_pass" in result["failed_gates"]
