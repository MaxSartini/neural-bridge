from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as blend


def _rows(ensemble_delta: float = 0.0) -> pd.DataFrame:
    rows = []
    originals = (0.260, 0.264, 0.258, 0.263, 0.259)
    trial4s = (0.258, 0.261, 0.264, 0.259, 0.263)
    for offset, seed in enumerate(blend.FRESH_SEEDS):
        ar = 0.250 + offset * 0.0001
        original = originals[offset]
        trial4 = trial4s[offset]
        ensemble = 0.2635 + offset * 0.00002 + ensemble_delta
        for lane, value in (
            ("frozen_ar_only", ar),
            ("original_real_residual", original),
            ("trial4_real_residual", trial4),
            ("fixed_50_50_blend", ensemble),
        ):
            rows.append({"seed": seed, "lane": lane, "pr_auc": value})
    return pd.DataFrame(rows)


def test_scope_and_weights_are_literal_and_fresh() -> None:
    assert blend.FRESH_SEEDS == (20260640, 20260641, 20260642, 20260643, 20260644)
    assert not set(blend.FRESH_SEEDS) & set(range(20260625, 20260640))
    assert blend.BLEND_WEIGHTS == {"original": 0.5, "trial4": 0.5}
    assert blend.EXPECTED_ROWS == 20


def test_gate_requires_all_stability_and_improvement_checks() -> None:
    result, _ = blend.compute_verdict(_rows(), audit_pass=True)
    assert result["pilot_pass"] is True
    failed, _ = blend.compute_verdict(_rows(ensemble_delta=-0.005), audit_pass=True)
    assert failed["pilot_pass"] is False
    assert "ensemble_mean_exceeds_higher_component_by_0_0005" in failed["failed_gates"]


def test_gate_fails_closed_on_audit_or_missing_seed() -> None:
    failed_audit, _ = blend.compute_verdict(_rows(), audit_pass=False)
    assert failed_audit["pilot_pass"] is False
    incomplete = _rows().query("seed != 20260644")
    failed_rows, _ = blend.compute_verdict(incomplete, audit_pass=True)
    assert failed_rows["pilot_pass"] is False
    assert "exact_five_fresh_seeds" in failed_rows["failed_gates"]


def test_dry_run_attests_no_weight_search_or_viewed_score_reuse() -> None:
    script = Path(blend.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["weight_search"] is False
    assert payload["viewed_seed_scores_reused"] is False
    assert payload["matched_controls_deferred_until_pilot_pass"] is True
    assert payload["accelerator"] == "mlx"


def test_plan_forbids_post_result_weight_tuning() -> None:
    plan = (blend.REPO_ROOT / "docs/phase6_fixed_blend_fresh5_pilot_plan.md").read_text(encoding="utf-8")
    assert "no weight search" in plan.lower()
    assert "do not tune the weight after seeing" in plan.lower()
    assert "canonical 420" in plan
