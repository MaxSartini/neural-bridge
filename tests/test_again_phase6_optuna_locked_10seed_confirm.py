from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from backend.scripts import run_again_dense_2hz_phase6_optuna_locked_10seed_confirm as locked


def test_confirmation_scope_is_exactly_locked() -> None:
    assert len(locked.SEEDS) == 10
    assert locked.FOLLOWUP_SEEDS == locked.SEEDS[1:]
    assert locked.SEEDS[0] == 20260625
    assert len(locked.CONTROLS) == 6
    assert set(locked.PRIMARY_CONTROLS) < set(locked.CONTROLS)


def test_locked_winner_and_canonical_matrix_are_checksum_pinned() -> None:
    params, manifest = locked.load_locked_winner(locked.LOCKED_WINNER)
    canonical = locked.load_canonical_metrics(locked.CANONICAL_METRICS)
    assert params == locked.EXPECTED_PARAMS
    assert manifest["sha256"] == locked.LOCKED_WINNER_SHA256
    assert canonical.attrs["sha256"] == locked.CANONICAL_METRICS_SHA256
    assert canonical.shape[0] == 70


def test_prespecified_gate_passes_uniform_paired_improvement() -> None:
    canonical = locked.load_canonical_metrics(locked.CANONICAL_METRICS)
    tuned = canonical.copy()
    tuned.loc[tuned["control_type"] == "real_residual", "pr_auc"] += 0.002
    result, _summary, seed_df = locked.compute_result(
        tuned, canonical, provenance_pass=True, leakage_pass=True
    )
    assert len(seed_df) == 10
    assert result["locked_improvement_pass"] is True
    assert result["seeds_positive_vs_original"] == 10
    assert result["followup_seeds_positive_vs_original"] == 9


def test_dry_run_attests_no_optimization_or_retraining() -> None:
    script = Path(locked.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["rows"] == 70
    assert payload["per_seed_optimization"] is False
    assert payload["reuse_canonical_original_rows"] is True
    assert payload["reuse_all_canonical_frozen_ar_scores"] is True
    assert payload["accelerator"] == "mlx"
