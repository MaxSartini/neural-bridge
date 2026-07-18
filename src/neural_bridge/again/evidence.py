"""Explicit adapters that verify concluded AGAIN evidence without rerunning training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .configs import PHASE5_SELECTED_HEAD, PHASE7_BLOCKED, PHASE7_GROUPED

EvidenceName = Literal["phase5-selected", "phase7-blocked", "phase7-grouped"]

LEGACY_LANES = {
    "frozen_ar_only",
    "real_residual",
    "shuffled_pca_residual",
    "random_pca_residual",
    "label_permutation_residual",
    "train_only_video_mean_residual",
    "diagnostics_only_residual",
}


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _close(left: Any, right: Any, tolerance: float) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=tolerance))
    except (TypeError, ValueError):
        return False


def _phase5(root: Path, tolerance: float) -> dict[str, object]:
    rows_path = root / "metrics" / "selected_head_420_row_manifest.csv"
    summary_path = root / "metrics" / "selected_head_420_protocol_summary.csv"
    rows = pd.read_csv(rows_path)
    expected = pd.read_csv(summary_path)
    manifest = _load_json(root / "manifests" / "run_manifest.json")
    seeds = set(PHASE5_SELECTED_HEAD.run.seeds)
    keys = ["protocol", "fold", "seed", "lane"]
    checks = {
        "exact_rows": len(rows) == 420 == manifest.get("matrix_rows_actual"),
        "unique_matrix": not rows.duplicated(keys).any(),
        "lanes": set(rows["lane"]) == LEGACY_LANES,
        "seeds": set(rows["seed"].astype(int)) == seeds,
        "target": set(rows["target_name"]) == {PHASE5_SELECTED_HEAD.run.target.name},
        "architecture": set(rows["architecture"].dropna())
        == {PHASE5_SELECTED_HEAD.run.residual.architecture},
        "protocols": set(rows["protocol"]) == set(PHASE5_SELECTED_HEAD.run.protocols),
        "folds": set(rows.loc[rows.protocol == "grouped_video", "fold"].astype(int))
        == {1, 2, 3, 4, 5},
        "blocked_fold": set(rows.loc[rows.protocol == "blocked_temporal_70_30", "fold"].astype(int))
        == {1},
    }
    ar_groups = rows.groupby(["protocol", "fold", "seed"], dropna=False)
    checks["frozen_ar_identity"] = bool(
        (ar_groups["frozen_ar_train_checksum"].nunique(dropna=False) == 1).all()
        and (ar_groups["frozen_ar_test_checksum"].nunique(dropna=False) == 1).all()
    )

    actual = (
        rows.groupby(["protocol", "lane"])["pr_auc"]
        .agg(
            scored_rows="count",
            mean_pr_auc="mean",
            std_pr_auc="std",
            min_pr_auc="min",
            max_pr_auc="max",
        )
        .reset_index()
    )
    joined = expected.merge(actual, on=["protocol", "lane"], suffixes=("_expected", "_actual"))
    metric_columns = ("mean_pr_auc", "std_pr_auc", "min_pr_auc", "max_pr_auc")
    checks["score_parity"] = len(joined) == len(expected) == 14 and all(
        np.allclose(
            joined[f"{column}_expected"],
            joined[f"{column}_actual"],
            rtol=0.0,
            atol=tolerance,
        )
        for column in metric_columns
    )
    checks["score_row_counts"] = bool(
        (joined["scored_rows_expected"] == joined["scored_rows_actual"]).all()
    )
    grouped = actual[actual.protocol == "grouped_video"].set_index("lane")
    return {
        "endpoint": "phase5-selected",
        "verification_pass": all(checks.values()),
        "scientific_pass": manifest.get("overall_pass") is True,
        "checks": checks,
        "rows": len(rows),
        "reported": {
            "grouped_real_pr_auc": float(grouped.loc["real_residual", "mean_pr_auc"]),
            "grouped_ar_pr_auc": float(grouped.loc["frozen_ar_only", "mean_pr_auc"]),
        },
    }


def _phase7(root: Path, endpoint: EvidenceName, tolerance: float) -> dict[str, object]:
    rows = pd.read_csv(root / "metrics" / "rows.csv")
    result = _load_json(root / "metrics" / "result.json")
    grouped = endpoint == "phase7-grouped"
    frozen = PHASE7_GROUPED if grouped else PHASE7_BLOCKED
    expected_rows = 420 if grouped else 140
    member = rows[rows.row_type == "member"]
    ensemble = rows[rows.row_type == "ensemble"]
    expected_groups = {",".join(map(str, group)) for group in frozen.run.checkpoint_ensembles}
    keys = ["row_type", "group", "seed", "lane"]
    ar_group_keys = ["row_type", "group", "seed"]
    if grouped:
        keys.insert(0, "fold")
        ar_group_keys.insert(0, "fold")
    checks = {
        "exact_rows": len(rows)
        == expected_rows
        == result.get("rows_actual")
        == result.get("rows_expected"),
        "unique_matrix": not rows.duplicated(keys).any(),
        "lanes": set(rows["lane"]) == LEGACY_LANES,
        "member_seeds": set(member["seed"].astype(int)) == set(frozen.run.seeds),
        "ensemble_groups": set(ensemble["member_seeds"]) == expected_groups,
        "row_types": set(rows["row_type"]) == {"member", "ensemble"},
        "target": result.get("target") == frozen.run.target.name,
        "architecture": result.get("architecture") == frozen.run.residual.architecture,
    }
    if grouped:
        checks["folds"] = set(rows["fold"].astype(int)) == {1, 2, 3, 4, 5}
        checksum_columns = ("frozen_ar_train_checksum", "frozen_ar_test_checksum")
    else:
        checksum_columns = ("ar_test_checksum",)
    ar_groups = rows.groupby(ar_group_keys, dropna=False)
    checks["frozen_ar_identity"] = all(
        bool((ar_groups[column].nunique(dropna=False) == 1).all()) for column in checksum_columns
    )

    means = ensemble.groupby("lane").mean(numeric_only=True)
    controls = sorted(
        LEGACY_LANES - {"real_residual", "frozen_ar_only", "diagnostics_only_residual"}
    )
    calculated = {
        "real_spearman": means.loc["real_residual", "continuous_spearman"],
        "ar_spearman": means.loc["frozen_ar_only", "continuous_spearman"],
        "real_top5_lift": means.loc["real_residual", "top_5pct_continuous_lift"],
        "ar_top5_lift": means.loc["frozen_ar_only", "top_5pct_continuous_lift"],
        "best_control_spearman": means.loc[controls, "continuous_spearman"].max(),
        "best_control_top5_lift": means.loc[controls, "top_5pct_continuous_lift"].max(),
    }
    checks["score_parity"] = all(
        _close(result.get(name), value, tolerance) for name, value in calculated.items()
    )
    scientific_key = (
        "grouped_continuous_ranking_lift_pass"
        if grouped
        else "blocked_continuous_ranking_lift_confirmation_pass"
    )
    return {
        "endpoint": endpoint,
        "verification_pass": all(checks.values()),
        "scientific_pass": result.get(scientific_key) is True,
        "failed_gates": result.get("failed_gates", []),
        "checks": checks,
        "rows": len(rows),
        "reported": {name: float(value) for name, value in calculated.items()},
    }


def verify_evidence(
    endpoint: EvidenceName, root: Path, *, tolerance: float = 1e-12
) -> dict[str, object]:
    """Verify one explicitly named evidence directory; never search for a replacement."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if endpoint == "phase5-selected":
        return _phase5(root, tolerance)
    return _phase7(root, endpoint, tolerance)
