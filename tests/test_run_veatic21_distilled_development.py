from __future__ import annotations

import pandas as pd

from backend.scripts import run_veatic21_distilled_development as runner
from backend.scripts import veatic21_distilled_program as program


def test_contract_identity_ignores_only_runtime_metadata() -> None:
    first = {
        "schema_version": "v1",
        "created_at": "first",
        "contract_digest": "old",
        "folds": (1, 2, 3, 4),
        "cache_checksums_verified": True,
    }
    second = {
        **first,
        "created_at": "second",
        "contract_digest": "new",
        # JSON round-tripping normalizes tuples to lists.  The scientific
        # identity must remain restart-stable across that serialization.
        "folds": [1, 2, 3, 4],
    }

    assert runner.stable_contract_identity(first) != runner.stable_contract_identity(second)
    assert program.canonical_digest(runner.stable_contract_identity(first)) == program.canonical_digest(
        runner.stable_contract_identity(second)
    )


def test_contract_identity_rejects_scientific_setting_change() -> None:
    checked = {
        "schema_version": "v1",
        "created_at": "first",
        "cache_checksums_verified": True,
    }
    unchecked = {**checked, "created_at": "second", "cache_checksums_verified": False}

    assert runner.stable_contract_identity(checked) != runner.stable_contract_identity(unchecked)


def _complete_matrix() -> pd.DataFrame:
    rows = []
    seeds = (11, 12, 13)
    group = "_".join(str(seed) for seed in seeds)
    for target in ("target_a", "target_b"):
        for fold in (1, 2):
            for lane in runner.ALL_LANES:
                for seed in seeds:
                    rows.append(
                        {
                            "target": target,
                            "fold": fold,
                            "lane": lane,
                            "seed_or_group": str(seed),
                            "row_type": "member",
                        }
                    )
                rows.append(
                    {
                        "target": target,
                        "fold": fold,
                        "lane": lane,
                        "seed_or_group": group,
                        "row_type": "ensemble",
                    }
                )
    return pd.DataFrame(rows)


def test_matrix_audit_requires_exact_target_fold_lane_seed_cartesian_product() -> None:
    frame = _complete_matrix()
    complete = runner._matrix_audit(
        frame,
        expected_targets=("target_a", "target_b"),
        expected_fold_ids=(1, 2),
        seeds=(11, 12, 13),
    )
    assert complete["matrix_complete"] is True
    assert complete["expected_rows"] == len(frame)

    missing = runner._matrix_audit(
        frame.iloc[:-1].copy(),
        expected_targets=("target_a", "target_b"),
        expected_fold_ids=(1, 2),
        seeds=(11, 12, 13),
    )
    assert missing["matrix_complete"] is False
    assert len(missing["missing_cells"]) == 1


def test_matrix_audit_rejects_duplicate_cells() -> None:
    frame = _complete_matrix()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    audit = runner._matrix_audit(
        duplicated,
        expected_targets=("target_a", "target_b"),
        expected_fold_ids=(1, 2),
        seeds=(11, 12, 13),
    )
    assert audit["matrix_complete"] is False
    assert audit["duplicate_rows"] == 1
