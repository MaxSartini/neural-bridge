from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from backend.scripts.assemble_again_dense_2hz_phase5_selected_head_420_confirmation import (
    ARCHITECTURE,
    FEATURE,
    LANES,
    SEEDS,
    TARGET,
    array_digest,
    compose_overall_gate,
    expected_matrix_keys,
    feature_policy_for,
    matrix_discrepancies,
    normalize_source_rows,
)


def source_frame(protocol: str, folds: tuple[int, ...], frozen_name: str) -> pd.DataFrame:
    rows = []
    for fold in folds:
        for seed in SEEDS:
            for lane in LANES:
                control = frozen_name if lane == "frozen_ar_only" else lane
                rows.append(
                    {
                        "schema_version": f"source_{protocol}_v1",
                        "target_name": TARGET,
                        "target_type": "binary",
                        "validation_protocol": protocol,
                        "fold": fold,
                        "seed": seed,
                        "architecture": ARCHITECTURE,
                        "control_type": control,
                        "feature_name": FEATURE,
                        "pr_auc": 0.2,
                    }
                )
    return pd.DataFrame(rows)


def full_matrix() -> pd.DataFrame:
    blocked = source_frame("blocked_temporal_70_30", (1,), "frozen_ar_only")
    grouped = source_frame("grouped_video", (1, 2, 3, 4, 5), "frozen_or_ar_only")
    return normalize_source_rows(blocked, grouped)


def passing_source_gates() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {"binary_pass": True, "failed_gates": []},
        {"grouped_compatibility_pass": True, "failed_gates": []},
    )


def test_expected_matrix_is_exact_selected_head_420_not_historical_504():
    keys = expected_matrix_keys()
    assert len(keys) == 420
    assert len([key for key in keys if key[0] == "blocked_temporal_70_30"]) == 70
    assert len([key for key in keys if key[0] == "grouped_video"]) == 350
    assert {key[3] for key in keys} == set(LANES)


def test_lane_normalization_preserves_source_label_and_yields_unique_matrix():
    rows = full_matrix()
    discrepancies = matrix_discrepancies(rows)
    grouped_ar = rows[(rows["protocol"] == "grouped_video") & (rows["lane"] == "frozen_ar_only")]
    assert len(rows) == 420
    assert set(grouped_ar["source_control_type"]) == {"frozen_or_ar_only"}
    assert discrepancies["matrix_completeness_pass"] is True
    assert discrepancies["matrix_uniqueness_pass"] is True
    assert discrepancies["missing_keys"] == []


def test_missing_or_duplicate_key_fails_closed():
    rows = full_matrix()
    broken = pd.concat([rows.iloc[:-1], rows.iloc[[0]]], ignore_index=True)
    discrepancies = matrix_discrepancies(broken)
    assert discrepancies["matrix_completeness_pass"] is False
    assert discrepancies["matrix_uniqueness_pass"] is False
    assert len(discrepancies["missing_keys"]) == 1
    assert len(discrepancies["duplicate_keys"]) == 2


def test_control_policy_contracts_are_semantic_not_name_only():
    real = {
        "control_type": "real_residual",
        "blocks": [{"block": "causal_pca_sequence_flat", "pca_control": "real"}],
    }
    shuffled = {
        "control_type": "shuffled_pca_residual",
        "blocks": [
            {"block": "causal_pca_sequence_flat", "pca_control": "shuffled_train_and_test_separately"}
        ],
    }
    bad_shuffled = {
        "control_type": "shuffled_pca_residual",
        "blocks": [{"block": "causal_pca_sequence_flat", "pca_control": "real"}],
    }
    diagnostics = {
        "control_type": "diagnostics_only_residual",
        "blocks": [{"block": "causal_diagnostics_sequence_flat", "control": "diagnostics_only"}],
    }
    assert feature_policy_for(real) == ("real", True)
    assert feature_policy_for(shuffled) == ("shuffled_train_and_test_separately", True)
    assert feature_policy_for(bad_shuffled) == ("real", False)
    assert feature_policy_for(diagnostics) == ("omitted_diagnostics_only", True)


def test_array_digest_matches_frozen_blake2b_contract():
    values = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    expected = hashlib.blake2b(np.ascontiguousarray(values).view(np.uint8), digest_size=16).hexdigest()
    assert array_digest(values) == expected


def test_overall_gate_requires_both_protocol_verdicts_and_all_integrity_checks():
    rows = full_matrix()
    discrepancies = matrix_discrepancies(rows)
    blocked, grouped = passing_source_gates()
    gate = compose_overall_gate(
        discrepancies=discrepancies,
        provenance_pass=True,
        frozen_ar_pass=True,
        control_policy_pass=True,
        checkpoint_pass=True,
        blocked_gate=blocked,
        grouped_gate=grouped,
    )
    assert gate["overall_selected_head_420_confirmation_pass"] is True
    grouped["grouped_compatibility_pass"] = False
    failed = compose_overall_gate(
        discrepancies=discrepancies,
        provenance_pass=True,
        frozen_ar_pass=True,
        control_policy_pass=True,
        checkpoint_pass=True,
        blocked_gate=blocked,
        grouped_gate=grouped,
    )
    assert failed["overall_selected_head_420_confirmation_pass"] is False
    assert "updated_grouped_compatibility_pass" in failed["failed_gates"]
