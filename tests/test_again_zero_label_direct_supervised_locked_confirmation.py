from __future__ import annotations

from dataclasses import fields

import pandas as pd

from backend.scripts import again_zero_label_deployment_stage_a as stage_a
from backend.scripts import (
    again_zero_label_direct_supervised_locked_confirmation as locked,
)
from backend.scripts import (
    run_again_dense_2hz_zero_label_direct_supervised_locked_confirmation as runner,
)
from backend.scripts import run_again_dense_2hz_zero_label_deployment_stage0 as stage0


def test_freeze_locks_one_candidate_seven_lanes_and_exact_140_rows() -> None:
    freeze = locked.implementation_freeze_manifest(preregistration_sha256="abc")
    assert freeze["authorization"] == locked.AUTHORIZATION
    assert freeze["candidate"] == locked.PRIMARY
    assert freeze["rows"] == {"member": 105, "ensemble": 35, "total": 140}
    assert freeze["teacher_retention_is_gate"] is False
    assert freeze["absolute_teacher_parity_required"] is False
    assert freeze["cpu_fallback"] is False
    assert tuple(freeze["lanes"]) == locked.LANES


def test_preflight_matches_stage0_target_and_prospective_split_locks() -> None:
    preflight = runner.implementation_preflight(
        stage0_root=runner.DEFAULT_STAGE0_ROOT,
        dense_root=stage0.DEFAULT_DENSE_ROOT,
        external_phase4_root=runner.DEFAULT_EXTERNAL_PHASE4_ROOT,
        output_root=runner.DEFAULT_OUTPUT_ROOT,
    )
    assert preflight["target_identity_digest"] == "446906dff30be33f204de0f973207975"
    assert preflight["development_split_digest"] == "cf65a766cd827e6201544dd753049cb4"
    assert preflight["locked_split_digest"] == "ded8bc2bf079fef91ae5c253b9a9ac2e"
    assert preflight["locked_video_count"] == 299
    assert preflight["authorized"] is True


def test_candidate_feature_schema_still_contains_no_response_inputs() -> None:
    names = {field.name for field in fields(stage_a.VideoFeatures)}
    assert not names & {"arousal", "target", "teacher_score", "ar_score", "ar_reg"}


def _synthetic_inputs(
    *, lose_metric: str | None = None, weak_bootstrap: bool = False, weak_first30: bool = False
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    base = {
        locked.PRIMARY: 0.16,
        locked.CURRENT_ROW: 0.145,
        locked.DIAGNOSTICS_ONLY: 0.12,
        locked.NO_VIDEO: 0.10,
        locked.SHUFFLED: 0.11,
        locked.PERMUTED: 0.105,
        locked.TEACHER: 0.24,
    }
    aggregate: dict[str, dict[str, float]] = {}
    for lane, value in base.items():
        aggregate[lane] = {}
        for index, metric in enumerate(locked.REQUIRED_METRICS):
            aggregate[lane][metric] = value + index * 0.01
            aggregate[lane][f"first30_{metric}"] = value + index * 0.01
    if lose_metric is not None:
        aggregate[locked.PRIMARY][lose_metric] = aggregate[locked.DIAGNOSTICS_ONLY][lose_metric] - 0.001
    if weak_first30:
        first30 = f"first30_{locked.REQUIRED_METRICS[0]}"
        aggregate[locked.PRIMARY][first30] = aggregate[locked.DIAGNOSTICS_ONLY][first30] - 0.001

    rows = []
    for panel in locked.PANELS:
        for lane in locked.LANES:
            for row_type, groups in (
                ("member", [str(seed) for seed in locked.SEEDS]),
                ("ensemble", [locked.GROUP_NAME]),
            ):
                for group in groups:
                    row = {
                        "stage": "locked_confirmation",
                        "split_digest": f"panel{panel}",
                        "panel": panel,
                        "lane": lane,
                        "row_type": row_type,
                        "seed_or_group": group,
                        "cold_start_policy": "row0_zero_history_no_label_burnin",
                    }
                    for index, metric in enumerate(locked.REQUIRED_METRICS):
                        row[metric] = base[lane] + index * 0.01
                        row[f"first30_{metric}"] = base[lane] + index * 0.01
                    rows.append(row)
    frame = pd.DataFrame(rows)
    bootstrap = {
        metric: {"lower_95_one_sided": -0.001 if weak_bootstrap else 0.005}
        for metric in locked.REQUIRED_METRICS
    }
    return frame, aggregate, bootstrap


def test_verdict_passes_all_three_tiers_when_every_endpoint_beats_controls() -> None:
    frame, aggregate, bootstrap = _synthetic_inputs()
    result = locked.compute_verdict(
        frame, aggregate=aggregate, bootstrap=bootstrap, audit_pass=True
    )
    assert result["rows_actual"] == 140
    assert result["tier1_zero_label_deployment_signal_confirmed"] is True
    assert result["tier2_high_consistency_confirmation"] is True
    assert result["tier3_first30_cold_start_confirmation"] is True
    assert result["failed_tier1_gates"] == []


def test_tier2_or_first30_technicality_does_not_erase_tier1() -> None:
    frame, aggregate, bootstrap = _synthetic_inputs(
        weak_bootstrap=True, weak_first30=True
    )
    result = locked.compute_verdict(
        frame, aggregate=aggregate, bootstrap=bootstrap, audit_pass=True
    )
    assert result["tier1_zero_label_deployment_signal_confirmed"] is True
    assert result["tier2_high_consistency_confirmation"] is False
    assert result["tier3_first30_cold_start_confirmation"] is False


def test_tier1_fails_if_primary_loses_any_required_endpoint() -> None:
    metric = locked.REQUIRED_METRICS[2]
    frame, aggregate, bootstrap = _synthetic_inputs(lose_metric=metric)
    result = locked.compute_verdict(
        frame, aggregate=aggregate, bootstrap=bootstrap, audit_pass=True
    )
    assert result["tier1_zero_label_deployment_signal_confirmed"] is False
    assert "tier1_all_endpoints_beat_false_signal_and_no_video_controls" in result[
        "failed_tier1_gates"
    ]


def test_current_row_is_mechanism_ablation_not_false_signal_control() -> None:
    assert locked.CURRENT_ROW not in locked.FALSE_SIGNAL_CONTROLS
    assert locked.DIAGNOSTICS_ONLY in locked.FALSE_SIGNAL_CONTROLS
    assert locked.NO_VIDEO in locked.FALSE_SIGNAL_CONTROLS
    assert locked.SHUFFLED in locked.FALSE_SIGNAL_CONTROLS
    assert locked.PERMUTED in locked.FALSE_SIGNAL_CONTROLS


def test_bootstrap_scope_is_fixed_before_locked_scoring() -> None:
    assert locked.BOOTSTRAP_RESAMPLES == 2000
    assert locked.BOOTSTRAP_SEED == 20260724
