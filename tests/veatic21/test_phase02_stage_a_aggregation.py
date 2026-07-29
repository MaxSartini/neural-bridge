from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.phase02_stage_a_aggregation import (
    FEATURE_FORMS,
    HISTORY_DEPTHS,
    _nonlinear_candidates,
    _stage_b_candidates,
    select_one_standard_error,
    select_stratified_finalists,
)

ROOT = Path(__file__).resolve().parents[2]
BACKTEST = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "fresh-method-rebuild-20260728/phase-02-target-specific-ar/benchmark/"
    "stage-a-aggregation-executor-backtest-v2-end-to-end"
)
SELECTED = ROOT / (
    "internal/active/veatic21-phase02-registration/selected-aggregation-executor.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configuration(
    identifier: str,
    *,
    score: float,
    standard_error: float,
    brier: float | None,
    history: int = 1,
    features: int = 1,
    family: str = "continuous_ridge",
    regularization_index: int = 5,
) -> dict[str, Any]:
    return {
        "aggregate_configuration_id": identifier,
        "disposition": "eligible_for_selection",
        "mean_raw_pr_auc": score,
        "standard_error_raw_pr_auc": standard_error,
        "mean_brier": brier,
        "history_depth_rows": history,
        "feature_count": features,
        "model_family": family,
        "regularization_index": regularization_index,
        "regularization_multiplier": 10.0**regularization_index,
    }


def test_one_standard_error_uses_brier_before_capacity() -> None:
    best_ridge = _configuration("ridge", score=0.50, standard_error=0.05, brier=None)
    probabilistic = _configuration(
        "logistic",
        score=0.47,
        standard_error=0.01,
        brier=0.08,
        family="event_logistic_l2",
    )
    outside = _configuration(
        "outside",
        score=0.44,
        standard_error=0.01,
        brier=0.01,
        family="event_logistic_l2",
    )
    selected = select_one_standard_error([best_ridge, probabilistic, outside])
    assert selected["aggregate_configuration_id"] == "logistic"
    assert selected["one_standard_error_threshold"] == 0.45
    assert selected["one_standard_error_set_size"] == 2


def test_one_standard_error_never_admits_incomplete_configuration() -> None:
    invalid = _configuration("invalid", score=0.99, standard_error=0.0, brier=0.01)
    invalid["disposition"] = "excluded_incomplete_invalid_not_negative"
    complete = _configuration("complete", score=0.20, standard_error=0.0, brier=None)
    assert (
        select_one_standard_error([invalid, complete])["aggregate_configuration_id"] == "complete"
    )


def test_stratified_finalists_cover_every_form_and_history_region() -> None:
    rows: list[dict[str, Any]] = []
    for form_index, form in enumerate(FEATURE_FORMS):
        for depth in HISTORY_DEPTHS:
            representative = _configuration(
                f"config-{form}-{depth}",
                score=0.8 - form_index * 0.01 - depth * 0.0001,
                standard_error=0.001,
                brier=0.1 + form_index * 0.01,
                history=depth,
                features=depth + form_index,
                family="event_logistic_l2",
            )
            rows.append(
                {
                    "feature_set_id": f"feature-{form}-{depth}",
                    "feature_form": form,
                    "history_depth_rows": depth,
                    "history_region": ("low" if depth <= 7 else "mid" if depth <= 14 else "high"),
                    "disposition": "eligible_feature_set",
                    "representative": representative,
                }
            )
    finalists = select_stratified_finalists(rows)
    assert len(finalists) == 12
    assert {row["feature_form"] for row in finalists} == set(FEATURE_FORMS)
    assert {row["history_region"] for row in finalists} == {"low", "mid", "high"}


def test_stage_b_ofat_is_deduplicated_and_gru_is_sequence_only() -> None:
    assert len(_nonlinear_candidates("event_mlp", 43, 5_000)) == 16
    assert len(_nonlinear_candidates("event_gru", 43, 5_000)) == 13
    base_finalist = {
        "feature_count": 43,
        "feature_form": "current_only",
        "family_boundary_dispositions": [
            {
                "model_family": "continuous_ridge",
                "expansion_multiplier": 1e-7,
            },
            {
                "model_family": "event_logistic_l2",
                "expansion_multiplier": None,
            },
        ],
    }
    inner = {"train_rows": 5_000, "regularization_scale": 0.75}
    vector_candidates = _stage_b_candidates(base_finalist, inner)
    assert not any(row["family"] == "event_gru" for row in vector_candidates)
    sequence_candidates = _stage_b_candidates(
        {**base_finalist, "feature_form": "raw_sequence_with_availability_mask"},
        inner,
    )
    assert any(row["family"] == "event_gru" for row in sequence_candidates)
    assert len({str(row) for row in sequence_candidates}) == len(sequence_candidates)


def test_selected_aggregation_executor_matches_verified_end_to_end_backtest() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    result = json.loads((BACKTEST / "result.json").read_text(encoding="utf-8"))
    assert _sha256(BACKTEST / "request.json") == selected["backtest_request_sha256"]
    assert _sha256(BACKTEST / "result.json") == selected["backtest_result_sha256"]
    assert result["status"] == "PASS"
    verification = json.loads((BACKTEST / "verification.json").read_text(encoding="utf-8"))
    assert _sha256(BACKTEST / "verification.json") == selected["backtest_verification_sha256"]
    assert verification["status"] == "PASS"
    assert result["source_identity_gate"] == "PASS"
    assert result["analytic_identity_gate"] == "PASS"
    assert result["selected_aggregation_workers"] == selected["selected_aggregation_workers"] == 8
    assert result["selected_baseline_workers"] == selected["selected_baseline_workers"] == 8
    assert selected["eligible_for_main"] is True
    assert selected["schema_version"].endswith("_v2")
    assert result["outer_test_scores_opened"] is False
    assert result["cortical_values_opened"] is False
