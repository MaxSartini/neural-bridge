"""Contracts for the bounded Phase 6 selected-head Optuna pilot."""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("optuna")

from backend.scripts import (  # noqa: E402
    run_again_dense_2hz_phase6_optuna_selected_head_pilot as pilot,
)


def test_pilot_is_exactly_the_proven_target_and_head() -> None:
    assert pilot.SEED == 20260625
    assert pilot.N_TRIALS == 16
    assert pilot.ARCHITECTURE == "short_temporal_conv_residual"
    assert pilot.confirm.TARGET_NAME == "future_arousal_max_delta_rows_4_10_train_q90"
    assert pilot.ORIGINAL_PARAMS == {
        "hidden": 64,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "alpha_initial_logit": -4.0,
        "alpha_cap": 0.12,
        "gate_bias": 4.0,
        "lambda_binary": 0.5,
    }


def test_inner_objective_interface_has_no_heldout_arrays() -> None:
    assert tuple(pilot.InnerPack.__dataclass_fields__) == ("train_x", "dims")
    signature = inspect.signature(pilot.train_inner_only)
    assert "test" not in " ".join(signature.parameters).lower()


@pytest.mark.parametrize(
    ("tuned", "original", "ar", "control", "expected"),
    [
        (0.271, 0.269, 0.263, 0.268, True),
        (0.2695, 0.269, 0.263, 0.268, False),
        (0.271, 0.269, 0.2705, 0.268, False),
        (0.271, 0.269, 0.263, 0.2705, False),
    ],
)
def test_promising_gate_requires_all_three_deltas(
    tuned: float, original: float, ar: float, control: float, expected: bool
) -> None:
    assert (
        pilot.promising_followup(
            tuned_pr=tuned,
            original_pr=original,
            ar_pr=ar,
            best_control_pr=control,
        )
        is expected
    )
