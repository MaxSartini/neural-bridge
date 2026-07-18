"""Frozen AGAIN endpoint definitions; these are reproduction records, not VEATIC priors."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import FUTURE_EVENT_TARGET, RESIDUAL_CONTINUOUS_TARGET
from .engine import RunConfig
from .models import ResidualConfig, TargetType

SELECTED_LANES = (
    "frozen_ar_only",
    "real",
    "shuffled",
    "random",
    "label_permutation",
    "train_video_mean",
    "diagnostics_only",
)


@dataclass(frozen=True)
class Endpoint:
    name: str
    target_type: TargetType
    representation: str
    run: RunConfig


def _residual() -> ResidualConfig:
    return ResidualConfig(
        architecture="short_temporal_conv_residual",
        sequence_window=5,
        sequence_channels=256,
    )


PHASE5_SELECTED_HEAD = Endpoint(
    name="phase5_selected_head_420_row_confirmation",
    target_type="event",
    representation="fold_safe_temporal_mean_2s_then_pca256",
    run=RunConfig(
        target=FUTURE_EVENT_TARGET,
        seeds=tuple(range(20260625, 20260635)),
        controls=SELECTED_LANES,
        head="learned_residual",
        residual=_residual(),
    ),
)

PHASE7_BLOCKED = Endpoint(
    name="phase7_blocked_continuous_checkpoint_ensemble",
    target_type="continuous",
    representation="existing_fold_safe_pca256",
    run=RunConfig(
        target=RESIDUAL_CONTINUOUS_TARGET,
        protocols=("blocked_temporal_70_30",),
        seeds=tuple(range(20260693, 20260708)),
        controls=SELECTED_LANES,
        head="learned_residual",
        residual=_residual(),
        checkpoint_ensembles=(
            (20260693, 20260694, 20260695),
            (20260696, 20260697, 20260698),
            (20260699, 20260700, 20260701),
            (20260702, 20260703, 20260704),
            (20260705, 20260706, 20260707),
        ),
    ),
)

PHASE7_GROUPED = Endpoint(
    name="phase7_grouped_continuous_checkpoint_ensemble",
    target_type="continuous",
    representation="existing_grouped_fold_safe_pca256",
    run=RunConfig(
        target=RESIDUAL_CONTINUOUS_TARGET,
        protocols=("grouped_video",),
        seeds=tuple(range(20260708, 20260717)),
        controls=SELECTED_LANES,
        head="learned_residual",
        residual=_residual(),
        checkpoint_ensembles=(
            (20260708, 20260709, 20260710),
            (20260711, 20260712, 20260713),
            (20260714, 20260715, 20260716),
        ),
    ),
)
