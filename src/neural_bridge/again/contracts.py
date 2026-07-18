"""Small, explicit contracts shared by the AGAIN scientific engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

BOUNDARY_POLICY = "use_annotation_covered_video_time_only"
ROW_RATE_HZ = 2.0
ROW_STEP_SECONDS = 0.5

Protocol = Literal["grouped_video", "blocked_temporal_70_30"]
TargetTransform = Literal["positive_delta", "abs_movement", "identity"]


@dataclass(frozen=True)
class TargetSpec:
    """A target whose threshold is fitted on each outer training split."""

    name: str
    value_column: str
    mask_column: str
    quantile: float = 0.90
    transform: TargetTransform = "positive_delta"


SPIKE_TARGET = TargetSpec(
    name="arousal_spike_rows_2_6_train_q90",
    value_column="future_arousal_max_delta_rows_2_6",
    mask_column="target_mask_arousal_spike_rows_2_6",
)

FUTURE_EVENT_TARGET = TargetSpec(
    name="future_arousal_max_delta_rows_4_10_train_q90",
    value_column="future_arousal_max_delta_rows_4_10",
    mask_column="target_mask_future_arousal_max_delta_rows_4_10",
)

RESIDUAL_CONTINUOUS_TARGET = TargetSpec(
    name="residual_future_max_delta_rows_4_10",
    value_column="residual_future_max_delta_rows_4_10",
    mask_column="target_mask_residual_future_max_delta_rows_4_10",
    transform="identity",
)


@dataclass(frozen=True)
class Split:
    """One leakage-checked outer evaluation split."""

    protocol: Protocol
    fold: int
    target: TargetSpec
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    threshold: float


@dataclass(frozen=True)
class FrozenArScores:
    """Target-specific AR predictions that every matched lane must reuse unchanged."""

    train_score: np.ndarray
    test_score: np.ndarray
    train_continuous: np.ndarray
    test_continuous: np.ndarray
    provenance: dict[str, object] = field(default_factory=dict)

    def validate(self, split: Split) -> None:
        expected = {
            "train_score": (self.train_score, len(split.train_idx)),
            "test_score": (self.test_score, len(split.test_idx)),
            "train_continuous": (self.train_continuous, len(split.train_idx)),
            "test_continuous": (self.test_continuous, len(split.test_idx)),
        }
        for name, (values, size) in expected.items():
            if values.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},), got {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class FoldData:
    """All arrays needed to score one fold without discovering files at runtime."""

    split: Split
    train_x: np.ndarray
    test_x: np.ndarray
    train_continuous: np.ndarray
    test_continuous: np.ndarray
    train_video_id: np.ndarray
    test_video_id: np.ndarray
    frozen_ar: FrozenArScores
    diagnostics_train: np.ndarray | None = None
    diagnostics_test: np.ndarray | None = None

    def validate(self) -> None:
        n_train, n_test = len(self.split.train_idx), len(self.split.test_idx)
        pairs = {
            "train_x": (self.train_x, n_train),
            "test_x": (self.test_x, n_test),
            "train_continuous": (self.train_continuous, n_train),
            "test_continuous": (self.test_continuous, n_test),
            "train_video_id": (self.train_video_id, n_train),
            "test_video_id": (self.test_video_id, n_test),
        }
        for name, (values, size) in pairs.items():
            if len(values) != size:
                raise ValueError(f"{name} must have {size} rows, got {len(values)}")
        if self.train_x.ndim != 2 or self.test_x.ndim != 2:
            raise ValueError("representation arrays must be two-dimensional")
        if self.train_x.shape[1] != self.test_x.shape[1]:
            raise ValueError("train/test representation widths differ")
        self.frozen_ar.validate(self.split)


def assert_again_only_output_path(path: Path) -> None:
    """Reject output paths that are ambiguous or cross the VEATIC boundary."""

    expanded = path.expanduser()
    parts = {part.lower() for part in expanded.parts}
    if "veatic" in parts:
        raise ValueError(f"AGAIN pipeline output cannot target a VEATIC path: {path}")
    if "again" not in str(expanded).lower():
        raise ValueError(f"AGAIN pipeline output path must be clearly AGAIN scoped: {path}")
