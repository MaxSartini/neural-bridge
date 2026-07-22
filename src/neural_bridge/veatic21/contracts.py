"""Fresh VEATIC 2.1 scientific contracts.

Nothing in this package imports fitted material or modeling choices from AGAIN.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

Representation = Literal[
    "vjepa_temporal_mean",
    "tribe_grouped_mean",
    "tribe_cortical",
]
PcaSolver = Literal["randomized", "incremental"]

PRIMARY_REPRESENTATIONS = (
    "vjepa_temporal_mean",
    "tribe_grouped_mean",
    "tribe_cortical",
)

CONTROL_LANES = (
    "target_specific_frozen_ar",
    "sequence_shuffled",
    "random",
    "video_mean",
    "diagnostics_only",
    "label_permutation",
)


@dataclass(frozen=True)
class DatasetContract:
    video_count: int
    row_count: int
    exclusion_count: int
    row_hz: float


CANONICAL_DATASET = DatasetContract(
    video_count=124,
    row_count=20_657,
    exclusion_count=923,
    row_hz=2.0,
)


@dataclass(frozen=True)
class TargetSpec:
    """Future target whose event threshold belongs to each fitting split."""

    name: str
    label: Literal["arousal", "valence"]
    horizon_rows: tuple[int, ...]
    quantile: float
    transform: Literal["absolute", "positive"] = "absolute"

    def validate(self) -> None:
        if not self.horizon_rows or min(self.horizon_rows) <= 0:
            raise ValueError("target horizons must contain positive row offsets")
        if tuple(sorted(set(self.horizon_rows))) != self.horizon_rows:
            raise ValueError("target horizons must be unique and increasing")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("target quantile must be between zero and one")


AROUSAL_SPIKE_1_3S = TargetSpec(
    name="arousal_spike_1_3s_train_q90",
    label="arousal",
    horizon_rows=(2, 3, 4, 5, 6),
    quantile=0.90,
)


@dataclass(frozen=True)
class CandidateSpec:
    """Complete recipe varied during fresh nested discovery."""

    name: str
    representation: Representation
    pca_width: int
    regularization_c: float
    max_iter: int = 2_000
    tolerance: float = 1e-5
    pca_solver: PcaSolver = "randomized"
    pca_batch_rows: int | None = None

    def validate(self) -> None:
        if not self.name or any(char.isspace() for char in self.name):
            raise ValueError("candidate name must be a non-empty token")
        if self.representation not in PRIMARY_REPRESENTATIONS:
            raise ValueError(
                f"candidate representation must be a primary representation, "
                f"not {self.representation!r}"
            )
        if self.pca_width <= 0:
            raise ValueError("pca_width must be positive")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        if self.max_iter <= 0 or self.tolerance <= 0:
            raise ValueError("convergence settings must be positive")
        if self.pca_solver not in {"randomized", "incremental"}:
            raise ValueError("pca_solver must be randomized or incremental")
        if self.pca_solver == "incremental":
            if (
                isinstance(self.pca_batch_rows, bool)
                or not isinstance(self.pca_batch_rows, int)
                or self.pca_batch_rows < self.pca_width
            ):
                raise ValueError("incremental PCA requires pca_batch_rows >= pca_width")
        elif self.pca_batch_rows is not None:
            raise ValueError("pca_batch_rows is only valid for incremental PCA")
        if self.representation == "tribe_cortical" and self.pca_solver != "incremental":
            raise ValueError("tribe_cortical requires preregistered incremental PCA")


@dataclass(frozen=True)
class CellSpec:
    """One outer held-out-video confirmation cell."""

    target: TargetSpec
    outer_fold: int
    seed: int
    outer_folds: int = 5
    inner_folds: int = 3
    split_seed: int = 20_260_721
    promotable: bool = False

    def __post_init__(self) -> None:
        if self.promotable is not False:
            raise ValueError("VEATIC 2.1 foundation cells cannot be marked promotable")

    def validate(self) -> None:
        self.target.validate()
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("nested discovery requires at least two outer and inner folds")
        if not 0 <= self.outer_fold < self.outer_folds:
            raise ValueError("outer_fold is outside the declared fold range")


@dataclass(frozen=True)
class SubstrateIdentity:
    video_ids: tuple[str, ...]
    row_count: int
    exclusion_count: int
    row_hz: float
    vjepa_artifact_id: str
    vjepa_sha256_tree: str
    vjepa_file_count: int
    vjepa_size_bytes: int
    tribe_artifact_id: str
    tribe_sha256_tree: str
    tribe_file_count: int
    tribe_size_bytes: int
    row_plan_sha256: str
    source_tree_sha256: str
    encoder_model_sha256: str

    def validate(self, contract: DatasetContract = CANONICAL_DATASET) -> None:
        if len(self.video_ids) != contract.video_count or len(set(self.video_ids)) != len(
            self.video_ids
        ):
            raise ValueError("canonical VEATIC substrate must contain 124 unique videos")
        if self.row_count != contract.row_count:
            raise ValueError(f"expected {contract.row_count} rows, found {self.row_count}")
        if self.exclusion_count != contract.exclusion_count:
            raise ValueError(
                f"expected {contract.exclusion_count} quality exclusions, "
                f"found {self.exclusion_count}"
            )
        if not np.isclose(self.row_hz, contract.row_hz, rtol=0.0, atol=1e-12):
            raise ValueError(f"expected {contract.row_hz} Hz rows, found {self.row_hz}")
        for artifact_id in (self.vjepa_artifact_id, self.tribe_artifact_id):
            if not artifact_id.startswith("veatic-2.1-"):
                raise ValueError(f"non-VEATIC artifact is forbidden: {artifact_id}")
        for name, value in (
            ("vjepa_file_count", self.vjepa_file_count),
            ("vjepa_size_bytes", self.vjepa_size_bytes),
            ("tribe_file_count", self.tribe_file_count),
            ("tribe_size_bytes", self.tribe_size_bytes),
        ):
            if value <= 0:
                raise ValueError(f"canonical artifact {name} must be positive, found {value}")


@dataclass(frozen=True)
class FeatureRows:
    video_id: np.ndarray
    row_index: np.ndarray
    time_seconds: np.ndarray
    quality_eligible: np.ndarray
    representations: Mapping[str, np.ndarray]

    def validate(self) -> None:
        size = len(self.video_id)
        for name, values in (
            ("row_index", self.row_index),
            ("time_seconds", self.time_seconds),
            ("quality_eligible", self.quality_eligible),
        ):
            if len(values) != size:
                raise ValueError(f"{name} has {len(values)} rows; expected {size}")
        if not self.representations:
            raise ValueError("at least one representation is required")
        for name, values in self.representations.items():
            if values.ndim != 2 or len(values) != size:
                raise ValueError(f"representation {name} must be a row-aligned matrix")
            for start in range(0, size, 256):
                if not np.isfinite(values[start : start + 256]).all():
                    raise ValueError(f"representation {name} contains non-finite values")
        if not np.isfinite(self.time_seconds).all():
            raise ValueError("time_seconds contains non-finite values")

    def subset(self, mask: np.ndarray) -> FeatureRows:
        if mask.shape != (len(self.video_id),):
            raise ValueError("feature subset mask has the wrong shape")
        return FeatureRows(
            video_id=self.video_id[mask],
            row_index=self.row_index[mask],
            time_seconds=self.time_seconds[mask],
            quality_eligible=self.quality_eligible[mask],
            representations={name: values[mask] for name, values in self.representations.items()},
        )


@dataclass(frozen=True)
class LabelRows:
    video_id: np.ndarray
    row_index: np.ndarray
    time_seconds: np.ndarray
    arousal: np.ndarray
    valence: np.ndarray

    def validate(self) -> None:
        size = len(self.video_id)
        for name, values in (
            ("row_index", self.row_index),
            ("time_seconds", self.time_seconds),
            ("arousal", self.arousal),
            ("valence", self.valence),
        ):
            if len(values) != size:
                raise ValueError(f"{name} has {len(values)} rows; expected {size}")
            if name not in {"row_index"} and not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class VideoSplit:
    outer_fold: int
    train_video_ids: tuple[str, ...]
    test_video_ids: tuple[str, ...]
    inner_splits: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    digest: str


@dataclass(frozen=True)
class FrozenWinner:
    candidate: CandidateSpec
    cell: CellSpec
    split_digest: str
    selection_metric: str
    tie_break: str
    inner_scores: tuple[Mapping[str, object], ...]
    digest: str


@dataclass(frozen=True)
class FrozenRecipe:
    candidate: CandidateSpec
    discovery_digests: tuple[str, ...]
    outer_fold_count: int
    selection_metric: str
    tie_break: str
    refit_seed: int
    promotable: bool
    digest: str

    def __post_init__(self) -> None:
        if self.promotable is not False:
            raise ValueError("VEATIC 2.1 foundation recipes cannot be marked promotable")
