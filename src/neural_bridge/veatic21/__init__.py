"""Fresh VEATIC 2.1 foundation."""

from .contracts import (
    AROUSAL_SPIKE_1_3S,
    CANONICAL_DATASET,
    CONTROL_LANES,
    CandidateSpec,
    CellSpec,
    DatasetContract,
    FeatureRows,
    FrozenRecipe,
    FrozenWinner,
    LabelRows,
    SubstrateIdentity,
    TargetSpec,
    VideoSplit,
)
from .data import CanonicalSubstrate
from .event_screen import run_event_target_screen, targets_from_calibration
from .preregistration import (
    EventTargetHypothesis,
    benchmark_partition_mask,
    build_event_preregistration,
    calibrate_event_preregistration,
)
from .protocol import build_video_splits, freeze_final_recipe
from .runner import (
    predict_exported_model,
    refit_all_124,
    run_confirmation_cell,
    run_nested_discovery,
    verify_confirmation_cell,
)

__all__ = [
    "AROUSAL_SPIKE_1_3S",
    "CANONICAL_DATASET",
    "CONTROL_LANES",
    "CandidateSpec",
    "CanonicalSubstrate",
    "CellSpec",
    "DatasetContract",
    "EventTargetHypothesis",
    "FeatureRows",
    "FrozenRecipe",
    "FrozenWinner",
    "LabelRows",
    "SubstrateIdentity",
    "TargetSpec",
    "VideoSplit",
    "benchmark_partition_mask",
    "build_video_splits",
    "build_event_preregistration",
    "calibrate_event_preregistration",
    "freeze_final_recipe",
    "predict_exported_model",
    "refit_all_124",
    "run_confirmation_cell",
    "run_event_target_screen",
    "run_nested_discovery",
    "targets_from_calibration",
    "verify_confirmation_cell",
]
