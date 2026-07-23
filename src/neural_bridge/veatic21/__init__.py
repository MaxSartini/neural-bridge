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
from .preregistration import (
    EventTargetHypothesis,
    benchmark_partition_mask,
    build_event_preregistration,
    calibrate_event_preregistration,
    targets_from_calibration,
)
from .protocol import build_video_splits, freeze_final_recipe
from .runner import (
    predict_exported_model,
    refit_all_124,
    run_confirmation_cell,
    run_nested_discovery,
    verify_confirmation_cell,
)
from .stage1 import (
    CheckpointSelector,
    Stage1CellConfig,
    build_stage1_plan,
    probe_stage1_capacity,
    run_stage1_ar_benchmark,
    run_stage1_discovery_cell,
)

__all__ = [
    "AROUSAL_SPIKE_1_3S",
    "CANONICAL_DATASET",
    "CONTROL_LANES",
    "CandidateSpec",
    "CanonicalSubstrate",
    "CellSpec",
    "CheckpointSelector",
    "Stage1CellConfig",
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
    "build_stage1_plan",
    "build_event_preregistration",
    "calibrate_event_preregistration",
    "freeze_final_recipe",
    "predict_exported_model",
    "probe_stage1_capacity",
    "refit_all_124",
    "run_confirmation_cell",
    "run_stage1_ar_benchmark",
    "run_stage1_discovery_cell",
    "run_nested_discovery",
    "targets_from_calibration",
    "verify_confirmation_cell",
]
