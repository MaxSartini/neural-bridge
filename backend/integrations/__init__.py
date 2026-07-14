"""Official research-tooling integrations for future Neural Bridge work."""

from .acceleration import AcceleratorStatus, require_accelerator
from .mlflow_tracking import MLflowRun, RunProvenance
from .optuna_search import AcceleratedObjectiveResult, TrainOnlyStudySpec, run_train_only_study
from .polars_io import collect_mlx, collect_pandas, scan_table, write_table
from .shap_explainability import MLXShapPredictor, ShapExplanationBundle, explain_model_behavior

__all__ = [
    "AcceleratedObjectiveResult",
    "AcceleratorStatus",
    "MLXShapPredictor",
    "MLflowRun",
    "RunProvenance",
    "ShapExplanationBundle",
    "TrainOnlyStudySpec",
    "collect_mlx",
    "collect_pandas",
    "explain_model_behavior",
    "require_accelerator",
    "run_train_only_study",
    "scan_table",
    "write_table",
]
