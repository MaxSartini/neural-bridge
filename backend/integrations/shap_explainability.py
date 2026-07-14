"""Official SHAP explanations with MLX-accelerated prediction batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ._optional import require_upstream
from .acceleration import require_accelerator


CLAIM_BOUNDARY = (
    "Model-behavior attribution only; not causal neural evidence, not individual "
    "profiling, and not a benchmark promotion gate."
)


class MLXShapPredictor:
    """Adapter that lets official model-agnostic SHAP batch inference on MLX GPU."""

    def __init__(self, predict_mlx: Callable[[Any], Any], *, output_dtype: Any = np.float64) -> None:
        self.predict_mlx = predict_mlx
        self.output_dtype = output_dtype
        require_accelerator("mlx")

    def __call__(self, values: np.ndarray) -> np.ndarray:
        mx = require_upstream("mlx.core")
        inputs = mx.asarray(np.asarray(values), dtype=mx.float32)
        outputs = self.predict_mlx(inputs)
        mx.eval(outputs)
        return np.asarray(outputs, dtype=self.output_dtype)


@dataclass(frozen=True)
class ShapExplanationBundle:
    explanation: Any
    feature_names: tuple[str, ...]
    background_rows: int
    explained_rows: int
    predictor_backend: str
    claim_boundary: str = CLAIM_BOUNDARY


def explain_model_behavior(
    predict_mlx: Callable[[Any], Any],
    *,
    background: np.ndarray,
    samples: np.ndarray,
    feature_names: Sequence[str],
    background_is_inner_train_only: bool,
    max_background_rows: int = 128,
    max_explained_rows: int = 256,
    algorithm: str = "permutation",
) -> ShapExplanationBundle:
    """Run real ``shap.Explainer`` while all model predictions execute on MLX."""

    if not background_is_inner_train_only:
        raise ValueError("SHAP background data must come only from the inner-training split")
    background_array = np.asarray(background)
    samples_array = np.asarray(samples)
    if background_array.ndim != 2 or samples_array.ndim != 2:
        raise ValueError("background and samples must be two-dimensional")
    if background_array.shape[1] != samples_array.shape[1]:
        raise ValueError("background and samples must have the same feature width")
    names = tuple(str(name) for name in feature_names)
    if len(names) != background_array.shape[1]:
        raise ValueError("feature_names must match the feature width")
    if max_background_rows < 1 or max_explained_rows < 1:
        raise ValueError("SHAP row limits must be positive")
    bounded_background = background_array[:max_background_rows]
    bounded_samples = samples_array[:max_explained_rows]
    if not len(bounded_background) or not len(bounded_samples):
        raise ValueError("SHAP background and samples cannot be empty")

    shap = require_upstream("shap")
    predictor = MLXShapPredictor(predict_mlx)
    explainer = shap.Explainer(
        predictor,
        bounded_background,
        feature_names=list(names),
        algorithm=algorithm,
    )
    explanation = explainer(bounded_samples)
    return ShapExplanationBundle(
        explanation=explanation,
        feature_names=names,
        background_rows=len(bounded_background),
        explained_rows=len(bounded_samples),
        predictor_backend="mlx_gpu",
    )
