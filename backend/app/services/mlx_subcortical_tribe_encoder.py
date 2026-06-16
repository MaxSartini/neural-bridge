"""MLX runtime for the released TRIBE v2 Lahner subcortical checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .mlx_tribe_encoder import MlxTribeEncoder
from .subcortical_roi_adapter import TRIBE_SUBCORTICAL_VOXELS


class MlxSubcorticalTribeEncoder(MlxTribeEncoder):
    """Run the full subcortical checkpoint and average across subject heads.

    The released model has ten Lahner participant heads plus one subject
    dropout head. Default population-level output averages the ten measured
    participant heads and reports disagreement as epistemic uncertainty.
    """

    FEATURE_CONTRACT = {
        "text": "Qwen/Qwen3-0.6B, layer=0.6666666666666666, token mean, contextualized",
        "audio": "facebook/w2v-bert-2.0, layer=0.6666666666666666",
        "video": "facebook/vjepa2-vitl-fpc64-256, layer=0.6666666666666666",
    }

    def __init__(self, model_dir: str, include_dropout_head: bool = False) -> None:
        root = Path(model_dir).expanduser().resolve()
        build_args = json.loads((root / "build_args.json").read_text(encoding="utf-8"))
        self.config = {
            "hidden": 1152,
            "projector_out": 384,
            "heads": 8,
            "depth": 8,
            "max_seq_len": 1024,
            "low_rank": 2048,
            "n_outputs": int(build_args["n_outputs"]),
            "n_output_timesteps": int(build_args["n_output_timesteps"]),
            "rotary_dim": 72,
        }
        if self.config["n_outputs"] != TRIBE_SUBCORTICAL_VOXELS:
            raise ValueError(
                f"Expected {TRIBE_SUBCORTICAL_VOXELS} outputs, got {self.config['n_outputs']}"
            )
        with safe_open(str(root / "best.safetensors"), framework="np") as bundle:
            self.weights = {key: mx.array(bundle.get_tensor(key)) for key in bundle.keys()}
        self.hidden = int(self.config["hidden"])
        self.heads = int(self.config["heads"])
        self.head_dim = self.hidden // self.heads
        self.include_dropout_head = bool(include_dropout_head)

    def predict(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        mean, _ = self.predict_with_uncertainty(features)
        return mean

    def predict_with_uncertainty(
        self, features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_features(features)
        x = self._encode(features)
        weights = self.weights["predictor.weights"]
        bias = self.weights["predictor.bias"]
        if not self.include_dropout_head and weights.shape[0] > 10:
            weights = weights[:10]
            bias = bias[:10]
        predictions = mx.einsum("btc,sco->sbot", x, weights) + bias[:, None, :, None]
        predictions = self._adaptive_avg_pool_1d(
            predictions, int(self.config["n_output_timesteps"])
        )
        mean = mx.mean(predictions, axis=0)
        std = mx.std(predictions, axis=0)
        mx.eval(mean, std)
        return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)

    def _encode(self, features: Dict[str, np.ndarray]) -> mx.array:
        x = self._aggregate_features(features)
        x = x + self.weights["time_pos_embed"][:, : x.shape[1]]
        for index in range(int(self.config["depth"]) * 2):
            residual = x
            normed = self._scale_norm(x, self.weights[f"encoder.layers.{index}.0.0.g"])
            out = self._attention(normed, index) if index % 2 == 0 else self._feed_forward(normed, index)
            x = out + residual * self.weights[f"encoder.layers.{index}.2.residual_scale"]
        x = self._scale_norm(x, self.weights["encoder.final_norm.g"])
        return self._linear(x, self.weights["low_rank_head.weight"])

    @staticmethod
    def _validate_features(features: Dict[str, np.ndarray]) -> None:
        if not features:
            raise ValueError("No subcortical TRIBE features supplied")
        for modality, values in features.items():
            array = np.asarray(values)
            if modality not in {"text", "audio", "video"}:
                raise ValueError(f"Unsupported modality: {modality}")
            if array.ndim not in {3, 4}:
                raise ValueError(f"Expected 3D/4D {modality} features, got {array.shape}")
            feature_dim = int(array.shape[-2])
            if feature_dim != 1024:
                raise ValueError(
                    f"Expected {modality} feature dimension 1024 under the released "
                    f"subcortical contract, got {feature_dim}"
                )
