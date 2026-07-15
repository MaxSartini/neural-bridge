"""MLX implementation of the released TRIBE v2 cortical brain encoder."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np


def _array_with_dtype(array: np.ndarray, dtype_name: str) -> mx.array:
    if dtype_name == "float16":
        return mx.array(array.astype(np.float16, copy=False))
    if dtype_name == "bfloat16":
        return mx.array(array.astype(np.float32, copy=False), dtype=mx.bfloat16)
    if dtype_name == "float32":
        return mx.array(array.astype(np.float32, copy=False))
    raise ValueError(f"Unsupported MLX dtype: {dtype_name}")


def _mx_dtype_for_name(dtype_name: str) -> mx.Dtype:
    if dtype_name == "float16":
        return mx.float16
    if dtype_name == "bfloat16":
        return mx.bfloat16
    if dtype_name == "float32":
        return mx.float32
    raise ValueError(f"Unsupported MLX dtype: {dtype_name}")


class MlxTribeEncoder:
    """Run converted TRIBE v2 encoder weights on precomputed feature tensors."""

    def __init__(self, model_dir: str, *, dtype: str = "float16", allow_dtype_fallback: bool = False) -> None:
        root = Path(model_dir)
        self.config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.dtype = dtype
        self.weights_path = root / f"tribev2_mlx_{dtype}.npz"
        if not self.weights_path.exists():
            if not allow_dtype_fallback:
                raise FileNotFoundError(self.weights_path)
            for fallback_name in ("tribev2_mlx_float16.npz", "tribev2_mlx_float32.npz"):
                fallback = root / fallback_name
                if fallback.exists():
                    self.weights_path = fallback
                    self.dtype = "float16" if "float16" in fallback_name else "float32"
                    break
            else:
                raise FileNotFoundError(self.weights_path)
        if self.dtype == "bfloat16":
            bundle = mx.load(str(self.weights_path))
            self.weights = {key: value.astype(mx.bfloat16) for key, value in bundle.items()}
        else:
            with np.load(self.weights_path) as bundle:
                self.weights = {key: _array_with_dtype(bundle[key], self.dtype) for key in bundle.files}
        self.hidden = int(self.config["hidden"])
        self.heads = int(self.config["heads"])
        self.head_dim = self.hidden // self.heads
        self.mx_dtype = _mx_dtype_for_name(self.dtype)

    def predict(
        self,
        features: Dict[str, np.ndarray],
        *,
        pool_outputs: bool = True,
    ) -> np.ndarray:
        """Predict cortical rows, optionally preserving the input time grid.

        ``pool_outputs=True`` retains the released TRIBE demo behavior.  Dense
        Neural Bridge inference uses ``False`` so one cortical prediction is
        retained for every upstream 2 Hz feature row.
        """
        x = self._aggregate_features(features)
        x = x + self.weights["time_pos_embed"][:, : x.shape[1]]
        for index in range(int(self.config["depth"]) * 2):
            residual = x
            normed = self._scale_norm(x, self.weights[f"encoder.layers.{index}.0.0.g"])
            if index % 2 == 0:
                out = self._attention(normed, index)
            else:
                out = self._feed_forward(normed, index)
            x = out + residual * self.weights[f"encoder.layers.{index}.2.residual_scale"]
        x = self._scale_norm(x, self.weights["encoder.final_norm.g"])
        x = self._linear(x, self.weights["low_rank_head.weight"])
        predictor = self.weights["predictor.weights"][0]
        bias = self.weights["predictor.bias"][0]
        x = mx.einsum("btc,co->bot", x, predictor) + bias[None, :, None]
        if pool_outputs:
            x = self._adaptive_avg_pool_1d(x, int(self.config["n_output_timesteps"]))
        mx.eval(x)
        return np.asarray(x, dtype=np.float32)

    def _aggregate_features(self, features: Dict[str, np.ndarray]) -> mx.array:
        first = next(iter(features.values()))
        batch, time = first.shape[0], first.shape[-1]
        projected = []
        for modality in ("text", "audio", "video"):
            if modality not in features:
                projected.append(mx.zeros((batch, time, int(self.config["projector_out"])), dtype=self.mx_dtype))
                continue
            data = _array_with_dtype(features[modality], self.dtype)
            if data.ndim == 4:
                data = mx.reshape(data, (batch, data.shape[1] * data.shape[2], time))
            data = mx.transpose(data, (0, 2, 1))
            projected.append(self._linear(
                data,
                self.weights[f"projectors.{modality}.weight"],
                self.weights[f"projectors.{modality}.bias"],
            ))
        return mx.concatenate(projected, axis=-1)

    def _attention(self, x: mx.array, index: int) -> mx.array:
        prefix = f"encoder.layers.{index}.1"
        q = self._linear(x, self.weights[f"{prefix}.to_q.weight"])
        k = self._linear(x, self.weights[f"{prefix}.to_k.weight"])
        v = self._linear(x, self.weights[f"{prefix}.to_v.weight"])
        batch, time, _ = q.shape
        q = mx.transpose(mx.reshape(q, (batch, time, self.heads, self.head_dim)), (0, 2, 1, 3))
        k = mx.transpose(mx.reshape(k, (batch, time, self.heads, self.head_dim)), (0, 2, 1, 3))
        v = mx.transpose(mx.reshape(v, (batch, time, self.heads, self.head_dim)), (0, 2, 1, 3))
        q = self._apply_rotary(q)
        k = self._apply_rotary(k)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * (self.head_dim ** -0.5)
        attended = mx.softmax(scores, axis=-1) @ v
        attended = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)), (batch, time, self.hidden))
        return self._linear(attended, self.weights[f"{prefix}.to_out.weight"])

    def _feed_forward(self, x: mx.array, index: int) -> mx.array:
        prefix = f"encoder.layers.{index}.1.ff"
        x = self._linear(
            x,
            self.weights[f"{prefix}.0.0.weight"],
            self.weights[f"{prefix}.0.0.bias"],
        )
        x = 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
        return self._linear(
            x,
            self.weights[f"{prefix}.2.weight"],
            self.weights[f"{prefix}.2.bias"],
        )

    def _apply_rotary(self, x: mx.array) -> mx.array:
        inv_freq = self.weights["encoder.rotary_pos_emb.inv_freq"]
        positions = mx.arange(x.shape[-2], dtype=mx.float32)
        half_freqs = positions[:, None] * inv_freq[None, :]
        freqs = mx.concatenate([half_freqs, half_freqs], axis=-1)
        rot_dim = freqs.shape[-1]
        rotated, untouched = x[..., :rot_dim], x[..., rot_dim:]
        half = rot_dim // 2
        rotate_half = mx.concatenate([-rotated[..., half:], rotated[..., :half]], axis=-1)
        rotated = rotated * mx.cos(freqs)[None, None] + rotate_half * mx.sin(freqs)[None, None]
        return mx.concatenate([rotated, untouched], axis=-1)

    def _scale_norm(self, x: mx.array, g: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        return x / mx.maximum(norm, mx.array(1e-12)) * math.sqrt(self.hidden) * g

    def _adaptive_avg_pool_1d(self, x: mx.array, output_size: int) -> mx.array:
        input_size = x.shape[-1]
        pooled = []
        for index in range(output_size):
            start = math.floor(index * input_size / output_size)
            end = math.ceil((index + 1) * input_size / output_size)
            pooled.append(mx.mean(x[..., start:end], axis=-1))
        return mx.stack(pooled, axis=-1)

    @staticmethod
    def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
        out = x @ mx.transpose(weight)
        return out if bias is None else out + bias
