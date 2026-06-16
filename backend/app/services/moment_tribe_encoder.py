"""Encode TRIBE cortical trajectories with MOMENT for downstream experiments."""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class MomentTribeEncoder:
    """Create a compact temporal embedding without collapsing TRIBE to prose."""

    def __init__(self, model_dir: str, device: str = "auto", channels: int = 400):
        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self.device = self._resolve_device(device)
        self.channels = max(1, int(channels))
        self._model = None

    def encode_npz(self, input_path: str, output_path: str) -> Dict[str, object]:
        source = Path(input_path).expanduser().resolve()
        with np.load(source) as bundle:
            if "schaefer400_trajectories" in bundle:
                predictions = np.asarray(bundle["schaefer400_trajectories"], dtype=np.float32)
                source_representation = "registered_schaefer400_cortical_trajectories"
            else:
                predictions = np.asarray(bundle["predictions"], dtype=np.float32)
                source_representation = "raw_fsaverage5_predictions"

        series, temporal_mask, metadata = self.prepare_series(predictions)
        metadata["source_representation"] = source_representation
        embedding = self.encode_series(series, temporal_mask)
        amplitude_features = self.amplitude_features(predictions)

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            embedding=embedding,
            prepared_series=series,
            temporal_available_mask=temporal_mask,
            source_channel_mean=amplitude_features["mean"],
            source_channel_std=amplitude_features["std"],
            source_channel_peak_abs=amplitude_features["peak_abs"],
        )
        metadata_path = target.with_suffix(".json")
        metadata.update(
            {
                "source_path": str(source),
                "output_path": str(target),
                "model_dir": self.model_dir,
                "device": self.device,
                "embedding_shape": list(embedding.shape),
                "amplitude_features": [
                    "source_channel_mean",
                    "source_channel_std",
                    "source_channel_peak_abs",
                ],
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def prepare_series(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        array = np.nan_to_num(np.asarray(predictions, dtype=np.float32))
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected TRIBE predictions with 2 dimensions, got {array.shape}")

        # TRIBE convention is time x cortical vertices. Detect and correct a
        # transposed archive while preserving the longer cortical dimension.
        if array.shape[0] > array.shape[1]:
            array = array.T
        time_steps, vertices = array.shape
        if time_steps < 1 or vertices < 1:
            raise ValueError("TRIBE prediction archive is empty")

        channel_count = min(self.channels, vertices)
        if channel_count == vertices:
            reduced = array.T
            spatial_reduction = "none"
        else:
            vertex_groups = np.array_split(np.arange(vertices), channel_count)
            reduced = np.stack([array[:, group].mean(axis=1) for group in vertex_groups], axis=0)
            spatial_reduction = "deterministic contiguous channel bins"
        windows = []
        masks = []
        for start in range(0, max(1, time_steps), 512):
            chunk = reduced[:, start:start + 512]
            mask = np.zeros(512, dtype=np.float32)
            mask[: chunk.shape[1]] = 1.0
            padded = np.zeros((channel_count, 512), dtype=np.float32)
            padded[:, : chunk.shape[1]] = chunk
            valid = padded[:, : chunk.shape[1]]
            means = valid.mean(axis=1, keepdims=True)
            scales = valid.std(axis=1, keepdims=True)
            padded[:, : chunk.shape[1]] = (valid - means) / np.maximum(scales, 1e-6)
            windows.append(padded)
            masks.append(mask)

        series = np.stack(windows, axis=0)
        temporal_mask = np.stack(masks, axis=0)
        return series, temporal_mask, {
            "source_shape": list(predictions.shape),
            "time_steps": time_steps,
            "cortical_vertices": vertices,
            "spatial_channels": channel_count,
            "target_time_steps": 512,
            "window_count": len(windows),
            "preprocessing": f"{spatial_reduction}, native-time masked windows, per-channel z-score over valid samples",
        }

    def encode_series(self, series: np.ndarray, temporal_mask: np.ndarray) -> np.ndarray:
        import torch
        from momentfm import MOMENTPipeline

        if self._model is None:
            self._model = MOMENTPipeline.from_pretrained(
                self.model_dir,
                local_files_only=True,
                model_kwargs={"task_name": "embedding"},
            )
            self._model.init()
            self._model.to(self.device)
            self._model.eval()

        tensor = torch.from_numpy(series).to(self.device)
        mask = torch.from_numpy(temporal_mask).to(self.device)
        with torch.inference_mode():
            outputs = self._model.embed(x_enc=tensor, input_mask=mask, reduction="mean")
        return outputs.embeddings.detach().float().cpu().numpy()

    @staticmethod
    def amplitude_features(predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Preserve magnitude information removed by MOMENT normalization."""
        array = np.nan_to_num(np.asarray(predictions, dtype=np.float32))
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Expected predictions with 2 dimensions, got {array.shape}")
        if array.shape[0] > array.shape[1]:
            array = array.T
        return {
            "mean": array.mean(axis=0).astype(np.float32),
            "std": array.std(axis=0).astype(np.float32),
            "peak_abs": np.abs(array).max(axis=0).astype(np.float32),
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
