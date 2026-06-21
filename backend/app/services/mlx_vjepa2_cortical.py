"""MLX V-JEPA2 ViT-G video extractor for TRIBE cortical features."""

import hashlib
import gc
import json
import logging
import math
import os
import subprocess
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pydantic
from exca import MapInfra
from tqdm import tqdm

from neuralset import base as nsbase
from neuralset.events import etypes as evts
from neuralset.extractors.video import (
    HuggingFaceVideo,
    _VideoImage,
)

logger = logging.getLogger(__name__)


def _video_window_checkpoint(
    event: evts.Video,
    cache_name: str,
    num_frames: int,
    frequency: float,
    clip_duration: float,
    frame_sampler: str,
) -> tuple[Path, Path]:
    """Return stable local cache paths for resumable MLX video-window encoding."""
    cache_root = Path(
        os.environ.get("TRIBE_VIDEO_WINDOW_CACHE_DIR", ".cache/tribev2/video_windows")
    ).expanduser()
    event_path = getattr(event, "filepath", "") or event.study_relative_path()
    payload = {
        "event": str(event_path),
        "offset": float(getattr(event, "offset", 0.0)),
        "duration": float(event.duration),
        "cache_name": cache_name,
        "num_frames": int(num_frames),
        "frequency": float(frequency),
        "clip_duration": float(clip_duration),
        "frame_sampler": str(frame_sampler),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{digest}.npy", cache_root / f"{digest}.progress.json"


def _event_video_path(event: evts.Video) -> Path | None:
    raw_path = getattr(event, "filepath", "") or getattr(event, "path", "")
    if not raw_path:
        try:
            raw_path = event.study_relative_path()
        except Exception:
            raw_path = ""
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    return path if path.exists() else None


def _ffmpeg_square_filter(image_size: int) -> str:
    short_side = int(256.0 / 224.0 * image_size)
    scale = (
        f"scale='if(gt(iw,ih),-2,{short_side})':"
        f"'if(gt(iw,ih),{short_side},-2)'"
    )
    return f"{scale},crop={image_size}:{image_size}"


def _decode_video_grid_ffmpeg(
    video_path: Path,
    *,
    fps: float,
    image_size: int,
) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"ffmpeg frame sampler requires positive fps, got {fps}")
    vf = f"fps={fps:.8f},{_ffmpeg_square_filter(image_size)}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-hwaccel",
        "videotoolbox",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg frame decode failed for {video_path}: {stderr.strip()}")
    frame_bytes = image_size * image_size * 3
    if len(proc.stdout) < frame_bytes:
        raise RuntimeError(f"ffmpeg returned no complete frames for {video_path}")
    if len(proc.stdout) % frame_bytes:
        raise RuntimeError(
            f"ffmpeg raw frame byte count is not divisible by frame size for {video_path}"
        )
    frame_count = len(proc.stdout) // frame_bytes
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(frame_count, image_size, image_size, 3).copy()


def _sample_decoded_grid(frames: np.ndarray, *, fps: float, times: list[float]) -> np.ndarray:
    if not len(frames):
        raise ValueError("Cannot sample an empty decoded frame grid")
    indices = np.rint(np.asarray(times, dtype=np.float64) * float(fps)).astype(np.int64)
    indices = np.clip(indices, 0, len(frames) - 1)
    return frames[indices]


@dataclass(frozen=True)
class MlxVjepa2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    mlp_ratio: float
    patch_size: int
    tubelet_size: int
    crop_size: int
    frames_per_clip: int
    in_chans: int = 3
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "MlxVjepa2Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            mlp_ratio=float(raw.get("mlp_ratio", 4.0)),
            patch_size=int(raw.get("patch_size", 16)),
            tubelet_size=int(raw.get("tubelet_size", 2)),
            crop_size=int(raw.get("image_size", raw.get("crop_size", 256))),
            frames_per_clip=int(raw.get("frames_per_clip", 64)),
            in_chans=int(raw.get("num_channels", raw.get("in_chans", 3))),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-6)),
            qkv_bias=bool(raw.get("qkv_bias", True)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def grid_size(self) -> int:
        return self.crop_size // self.patch_size

    @property
    def rope_dims(self) -> tuple[int, int, int]:
        axis = 2 * ((self.head_dim // 3) // 2)
        return axis, axis, axis


def _position_ids(num_tokens: int, grid_size: int) -> tuple[mx.array, mx.array, mx.array]:
    ids = mx.arange(num_tokens)
    tokens_per_frame = grid_size * grid_size
    frame = ids // tokens_per_frame
    height = (ids - tokens_per_frame * frame) // grid_size
    width = (ids - tokens_per_frame * frame) - grid_size * height
    return frame, height, width


def _rotate_queries_or_keys(x: mx.array, pos: mx.array) -> mx.array:
    dim = x.shape[-1]
    omega = mx.arange(dim // 2).astype(mx.float32) / (dim / 2.0)
    omega = 1.0 / (10000.0**omega)
    freq = pos.astype(mx.float32)[..., None] * omega
    sin = mx.concatenate([mx.sin(freq), mx.sin(freq)], axis=-1)
    cos = mx.concatenate([mx.cos(freq), mx.cos(freq)], axis=-1)

    pair = x.reshape(*x.shape[:-1], dim // 2, 2)
    rotated = mx.stack([-pair[..., 1], pair[..., 0]], axis=-1).reshape(x.shape)
    return x * cos + rotated * sin


def _apply_rotary_embeddings(qk: mx.array, pos_ids: tuple[mx.array, mx.array, mx.array], dims: tuple[int, int, int]) -> mx.array:
    d_dim, h_dim, w_dim = dims
    pos_d, pos_h, pos_w = pos_ids
    start = 0
    parts = [
        _rotate_queries_or_keys(qk[..., start:start + d_dim], pos_d),
    ]
    start += d_dim
    parts.append(_rotate_queries_or_keys(qk[..., start:start + h_dim], pos_h))
    start += h_dim
    parts.append(_rotate_queries_or_keys(qk[..., start:start + w_dim], pos_w))
    start += w_dim
    if start < qk.shape[-1]:
        parts.append(qk[..., start:])
    return mx.concatenate(parts, axis=-1)


class MlxVjepa2PatchEmbeddings3D(nn.Module):
    def __init__(self, config: MlxVjepa2Config) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            config.in_chans,
            config.hidden_size,
            kernel_size=(config.tubelet_size, config.patch_size, config.patch_size),
            stride=(config.tubelet_size, config.patch_size, config.patch_size),
        )

    def __call__(self, video_btchw: mx.array) -> mx.array:
        x = video_btchw.transpose(0, 1, 3, 4, 2)
        x = self.proj(x)
        batch, depth, height, width, hidden = x.shape
        return x.reshape(batch, depth * height * width, hidden)


class MlxVjepa2Embeddings(nn.Module):
    def __init__(self, config: MlxVjepa2Config) -> None:
        super().__init__()
        self.patch_embeddings = MlxVjepa2PatchEmbeddings3D(config)

    def __call__(self, video_btchw: mx.array) -> mx.array:
        return self.patch_embeddings(video_btchw)


class MlxVjepa2Attention(nn.Module):
    def __init__(self, config: MlxVjepa2Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        batch, tokens, _ = x.shape

        def split_heads(value: mx.array) -> mx.array:
            return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = split_heads(self.query(x))
        k = split_heads(self.key(x))
        v = split_heads(self.value(x))
        pos_ids = _position_ids(tokens, self.config.grid_size)
        q = _apply_rotary_embeddings(q, pos_ids, self.config.rope_dims)
        k = _apply_rotary_embeddings(k, pos_ids, self.config.rope_dims)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        out = out.transpose(0, 2, 1, 3).reshape(batch, tokens, self.config.hidden_size)
        return self.proj(out)


class MlxVjepa2Mlp(nn.Module):
    def __init__(self, config: MlxVjepa2Config) -> None:
        super().__init__()
        intermediate = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, intermediate)
        self.fc2 = nn.Linear(intermediate, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class MlxVjepa2Layer(nn.Module):
    def __init__(self, config: MlxVjepa2Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = MlxVjepa2Attention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MlxVjepa2Mlp(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MlxVjepa2Encoder(nn.Module):
    def __init__(self, config: MlxVjepa2Config) -> None:
        super().__init__()
        self.embeddings = MlxVjepa2Embeddings(config)
        self.layer = [MlxVjepa2Layer(config) for _ in range(config.num_hidden_layers)]
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def selected_token_mean_states(self, video_btchw: mx.array, selected: list[int]) -> mx.array:
        selected_set = set(selected)
        captured: dict[int, mx.array] = {}
        hidden = self.embeddings(video_btchw)
        if 0 in selected_set:
            captured[0] = mx.mean(hidden, axis=1)
        for index, layer in enumerate(self.layer, start=1):
            hidden = layer(hidden)
            if index in selected_set:
                state = self.layernorm(hidden) if index == len(self.layer) else hidden
                captured[index] = mx.mean(state, axis=1)
        return mx.stack([captured[index] for index in selected], axis=1)[:, :, None, :]


class MlxVjepa2FeatureModel:
    def __init__(self, weights_dir: str, processor_model_name: str | None = None) -> None:
        root = Path(weights_dir).expanduser().resolve()
        self.weights_dir = root
        self.config = MlxVjepa2Config.from_json(root / "config.json")
        self.encoder = MlxVjepa2Encoder(self.config)
        weights = mx.load(str(root / "model.safetensors"))
        weights = {
            key[len("encoder."):]: value
            for key, value in weights.items()
            if key.startswith("encoder.")
        }
        self.encoder.load_weights(list(weights.items()), strict=True)
        mx.eval(self.encoder.parameters())

        from transformers import AutoVideoProcessor

        self.processor = AutoVideoProcessor.from_pretrained(
            processor_model_name or str(root),
            do_rescale=True,
        )
        self.num_frames = self.config.frames_per_clip

    def predict_hidden_states(self, images: np.ndarray, selected_indices: list[int]) -> np.ndarray:
        kwargs: dict[str, tp.Any] = {"videos": list(images), "return_tensors": "pt"}
        inputs = self.processor(**kwargs)
        tensor = inputs["pixel_values_videos"]
        if tensor.isnan().any():
            tensor[tensor.isnan()] = 0
        video = mx.array(tensor.detach().cpu().numpy().astype(np.float32, copy=False))
        states = self.encoder.selected_token_mean_states(video, selected_indices)
        mx.eval(states)
        return np.asarray(states, dtype=np.float32)


class MlxVjepa2Video(HuggingFaceVideo):
    """Neuralset video extractor using MLX for cortical V-JEPA2 ViT-G."""

    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=128,
        version="v5-mlx-vjepa2",
    )
    mlx_weights_dir: str = "models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256"
    processor_model_name: str | None = None
    cache_model_name: str | None = None
    frame_sampler: str = "moviepy"
    clear_cache_each_window: bool = True
    clear_cache_each_video: bool = True
    _model: MlxVjepa2FeatureModel | None = pydantic.PrivateAttr(default=None)

    @property
    def model(self) -> MlxVjepa2FeatureModel:
        if getattr(self, "_model", None) is None:
            self._model = MlxVjepa2FeatureModel(self.mlx_weights_dir, self.processor_model_name)
        return self._model

    def _selected_hidden_state_indices(self) -> list[int]:
        n_states = self.model.config.num_hidden_layers + 1
        cache_n_layers = self.image.cache_n_layers
        if cache_n_layers is not None:
            return [int(round(index)) for index in np.linspace(0, n_states - 1, cache_n_layers)]
        layers = self.image.layers if isinstance(self.image.layers, list) else [self.image.layers]
        return [int(float(layer) * (n_states - 1)) for layer in layers]

    @infra.apply(
        item_uid=lambda event: f"{event.study_relative_path()}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[evts.Video]) -> tp.Iterator[nsbase.TimedArray]:
        if not events:
            return
        model = self.model
        selected_indices = self._selected_hidden_state_indices()
        freq = events[0].frequency if self.frequency == "native" else self.frequency
        clip_duration = 1 / freq if self.clip_duration is None else float(self.clip_duration)
        subtimes = [index / model.num_frames * clip_duration for index in reversed(range(model.num_frames))]
        cache_name = self.cache_model_name or f"mlx:{Path(self.mlx_weights_dir).expanduser().resolve()}"

        frame_sampler = (self.frame_sampler or "moviepy").lower()
        for event in events:
            video = None
            video_duration = float(getattr(event, "duration", 0.0))
            event_offset = float(getattr(event, "offset", getattr(event, "start", 0.0)) or 0.0)
            decoded_grid: np.ndarray | None = None
            decoded_grid_fps = float(model.num_frames / clip_duration)
            using_ffmpeg_grid = False
            if frame_sampler == "ffmpeg":
                video_path = _event_video_path(event)
                square_size = int(getattr(self, "image_size", 0) or 0)
                if video_path is not None and square_size > 0:
                    try:
                        decoded_grid = _decode_video_grid_ffmpeg(
                            video_path,
                            fps=decoded_grid_fps,
                            image_size=square_size,
                        )
                        using_ffmpeg_grid = True
                    except Exception as exc:  # noqa: BLE001 - fallback path preserves extraction
                        logger.warning("Falling back to MoviePy video frame sampling: %s", exc)
                        decoded_grid = None
                else:
                    logger.warning("Falling back to MoviePy video frame sampling: no video path or image size")
            if decoded_grid is None:
                video = event.read()
                video_duration = float(video.duration)
            freq = self.frequency if self.frequency != "native" else event.frequency
            expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
            times = np.linspace(0, video_duration, expect_frames + 1)[1:]
            data_path, progress_path = _video_window_checkpoint(
                event,
                cache_name,
                model.num_frames,
                float(freq),
                float(clip_duration),
                frame_sampler if using_ffmpeg_grid else "moviepy",
            )
            output: np.ndarray = np.array([])
            next_index = 0
            if data_path.exists() and progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    candidate = np.load(data_path, mmap_mode="r+")
                    if candidate.shape[0] == len(times):
                        output = candidate
                        next_index = min(int(progress.get("next_index", 0)), len(times))
                        logger.info("Resuming MLX V-JEPA2 video extraction at %s/%s", next_index, len(times))
                except (OSError, ValueError, json.JSONDecodeError):
                    output = np.array([])
                    next_index = 0

            window_batch_size = max(1, int(os.environ.get("TRIBE_VIDEO_WINDOW_BATCH_SIZE", "1")))
            progress_label = (
                "Encoding video with MLX V-JEPA 2.1"
                if self.__class__.__name__ == "MlxVjepa21Video"
                else "Encoding video with MLX V-JEPA2"
            )
            progress_bar = tqdm(total=len(times), desc=progress_label)
            progress_bar.update(next_index)
            for start_index in range(next_index, len(times), window_batch_size):
                batch_indices = list(range(start_index, min(start_index + window_batch_size, len(times))))
                batch_items = []
                for item_index in batch_indices:
                    timepoint = times[item_index]
                    frame_times = [max(0, timepoint - delta) for delta in subtimes]
                    if decoded_grid is not None:
                        source_frame_times = [event_offset + frame_time for frame_time in frame_times]
                        frames = _sample_decoded_grid(
                            decoded_grid,
                            fps=decoded_grid_fps,
                            times=source_frame_times,
                        )
                        batch_items.append((item_index, frames))
                        continue
                    frames = [_VideoImage(video=video, time=frame_time).read() for frame_time in frame_times]
                    if frames and self.max_imsize is not None:
                        factor = max(frames[0].size) / self.max_imsize
                        if factor > 1:
                            size = tuple(int(size_part / factor) for size_part in frames[0].size)
                            frames = [frame.resize(size) for frame in frames]
                    batch_items.append((item_index, np.asarray([np.asarray(frame) for frame in frames])))

                if len(batch_items) == 1:
                    encoded = model.predict_hidden_states(batch_items[0][1], selected_indices)
                else:
                    encoded = model.predict_hidden_states(
                        np.stack([item[1] for item in batch_items], axis=0),
                        selected_indices,
                    )
                embds = []
                for batch_index in range(encoded.shape[0]):
                    embd = self.image._aggregate_tokens(encoded[batch_index])
                    embds.append(np.asarray(embd, dtype=np.float32))
                if not output.size:
                    output = np.lib.format.open_memmap(
                        data_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(times),) + embds[0].shape,
                    )
                    logger.debug("Created MLX V-JEPA2 cache tensor with size %s", output.shape)
                for (item_index, _), embd in zip(batch_items, embds):
                    output[item_index] = embd
                if isinstance(output, np.memmap):
                    output.flush()
                progress_path.write_text(
                    json.dumps({"next_index": batch_indices[-1] + 1, "total": len(times)}),
                    encoding="utf-8",
                )
                progress_bar.update(len(batch_items))
                if self.clear_cache_each_window:
                    mx.clear_cache()
            progress_bar.close()
            if video is not None:
                video.close()
            if self.clear_cache_each_video:
                mx.clear_cache()
                gc.collect()
            output = output.transpose(list(range(1, output.ndim)) + [0])
            yield nsbase.TimedArray(
                data=output.astype(np.float32),
                frequency=freq,
                start=nsbase._UNSET_START,
                duration=event.duration,
            )
