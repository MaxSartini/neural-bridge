"""MLX V-JEPA 2.1 ViT-g video extractor for TRIBE cortical features.

The V-JEPA 2.1 encoder architecture and key layout differ from the older
V-JEPA2 Hugging Face-style model used by ``mlx_vjepa2_cortical``. This module
keeps the existing Neural Bridge extractor contract but loads converted
``vjepa2_1_mlx_port`` weights with V-JEPA 2.1 keys such as ``blocks.*.attn.qkv``.

Portions of the encoder structure mirror the MIT-licensed
lukasugar/vjepa2.1-mlx implementation, adapted here to return selected
token-mean hidden states for TRIBE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import typing as tp

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pydantic
from PIL import Image

from .mlx_vjepa2_cortical import MlxVjepa2Video


IMAGENET_DEFAULT_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_DEFAULT_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class MlxVjepa21Config:
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
    use_rope: bool = True
    interpolate_rope: bool = True
    modality_embedding: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "MlxVjepa21Config":
        raw = json.loads(path.read_text(encoding="utf-8"))
        tensor_layout = raw.get("tensor_layout")
        if tensor_layout != "vjepa2_1_mlx_port":
            raise ValueError(f"Expected V-JEPA 2.1 MLX tensor layout, observed {tensor_layout!r}")
        return cls(
            hidden_size=int(raw.get("hidden_size", raw.get("embed_dim"))),
            num_hidden_layers=int(raw.get("num_hidden_layers", raw.get("depth"))),
            num_attention_heads=int(raw.get("num_attention_heads", raw.get("num_heads"))),
            mlp_ratio=float(raw.get("mlp_ratio", 48 / 11)),
            patch_size=int(raw.get("patch_size", 16)),
            tubelet_size=int(raw.get("tubelet_size", 2)),
            crop_size=int(raw.get("image_size", raw.get("crop_size", raw.get("img_size", 384)))),
            frames_per_clip=int(raw.get("frames_per_clip", raw.get("num_frames", 64))),
            in_chans=int(raw.get("num_channels", raw.get("in_chans", 3))),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-6)),
            qkv_bias=bool(raw.get("qkv_bias", True)),
            use_rope=bool(raw.get("use_rope", True)),
            interpolate_rope=bool(raw.get("interpolate_rope", True)),
            modality_embedding=bool(raw.get("modality_embedding", True)),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _resize_and_center_crop(image: Image.Image, image_size: int) -> Image.Image:
    short_side = int(256.0 / 224.0 * image_size)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size: {image.size}")
    scale = short_side / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    return resized.crop((left, top, left + image_size, top + image_size))


def _normalize_chw(chw_image: np.ndarray) -> np.ndarray:
    return (chw_image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            high = 255.0 if float(np.nanmax(array)) > 1.5 else 1.0
            array = np.clip(array / high, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got shape={array.shape}")
    return Image.fromarray(array, mode="RGB")


def _preprocess_video_batch(images: np.ndarray, image_size: int) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim == 4:
        array = array[None, ...]
    if array.ndim != 5:
        raise ValueError(f"expected [T,H,W,C] or [B,T,H,W,C] images, got shape={array.shape}")
    if array.shape[2] == image_size and array.shape[3] == image_size:
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                high = 255.0 if float(np.nanmax(array)) > 1.5 else 1.0
                array = np.clip(array / high, 0.0, 1.0) * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        video = array.astype(np.float32, copy=False) / 255.0
        video = np.transpose(video, (0, 4, 1, 2, 3))
        mean = IMAGENET_DEFAULT_MEAN[None, :, None, :, :]
        std = IMAGENET_DEFAULT_STD[None, :, None, :, :]
        return ((video - mean) / std).astype(np.float32, copy=False)
    processed_videos = []
    for video in array:
        processed_frames = []
        for frame in video:
            image = _resize_and_center_crop(_frame_to_pil(frame), image_size)
            hwc = np.asarray(image, dtype=np.float32) / 255.0
            chw = np.transpose(hwc, (2, 0, 1))
            processed_frames.append(_normalize_chw(chw))
        processed_videos.append(np.stack(processed_frames, axis=1))
    return np.stack(processed_videos, axis=0).astype(np.float32, copy=False)


def _repeat_interleave_last_dim(x: mx.array, repeats: int) -> mx.array:
    x = mx.expand_dims(x, axis=-1)
    x = mx.repeat(x, repeats, axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def _rotate_queries_or_keys(x: mx.array, pos: mx.array) -> mx.array:
    *_, dim = x.shape
    if dim % 2:
        raise ValueError("RoPE dim must be even")
    omega = mx.arange(dim // 2, dtype=x.dtype) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    freq = mx.expand_dims(pos, axis=-1) * omega
    emb_sin = _repeat_interleave_last_dim(mx.sin(freq), 2)
    emb_cos = _repeat_interleave_last_dim(mx.cos(freq), 2)
    pair = x.reshape(*x.shape[:-1], -1, 2)
    rotated = mx.stack((-pair[..., 1], pair[..., 0]), axis=-1).reshape(x.shape)
    return x * emb_cos + rotated * emb_sin


_ROPE_EMBEDDING_CACHE: dict[tuple[str, int, int, int, int, int, str], tuple[mx.array, mx.array]] = {}


def _rotate_with_cached_embeddings(x: mx.array, emb_sin: mx.array, emb_cos: mx.array) -> mx.array:
    *_, dim = x.shape
    pair = x.reshape(*x.shape[:-1], -1, 2)
    rotated = mx.stack((-pair[..., 1], pair[..., 0]), axis=-1).reshape(x.shape)
    return x * emb_cos + rotated * emb_sin


class MlxVjepa21PatchEmbed3D(nn.Module):
    def __init__(self, config: MlxVjepa21Config, *, tubelet_size: int | None = None) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=config.in_chans,
            out_channels=config.hidden_size,
            kernel_size=(tubelet_size or config.tubelet_size, config.patch_size, config.patch_size),
            stride=(tubelet_size or config.tubelet_size, config.patch_size, config.patch_size),
        )

    def __call__(self, video_bcthw: mx.array) -> mx.array:
        x = video_bcthw.transpose(0, 2, 3, 4, 1)
        x = self.proj(x)
        batch = x.shape[0]
        return x.reshape(batch, -1, x.shape[-1])


class MlxVjepa21Mlp(nn.Module):
    def __init__(self, config: MlxVjepa21Config) -> None:
        super().__init__()
        intermediate = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, intermediate)
        self.fc2 = nn.Linear(intermediate, config.hidden_size)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class MlxVjepa21RoPEAttention(nn.Module):
    def __init__(self, config: MlxVjepa21Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        axis = int(2 * ((self.head_dim // 3) // 2))
        self.d_dim = axis
        self.h_dim = axis
        self.w_dim = axis
        self.pretrained_grid_size = 16 if config.patch_size == 16 else config.crop_size // config.patch_size

    def _separate_positions(
        self,
        ids: mx.array,
        h_patches: int,
        w_patches: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        tokens_per_frame = int(h_patches * w_patches)
        frame_ids = ids // tokens_per_frame
        local = ids - tokens_per_frame * frame_ids
        height_ids = local // w_patches
        width_ids = local - w_patches * height_ids
        return frame_ids.astype(mx.float32), height_ids.astype(mx.float32), width_ids.astype(mx.float32)

    def _rope_embeddings(
        self,
        axis: str,
        dim: int,
        temporal: int,
        h_patches: int,
        w_patches: int,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, mx.array]:
        key = (
            axis,
            int(dim),
            int(temporal),
            int(h_patches),
            int(w_patches),
            int(self.pretrained_grid_size),
            str(dtype),
        )
        cached = _ROPE_EMBEDDING_CACHE.get(key)
        if cached is not None:
            return cached
        mask = mx.arange(int(temporal * h_patches * w_patches), dtype=mx.int32)
        d_mask, h_mask, w_mask = self._separate_positions(mask, h_patches, w_patches)
        if self.config.interpolate_rope:
            if h_patches > 1:
                h_mask = h_mask * (self.pretrained_grid_size - 1) / (h_patches - 1)
            if w_patches > 1:
                w_mask = w_mask * (self.pretrained_grid_size - 1) / (w_patches - 1)
        pos = {"d": d_mask, "h": h_mask, "w": w_mask}[axis]
        omega = mx.arange(dim // 2, dtype=dtype) / (dim / 2.0)
        omega = 1.0 / (10000**omega)
        freq = mx.expand_dims(pos, axis=-1).astype(dtype) * omega
        emb_sin = _repeat_interleave_last_dim(mx.sin(freq), 2)
        emb_cos = _repeat_interleave_last_dim(mx.cos(freq), 2)
        cached = (emb_sin, emb_cos)
        _ROPE_EMBEDDING_CACHE[key] = cached
        return cached

    def __call__(self, x: mx.array, *, temporal: int, h_patches: int, w_patches: int) -> mx.array:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        split = 0
        sin, cos = self._rope_embeddings("d", self.d_dim, temporal, h_patches, w_patches, q.dtype)
        qd = _rotate_with_cached_embeddings(q[..., split : split + self.d_dim], sin, cos)
        kd = _rotate_with_cached_embeddings(k[..., split : split + self.d_dim], sin, cos)
        split += self.d_dim
        sin, cos = self._rope_embeddings("h", self.h_dim, temporal, h_patches, w_patches, q.dtype)
        qh = _rotate_with_cached_embeddings(q[..., split : split + self.h_dim], sin, cos)
        kh = _rotate_with_cached_embeddings(k[..., split : split + self.h_dim], sin, cos)
        split += self.h_dim
        sin, cos = self._rope_embeddings("w", self.w_dim, temporal, h_patches, w_patches, q.dtype)
        qw = _rotate_with_cached_embeddings(q[..., split : split + self.w_dim], sin, cos)
        kw = _rotate_with_cached_embeddings(k[..., split : split + self.w_dim], sin, cos)
        split += self.w_dim
        if split < self.head_dim:
            q = mx.concatenate([qd, qh, qw, q[..., split:]], axis=-1)
            k = mx.concatenate([kd, kh, kw, k[..., split:]], axis=-1)
        else:
            q = mx.concatenate([qd, qh, qw], axis=-1)
            k = mx.concatenate([kd, kh, kw], axis=-1)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(batch, tokens, channels)
        return self.proj(out)


class MlxVjepa21Block(nn.Module):
    def __init__(self, config: MlxVjepa21Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MlxVjepa21RoPEAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MlxVjepa21Mlp(config)

    def __call__(self, x: mx.array, *, temporal: int, h_patches: int, w_patches: int) -> mx.array:
        x = x + self.attn(self.norm1(x), temporal=temporal, h_patches=h_patches, w_patches=w_patches)
        return x + self.mlp(self.norm2(x))


class MlxVjepa21Encoder(nn.Module):
    def __init__(self, config: MlxVjepa21Config) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = MlxVjepa21PatchEmbed3D(config)
        self.patch_embed_img = MlxVjepa21PatchEmbed3D(config, tubelet_size=1)
        self.blocks = [MlxVjepa21Block(config) for _ in range(config.num_hidden_layers)]
        self.norms_block = [nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(4)]
        if config.modality_embedding:
            self.img_mod_embed = mx.zeros((1, 1, config.hidden_size))
            self.video_mod_embed = mx.zeros((1, 1, config.hidden_size))

    @staticmethod
    def _token_mean(x: mx.array) -> mx.array:
        return mx.mean(x.astype(mx.float32), axis=1)

    @staticmethod
    def _state_summaries(
        state: mx.array,
        *,
        batch: int,
        temporal: int,
        spatial_tokens: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Reduce one hidden state using the dense-cache temporal contract."""
        shaped = mx.reshape(
            state.astype(mx.float32),
            (batch, temporal, spatial_tokens, state.shape[-1]),
        )
        temporal_mean = mx.mean(shaped, axis=2)
        centered = shaped - temporal_mean[:, :, None, :]
        temporal_std = mx.sqrt(mx.mean(mx.square(centered), axis=2))
        token_mean = mx.mean(temporal_mean, axis=1)
        return token_mean, temporal_mean, temporal_std

    @staticmethod
    def _compact_temporal_diagnostics(temporal_stds: mx.array) -> mx.array:
        """Reduce ``[batch,state,tubelet,feature]`` stds to the canonical 53 columns."""
        if temporal_stds.ndim != 4:
            raise ValueError(
                "expected temporal stds with shape [batch,state,tubelet,feature], "
                f"got {temporal_stds.shape}"
            )
        # The H100 cache stored full temporal stds as float16. Its postpass then
        # reduced float32 views, persisted the state summaries as float16, and
        # the benchmark loader restored those summaries to float32. Preserve
        # that quantization order without materializing the full arrays on CPU.
        cached_std = temporal_stds.astype(mx.float16).astype(mx.float32)
        global_std = mx.mean(cached_std, axis=(1, 2, 3))[:, None]
        by_state = mx.mean(cached_std, axis=(2, 3)).astype(mx.float16).astype(mx.float32)
        by_state_token = mx.mean(cached_std, axis=3).astype(mx.float16).astype(mx.float32)
        by_token = mx.mean(by_state_token, axis=1)
        return mx.concatenate((global_std, by_state, by_token), axis=1)

    def selected_token_mean_states(self, video_bcthw: mx.array, selected: list[int]) -> mx.array:
        selected_set = set(selected)
        captured: dict[int, mx.array] = {}
        batch, _channels, frames, height, width = video_bcthw.shape
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError(
                f"input size {(height, width)} must be divisible by patch size {self.config.patch_size}"
            )
        temporal = frames // self.config.tubelet_size
        h_patches = height // self.config.patch_size
        w_patches = width // self.config.patch_size
        hidden = self.patch_embed(video_bcthw)
        if self.config.modality_embedding:
            hidden = hidden + mx.broadcast_to(self.video_mod_embed, (batch, 1, self.config.hidden_size))
        if 0 in selected_set:
            captured[0] = self._token_mean(hidden)
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden, temporal=temporal, h_patches=h_patches, w_patches=w_patches)
            if index in selected_set:
                state = self.norms_block[-1](hidden) if index == len(self.blocks) else hidden
                captured[index] = self._token_mean(state)
        missing = [index for index in selected if index not in captured]
        if missing:
            raise ValueError(f"selected hidden-state indices out of range or not captured: {missing}")
        return mx.stack([captured[index] for index in selected], axis=1)[:, :, None, :]

    def selected_states_with_temporal_stats(
        self,
        video_bcthw: mx.array,
        selected: list[int],
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return token means plus per-tubelet spatial mean/std summaries."""
        selected_set = set(selected)
        captured: dict[int, tuple[mx.array, mx.array, mx.array]] = {}
        batch, _channels, frames, height, width = video_bcthw.shape
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError(
                f"input size {(height, width)} must be divisible by patch size {self.config.patch_size}"
            )
        temporal = frames // self.config.tubelet_size
        h_patches = height // self.config.patch_size
        w_patches = width // self.config.patch_size
        spatial_tokens = h_patches * w_patches
        hidden = self.patch_embed(video_bcthw)
        if self.config.modality_embedding:
            hidden = hidden + mx.broadcast_to(self.video_mod_embed, (batch, 1, self.config.hidden_size))
        if 0 in selected_set:
            captured[0] = self._state_summaries(
                hidden,
                batch=batch,
                temporal=temporal,
                spatial_tokens=spatial_tokens,
            )
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden, temporal=temporal, h_patches=h_patches, w_patches=w_patches)
            if index in selected_set:
                state = self.norms_block[-1](hidden) if index == len(self.blocks) else hidden
                captured[index] = self._state_summaries(
                    state,
                    batch=batch,
                    temporal=temporal,
                    spatial_tokens=spatial_tokens,
                )
        missing = [index for index in selected if index not in captured]
        if missing:
            raise ValueError(f"selected hidden-state indices out of range or not captured: {missing}")
        token_means = mx.stack([captured[index][0] for index in selected], axis=1)[:, :, None, :]
        temporal_means = mx.stack([captured[index][1] for index in selected], axis=1)
        temporal_stds = mx.stack([captured[index][2] for index in selected], axis=1)
        return token_means, temporal_means, temporal_stds

    def selected_states_with_compact_temporal_diagnostics(
        self,
        video_bcthw: mx.array,
        selected: list[int],
    ) -> tuple[mx.array, mx.array]:
        """Return selected token means and only the diagnostics consumed downstream."""
        selected_set = set(selected)
        captured: dict[int, tuple[mx.array, mx.array]] = {}
        batch, _channels, frames, height, width = video_bcthw.shape
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError(
                f"input size {(height, width)} must be divisible by patch size {self.config.patch_size}"
            )
        temporal = frames // self.config.tubelet_size
        h_patches = height // self.config.patch_size
        w_patches = width // self.config.patch_size
        spatial_tokens = h_patches * w_patches
        hidden = self.patch_embed(video_bcthw)
        if self.config.modality_embedding:
            hidden = hidden + mx.broadcast_to(self.video_mod_embed, (batch, 1, self.config.hidden_size))

        def capture(index: int, state: mx.array) -> None:
            token_mean, _temporal_mean, temporal_std = self._state_summaries(
                state,
                batch=batch,
                temporal=temporal,
                spatial_tokens=spatial_tokens,
            )
            captured[index] = (token_mean, temporal_std)

        if 0 in selected_set:
            capture(0, hidden)
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden, temporal=temporal, h_patches=h_patches, w_patches=w_patches)
            if index in selected_set:
                state = self.norms_block[-1](hidden) if index == len(self.blocks) else hidden
                capture(index, state)
        missing = [index for index in selected if index not in captured]
        if missing:
            raise ValueError(f"selected hidden-state indices out of range or not captured: {missing}")
        token_means = mx.stack([captured[index][0] for index in selected], axis=1)[:, :, None, :]
        temporal_stds = mx.stack([captured[index][1] for index in selected], axis=1)
        diagnostics = self._compact_temporal_diagnostics(temporal_stds)
        return token_means, diagnostics


def _mlx_array_with_dtype(array: np.ndarray, dtype_name: str) -> mx.array:
    if dtype_name == "float16":
        return mx.array(array.astype(np.float16, copy=False))
    if dtype_name == "bfloat16":
        return mx.array(array.astype(np.float32, copy=False), dtype=mx.bfloat16)
    if dtype_name == "float32":
        return mx.array(array.astype(np.float32, copy=False))
    raise ValueError(f"Unsupported MLX input dtype: {dtype_name}")


class MlxVjepa21FeatureModel:
    def __init__(
        self,
        weights_dir: str,
        image_size: int | None = None,
        *,
        compile_encoder: bool = False,
        input_dtype: str = "float16",
    ) -> None:
        root = Path(weights_dir).expanduser().resolve()
        self.weights_dir = root
        self.config = MlxVjepa21Config.from_json(root / "config.json")
        self.image_size = int(image_size or self.config.crop_size)
        self.compile_encoder = bool(compile_encoder)
        self.input_dtype = input_dtype
        self._compiled_selected_state_fns: dict[tuple[int, ...], tp.Callable[[mx.array], mx.array]] = {}
        self._compiled_temporal_state_fns: dict[
            tuple[int, ...],
            tp.Callable[[mx.array], tuple[mx.array, mx.array, mx.array]],
        ] = {}
        self._compiled_compact_temporal_state_fns: dict[
            tuple[int, ...],
            tp.Callable[[mx.array], tuple[mx.array, mx.array]],
        ] = {}
        self.encoder = MlxVjepa21Encoder(self.config)
        weights = mx.load(str(root / "model.safetensors"))
        self.encoder.load_weights(list(weights.items()), strict=True)
        mx.eval(self.encoder.parameters())
        self.num_frames = self.config.frames_per_clip

    def predict_hidden_states(self, images: np.ndarray, selected_indices: list[int]) -> np.ndarray:
        batch = _preprocess_video_batch(images, self.image_size)
        video = _mlx_array_with_dtype(batch, self.input_dtype)
        if self.compile_encoder:
            selected_key = tuple(int(index) for index in selected_indices)
            forward = self._compiled_selected_state_fns.get(selected_key)
            if forward is None:
                def selected_forward(video_batch: mx.array) -> mx.array:
                    return self.encoder.selected_token_mean_states(video_batch, list(selected_key))

                forward = mx.compile(selected_forward)
                self._compiled_selected_state_fns[selected_key] = forward
            states = forward(video)
        else:
            states = self.encoder.selected_token_mean_states(video, selected_indices)
        mx.eval(states)
        return np.asarray(states.astype(mx.float32))

    def predict_hidden_states_with_temporal_stats(
        self,
        images: np.ndarray,
        selected_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode windows and preserve compactable temporal diagnostics."""
        batch = _preprocess_video_batch(images, self.image_size)
        video = _mlx_array_with_dtype(batch, self.input_dtype)
        if self.compile_encoder:
            selected_key = tuple(int(index) for index in selected_indices)
            forward = self._compiled_temporal_state_fns.get(selected_key)
            if forward is None:
                def temporal_forward(
                    video_batch: mx.array,
                ) -> tuple[mx.array, mx.array, mx.array]:
                    return self.encoder.selected_states_with_temporal_stats(
                        video_batch,
                        list(selected_key),
                    )

                forward = mx.compile(temporal_forward)
                self._compiled_temporal_state_fns[selected_key] = forward
            states, temporal_mean, temporal_std = forward(video)
        else:
            states, temporal_mean, temporal_std = self.encoder.selected_states_with_temporal_stats(
                video,
                selected_indices,
            )
        mx.eval(states, temporal_mean, temporal_std)
        return (
            np.asarray(states.astype(mx.float32)),
            np.asarray(temporal_mean.astype(mx.float32)),
            np.asarray(temporal_std.astype(mx.float32)),
        )

    def predict_hidden_states_with_compact_temporal_diagnostics(
        self,
        images: np.ndarray,
        selected_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode windows and transfer only selected states plus canonical diagnostics."""
        batch = _preprocess_video_batch(images, self.image_size)
        video = _mlx_array_with_dtype(batch, self.input_dtype)
        if self.compile_encoder:
            selected_key = tuple(int(index) for index in selected_indices)
            forward = self._compiled_compact_temporal_state_fns.get(selected_key)
            if forward is None:
                def compact_forward(video_batch: mx.array) -> tuple[mx.array, mx.array]:
                    return self.encoder.selected_states_with_compact_temporal_diagnostics(
                        video_batch,
                        list(selected_key),
                    )

                forward = mx.compile(compact_forward)
                self._compiled_compact_temporal_state_fns[selected_key] = forward
            states, diagnostics = forward(video)
        else:
            states, diagnostics = self.encoder.selected_states_with_compact_temporal_diagnostics(
                video,
                selected_indices,
            )
        mx.eval(states, diagnostics)
        return (
            np.asarray(states.astype(mx.float32)),
            np.asarray(diagnostics.astype(mx.float32)),
        )


class MlxVjepa21Video(MlxVjepa2Video):
    """Neuralset video extractor using MLX V-JEPA 2.1 ViT-g."""

    mlx_weights_dir: str = "models/vjepa21_mlx/vitg"
    image_size: int = 384
    compile_encoder: bool = False
    input_dtype: str = "bfloat16"
    cache_model_name: str | None = "mlx-vjepa21-vitg-384-selected-hidden-states-v1"
    _model: MlxVjepa21FeatureModel | None = pydantic.PrivateAttr(default=None)

    @property
    def model(self) -> MlxVjepa21FeatureModel:
        if getattr(self, "_model", None) is None:
            self._model = MlxVjepa21FeatureModel(
                self.mlx_weights_dir,
                self.image_size,
                compile_encoder=self.compile_encoder,
                input_dtype=self.input_dtype,
            )
        return self._model


__all__ = ["MlxVjepa21FeatureModel", "MlxVjepa21Video"]
