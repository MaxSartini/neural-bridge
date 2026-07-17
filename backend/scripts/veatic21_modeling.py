"""MLX modeling primitives for the VEATIC 2.1 retraining programme.

The caller owns every grouped-video split.  This module never invents row or
video ownership: inner-training and inner-validation indices are explicit,
are sealed into the artifact identity, and are checked for overlap.  It
supports the three bounded video heads, the exact seven-feature neural AR
baseline, continuous Huber training, true binary BCE-with-logits training,
and an optional frozen prediction/logit offset for residual correction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.scripts import run_again_dense_2hz_phase5_learned_heads as mlx_base


SCHEMA_VERSION = "veatic21_mlx_modeling_v1"
WINDOW_ROWS = 5
ALLOWED_PCA_WIDTHS = (64, 128, 256)
VIDEO_HEADS = (
    "short_temporal_conv_residual",
    "flat_mlp_residual",
    "current_row_mlp",
)
AR_HEAD = "neural_ar7"
ALLOWED_HEADS = VIDEO_HEADS + (AR_HEAD,)
CONTINUOUS = "weighted_huber"
BINARY = "binary_bce_logits"
ALLOWED_OBJECTIVES = (CONTINUOUS, BINARY)
DEFAULT_HIDDEN = 64
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_RESIDUAL_ALPHA_CAP = 0.12
DEFAULT_RESIDUAL_ALPHA_INITIAL_LOGIT = -4.0
DEFAULT_RESIDUAL_GATE_BIAS = 4.0


class Veatic21ModelingError(RuntimeError):
    """Raised when a modeling or artifact contract fails closed."""


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_mlx_gpu() -> str:
    if mlx_base.mx is None or mlx_base.nn is None or mlx_base.optim is None:
        raise Veatic21ModelingError("MLX is required for VEATIC 2.1 model training")
    device = str(mlx_base.mx.default_device())
    if "gpu" not in device.lower():
        raise Veatic21ModelingError(f"VEATIC 2.1 training requires MLX GPU, got {device}")
    return device


@dataclass(frozen=True)
class ModelSpec:
    head: str
    objective: str
    input_dim: int
    pca_width: int | None
    hidden_dim: int = DEFAULT_HIDDEN
    condition_on_frozen_offset: bool = False

    def validate(self) -> None:
        if self.head not in ALLOWED_HEADS:
            raise Veatic21ModelingError(f"Unsupported head {self.head!r}")
        if self.objective not in ALLOWED_OBJECTIVES:
            raise Veatic21ModelingError(f"Unsupported objective {self.objective!r}")
        if not isinstance(self.input_dim, int) or self.input_dim < 1:
            raise Veatic21ModelingError("input_dim must be a positive integer")
        if not isinstance(self.hidden_dim, int) or self.hidden_dim < 2:
            raise Veatic21ModelingError("hidden_dim must be at least two")
        if self.head == AR_HEAD:
            if self.input_dim != 7 or self.pca_width is not None:
                raise Veatic21ModelingError("neural_ar7 requires exactly seven inputs and no PCA")
            if self.condition_on_frozen_offset:
                raise Veatic21ModelingError("neural_ar7 cannot consume a frozen offset")
        else:
            if self.pca_width not in ALLOWED_PCA_WIDTHS:
                raise Veatic21ModelingError(
                    f"video heads require PCA width in {ALLOWED_PCA_WIDTHS}"
                )
            if self.head == "short_temporal_conv_residual" and self.input_dim < (
                WINDOW_ROWS * int(self.pca_width)
            ):
                raise Veatic21ModelingError("short temporal input is narrower than its PCA sequence")


@dataclass(frozen=True)
class Standardization:
    mean: np.ndarray
    std: np.ndarray
    fit_row_digest: str


@dataclass(frozen=True)
class ModelResult:
    train_prediction: np.ndarray
    test_prediction: np.ndarray
    train_probability: np.ndarray | None
    test_probability: np.ndarray | None
    train_correction: np.ndarray
    test_correction: np.ndarray
    checkpoint_path: Path
    manifest_path: Path
    normalization_path: Path
    checkpoint_sha256: str
    artifact_digest: str
    best_epoch: int
    best_validation_loss: float
    curves: tuple[Mapping[str, Any], ...]
    cache_hit: bool
    device: str


class Veatic21ScalarHead(mlx_base.nn.Module):
    """Bounded video/AR scalar head with variable PCA width."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        spec.validate()
        self.head_name = spec.head
        self.pca_width = int(spec.pca_width or 0)
        self.input_dim = int(spec.input_dim) + int(spec.condition_on_frozen_offset)
        hidden = int(spec.hidden_dim)
        self.condition_on_frozen_offset = bool(spec.condition_on_frozen_offset)
        if self.condition_on_frozen_offset:
            # Preserve the promoted AGAIN do-no-harm mechanism: a learned
            # residual is both globally capped and row-gated before it can
            # alter the frozen AR prediction/logit.
            self.alpha = mlx_base.mx.array(
                [DEFAULT_RESIDUAL_ALPHA_INITIAL_LOGIT], dtype=mlx_base.mx.float32
            )
            self.alpha_cap = float(DEFAULT_RESIDUAL_ALPHA_CAP)
            self.gate_bias = float(DEFAULT_RESIDUAL_GATE_BIAS)
            self.gate = mlx_base.nn.Linear(self.input_dim, 1)
        if spec.head == "short_temporal_conv_residual":
            self.sequence_width = WINDOW_ROWS * self.pca_width
            self.conv = mlx_base.nn.Linear(self.pca_width * 3, hidden)
            self.post = mlx_base.nn.Linear(
                hidden + self.input_dim - self.sequence_width, hidden
            )
        else:
            self.sequence_width = 0
            second_hidden = max(32, hidden // 2) if spec.head == AR_HEAD else hidden
            self.layers = [
                mlx_base.nn.Linear(self.input_dim, hidden),
                mlx_base.nn.Linear(hidden, second_hidden),
            ]
            hidden = second_hidden
        self.out = mlx_base.nn.Linear(hidden, 1)

    def __call__(self, x: Any) -> Any:
        if self.head_name == "short_temporal_conv_residual":
            sequence = x[:, : self.sequence_width].reshape(
                (x.shape[0], WINDOW_ROWS, self.pca_width)
            )
            extras = x[:, self.sequence_width :]
            padded = mlx_base.mx.concatenate(
                [
                    mlx_base.mx.zeros(
                        (x.shape[0], 2, self.pca_width), dtype=x.dtype
                    ),
                    sequence,
                ],
                axis=1,
            )
            hidden_rows = []
            for position in range(WINDOW_ROWS):
                window = padded[:, position : position + 3, :].reshape(
                    (x.shape[0], self.pca_width * 3)
                )
                hidden_rows.append(mlx_base.nn.gelu(self.conv(window)))
            hidden = mlx_base.nn.gelu(
                self.post(mlx_base.mx.concatenate([hidden_rows[-1], extras], axis=1))
            )
        else:
            hidden = x
            for layer in self.layers:
                hidden = mlx_base.nn.gelu(layer(hidden))
        raw = self.out(hidden)[:, 0]
        if not self.condition_on_frozen_offset:
            return raw
        gate = mlx_base.mx.sigmoid(self.gate(x)[:, 0] - self.gate_bias)
        scale = mlx_base.mx.sigmoid(self.alpha)[0] * self.alpha_cap
        return scale * gate * raw


def _indices(
    values: Sequence[int] | np.ndarray,
    *,
    rows: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise Veatic21ModelingError(f"{name} must be a one-dimensional integer array")
    indices = raw.astype(np.int64, copy=False)
    if len(indices) == 0:
        raise Veatic21ModelingError(f"{name} must not be empty")
    if np.any(indices < 0) or np.any(indices >= rows) or len(np.unique(indices)) != len(indices):
        raise Veatic21ModelingError(f"{name} contains duplicate or out-of-range rows")
    return np.sort(indices)


def _validate_arrays(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_loss_mask: np.ndarray,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    spec: ModelSpec,
    frozen_train_offset: np.ndarray | None,
    frozen_test_offset: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec.validate()
    x_train = np.asarray(train_x, dtype=np.float32)
    x_test = np.asarray(test_x, dtype=np.float32)
    target = np.asarray(train_target, dtype=np.float32)
    mask = np.asarray(train_loss_mask)
    if x_train.ndim != 2 or x_test.ndim != 2 or x_train.shape[1] != x_test.shape[1]:
        raise Veatic21ModelingError("train/test features must be aligned two-dimensional matrices")
    if x_train.shape[1] != spec.input_dim:
        raise Veatic21ModelingError("feature width differs from ModelSpec.input_dim")
    if target.shape != (len(x_train),) or mask.shape != (len(x_train),) or mask.dtype != np.bool_:
        raise Veatic21ModelingError("target/loss mask must be row-aligned; mask must be boolean")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise Veatic21ModelingError("features contain non-finite values")
    if not np.isfinite(target[mask]).all():
        raise Veatic21ModelingError("eligible targets contain non-finite values")
    inner_train = _indices(inner_train_idx, rows=len(x_train), name="inner_train_idx")
    inner_val = _indices(inner_val_idx, rows=len(x_train), name="inner_val_idx")
    if np.intersect1d(inner_train, inner_val).size:
        raise Veatic21ModelingError("inner train and validation rows overlap")
    if not np.all(mask[inner_train]) or not np.all(mask[inner_val]):
        raise Veatic21ModelingError("inner ownership includes rows outside the loss mask")
    train_offset = (
        np.zeros(len(x_train), dtype=np.float32)
        if frozen_train_offset is None
        else np.asarray(frozen_train_offset, dtype=np.float32)
    )
    test_offset = (
        np.zeros(len(x_test), dtype=np.float32)
        if frozen_test_offset is None
        else np.asarray(frozen_test_offset, dtype=np.float32)
    )
    if train_offset.shape != (len(x_train),) or test_offset.shape != (len(x_test),):
        raise Veatic21ModelingError("frozen offsets must be row-aligned")
    if not np.isfinite(train_offset).all() or not np.isfinite(test_offset).all():
        raise Veatic21ModelingError("frozen offsets contain non-finite values")
    has_offsets = frozen_train_offset is not None or frozen_test_offset is not None
    if has_offsets != (frozen_train_offset is not None and frozen_test_offset is not None):
        raise Veatic21ModelingError("train and test frozen offsets must be supplied together")
    if spec.condition_on_frozen_offset and not has_offsets:
        raise Veatic21ModelingError("conditioned residual head requires frozen offsets")
    if spec.objective == BINARY:
        values = target[mask]
        if not np.all(np.isin(values, (0.0, 1.0))) or len(np.unique(values)) < 2:
            raise Veatic21ModelingError("binary BCE targets must contain both 0 and 1")
    return x_train, x_test, target, mask, train_offset, test_offset


def fit_standardization(x: np.ndarray, fit_rows: np.ndarray) -> Standardization:
    values = np.asarray(x[fit_rows], dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise Veatic21ModelingError("invalid train-only standardization")
    return Standardization(mean=mean, std=std, fit_row_digest=array_digest(fit_rows))


def _apply_standardization(x: np.ndarray, state: Standardization) -> np.ndarray:
    values = ((np.asarray(x, dtype=np.float32) - state.mean) / state.std).astype(np.float32)
    if not np.isfinite(values).all():
        raise Veatic21ModelingError("standardized features contain non-finite values")
    return values


def _model_input(
    x: np.ndarray,
    offset: np.ndarray,
    *,
    condition_on_offset: bool,
) -> np.ndarray:
    if not condition_on_offset:
        return x
    return np.concatenate([x, offset[:, None]], axis=1).astype(np.float32, copy=False)


def _predict(
    model: Veatic21ScalarHead,
    x: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if hasattr(model, "eval"):
        model.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        output = model(
            mlx_base.mx.array(x[start : start + int(batch_size)], dtype=mlx_base.mx.float32)
        )
        mlx_base.mx.eval(output)
        chunks.append(np.asarray(output, dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _numpy_loss(objective: str, prediction: np.ndarray, target: np.ndarray) -> float:
    if objective == CONTINUOUS:
        residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        absolute = np.abs(residual)
        return float(np.mean(np.where(absolute <= 1.0, 0.5 * residual**2, absolute - 0.5)))
    logits = np.asarray(prediction, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def _train_loop(
    *,
    spec: ModelSpec,
    x: np.ndarray,
    target: np.ndarray,
    offset: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray | None,
    seed: int,
    epochs: int,
    patience: int | None,
    batch_size: int,
    checkpoint_path: Path,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
) -> tuple[Veatic21ScalarHead, tuple[Mapping[str, Any], ...], int, float]:
    mlx_base.mx.random.seed(int(seed))
    model = Veatic21ScalarHead(spec)
    optimizer = mlx_base.optim.AdamW(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    rng = np.random.default_rng(int(seed) + 130003)
    if spec.objective == CONTINUOUS:
        q80, q90 = np.quantile(np.abs(target[fit_rows]), [0.80, 0.90])
    else:
        q80 = q90 = math.inf

    def loss_fn(model_obj: Any, xb: Any, yb: Any, ob: Any, wb: Any) -> Any:
        correction = model_obj(xb)
        prediction = correction + ob
        if spec.objective == CONTINUOUS:
            losses = mlx_base.nn.losses.huber_loss(prediction, yb, delta=1.0)
            return mlx_base.mx.mean(losses * wb)
        return mlx_base.mx.mean(
            mlx_base.nn.losses.binary_cross_entropy(prediction, yb, with_logits=True)
        )

    loss_and_grad = mlx_base.nn.value_and_grad(model, loss_fn)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    curves: list[Mapping[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        if hasattr(model, "train"):
            model.train()
        order = rng.permutation(fit_rows)
        losses: list[float] = []
        for start in range(0, len(order), int(batch_size)):
            rows = order[start : start + int(batch_size)]
            weights = (
                1.0
                + (np.abs(target[rows]) >= q80).astype(np.float32)
                + (np.abs(target[rows]) >= q90).astype(np.float32)
                if spec.objective == CONTINUOUS
                else np.ones(len(rows), dtype=np.float32)
            )
            loss, gradients = loss_and_grad(
                model,
                mlx_base.mx.array(x[rows], dtype=mlx_base.mx.float32),
                mlx_base.mx.array(target[rows], dtype=mlx_base.mx.float32),
                mlx_base.mx.array(offset[rows], dtype=mlx_base.mx.float32),
                mlx_base.mx.array(weights, dtype=mlx_base.mx.float32),
            )
            gradients, _ = mlx_base.optim.clip_grad_norm(gradients, float(grad_clip))
            optimizer.update(model, gradients)
            mlx_base.mx.eval(loss, model.parameters(), optimizer.state)
            losses.append(float(np.asarray(loss)))
        if validation_rows is None:
            validation_loss = float(np.mean(losses)) if losses else math.inf
        else:
            correction = _predict(model, x[validation_rows], batch_size=batch_size)
            validation_loss = _numpy_loss(
                spec.objective,
                correction + offset[validation_rows],
                target[validation_rows],
            )
        curves.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else math.nan,
                "validation_loss": validation_loss,
            }
        )
        if validation_rows is None:
            best_epoch = epoch
            best_loss = validation_loss
        elif math.isfinite(validation_loss) and validation_loss < best_loss:
            model.save_weights(str(checkpoint_path))
            best_epoch = epoch
            best_loss = validation_loss
            stale = 0
        else:
            stale += 1
        if validation_rows is not None and patience is not None and stale >= int(patience):
            break
    if validation_rows is None:
        model.save_weights(str(checkpoint_path))
    if best_epoch < 1 or not checkpoint_path.exists():
        raise Veatic21ModelingError("training did not produce a valid checkpoint")
    restored = Veatic21ScalarHead(spec)
    _ = restored(mlx_base.mx.array(x[: min(2, len(x))], dtype=mlx_base.mx.float32))
    restored.load_weights(str(checkpoint_path))
    if hasattr(restored, "eval"):
        restored.eval()
    return restored, tuple(curves), int(best_epoch), float(best_loss)


def _paths(checkpoint_path: Path) -> tuple[Path, Path]:
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix != ".npz":
        raise Veatic21ModelingError("checkpoint_path must end in .npz")
    return checkpoint.with_suffix(".json"), checkpoint.with_name(
        checkpoint.stem + "__normalization.npz"
    )


def train_scalar_model(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_target: np.ndarray,
    train_loss_mask: np.ndarray,
    inner_train_idx: Sequence[int] | np.ndarray,
    inner_val_idx: Sequence[int] | np.ndarray,
    spec: ModelSpec,
    seed: int,
    checkpoint_path: Path,
    artifact_identity: Mapping[str, Any],
    frozen_train_offset: np.ndarray | None = None,
    frozen_test_offset: np.ndarray | None = None,
    refit_after_selection: bool = False,
    batch_size: int = 8192,
    max_epochs: int = 80,
    patience: int = 12,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
) -> ModelResult:
    """Select an epoch on explicit inner ownership and optionally refit all rows."""

    device = require_mlx_gpu()
    x_train, x_test, target, mask, train_offset, test_offset = _validate_arrays(
        train_x=train_x,
        test_x=test_x,
        train_target=train_target,
        train_loss_mask=train_loss_mask,
        inner_train_idx=np.asarray(inner_train_idx),
        inner_val_idx=np.asarray(inner_val_idx),
        spec=spec,
        frozen_train_offset=frozen_train_offset,
        frozen_test_offset=frozen_test_offset,
    )
    inner_train = _indices(inner_train_idx, rows=len(x_train), name="inner_train_idx")
    inner_val = _indices(inner_val_idx, rows=len(x_train), name="inner_val_idx")
    eligible = np.flatnonzero(mask & np.isfinite(target)).astype(np.int64)
    if len(eligible) < 2:
        raise Veatic21ModelingError("not enough eligible training rows")
    if not isinstance(artifact_identity, Mapping) or not artifact_identity:
        raise Veatic21ModelingError("artifact_identity must be a non-empty mapping")
    manifest_path, normalization_path = _paths(Path(checkpoint_path))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_identity": dict(artifact_identity),
        "spec": asdict(spec),
        "seed": int(seed),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "target_digest": array_digest(target),
        "loss_mask_digest": array_digest(mask),
        "inner_train_digest": array_digest(inner_train),
        "inner_val_digest": array_digest(inner_val),
        "frozen_train_offset_digest": array_digest(train_offset),
        "frozen_test_offset_digest": array_digest(test_offset),
        "refit_after_selection": bool(refit_after_selection),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
    }
    artifact_digest = canonical_digest(identity)
    if Path(checkpoint_path).exists() or manifest_path.exists() or normalization_path.exists():
        if not (Path(checkpoint_path).exists() and manifest_path.exists() and normalization_path.exists()):
            raise Veatic21ModelingError("incomplete model artifact set")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_digest") != artifact_digest or manifest.get("identity") != identity:
            raise Veatic21ModelingError("model resume identity mismatch")
        if manifest.get("checkpoint_sha256") != file_sha256(Path(checkpoint_path)):
            raise Veatic21ModelingError("model checkpoint checksum drift")
        if manifest.get("normalization_sha256") != file_sha256(normalization_path):
            raise Veatic21ModelingError("model normalization checksum drift")
        with np.load(normalization_path, allow_pickle=False) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            std = np.asarray(payload["std"], dtype=np.float32)
            fit_rows = np.asarray(payload["fit_rows"], dtype=np.int64)
        standardization = Standardization(mean, std, array_digest(fit_rows))
        train_std = _apply_standardization(x_train, standardization)
        test_std = _apply_standardization(x_test, standardization)
        model_train = _model_input(
            train_std, train_offset, condition_on_offset=spec.condition_on_frozen_offset
        )
        model_test = _model_input(
            test_std, test_offset, condition_on_offset=spec.condition_on_frozen_offset
        )
        model = Veatic21ScalarHead(spec)
        _ = model(mlx_base.mx.array(model_train[:2], dtype=mlx_base.mx.float32))
        model.load_weights(str(checkpoint_path))
        correction_train = _predict(model, model_train, batch_size=batch_size)
        correction_test = _predict(model, model_test, batch_size=batch_size)
        prediction_train = correction_train + train_offset
        prediction_test = correction_test + test_offset
        curves = tuple(manifest.get("curves", ()))
        return ModelResult(
            train_prediction=prediction_train,
            test_prediction=prediction_test,
            train_probability=_sigmoid(prediction_train) if spec.objective == BINARY else None,
            test_probability=_sigmoid(prediction_test) if spec.objective == BINARY else None,
            train_correction=correction_train,
            test_correction=correction_test,
            checkpoint_path=Path(checkpoint_path),
            manifest_path=manifest_path,
            normalization_path=normalization_path,
            checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            artifact_digest=artifact_digest,
            best_epoch=int(manifest["best_epoch"]),
            best_validation_loss=float(manifest["best_validation_loss"]),
            curves=curves,
            cache_hit=True,
            device=str(manifest["device"]),
        )

    selection_state = fit_standardization(x_train, inner_train)
    selection_train = _apply_standardization(x_train, selection_state)
    selection_input = _model_input(
        selection_train,
        train_offset,
        condition_on_offset=spec.condition_on_frozen_offset,
    )
    selection_checkpoint = Path(checkpoint_path).with_name(
        Path(checkpoint_path).stem + "__selection.npz"
    )
    _selected, curves, best_epoch, best_loss = _train_loop(
        spec=spec,
        x=selection_input,
        target=target,
        offset=train_offset,
        fit_rows=inner_train,
        validation_rows=inner_val,
        seed=int(seed),
        epochs=int(max_epochs),
        patience=int(patience),
        batch_size=int(batch_size),
        checkpoint_path=selection_checkpoint,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_clip=float(grad_clip),
    )
    if refit_after_selection:
        final_state = fit_standardization(x_train, eligible)
        train_std = _apply_standardization(x_train, final_state)
        test_std = _apply_standardization(x_test, final_state)
        model_train = _model_input(
            train_std, train_offset, condition_on_offset=spec.condition_on_frozen_offset
        )
        model_test = _model_input(
            test_std, test_offset, condition_on_offset=spec.condition_on_frozen_offset
        )
        model, refit_curves, _refit_epoch, _refit_loss = _train_loop(
            spec=spec,
            x=model_train,
            target=target,
            offset=train_offset,
            fit_rows=eligible,
            validation_rows=None,
            seed=int(seed),
            epochs=int(best_epoch),
            patience=None,
            batch_size=int(batch_size),
            checkpoint_path=Path(checkpoint_path),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            grad_clip=float(grad_clip),
        )
        curves = tuple(curves) + tuple(
            {**row, "phase": "all_outer_train_refit"} for row in refit_curves
        )
        fit_rows_for_state = eligible
    else:
        final_state = selection_state
        train_std = selection_train
        test_std = _apply_standardization(x_test, final_state)
        model_train = selection_input
        model_test = _model_input(
            test_std, test_offset, condition_on_offset=spec.condition_on_frozen_offset
        )
        os.replace(selection_checkpoint, Path(checkpoint_path))
        model = Veatic21ScalarHead(spec)
        _ = model(mlx_base.mx.array(model_train[:2], dtype=mlx_base.mx.float32))
        model.load_weights(str(checkpoint_path))
        fit_rows_for_state = inner_train
    selection_checkpoint.unlink(missing_ok=True)
    correction_train = _predict(model, model_train, batch_size=batch_size)
    correction_test = _predict(model, model_test, batch_size=batch_size)
    prediction_train = correction_train + train_offset
    prediction_test = correction_test + test_offset
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        normalization_path,
        mean=final_state.mean,
        std=final_state.std,
        fit_rows=fit_rows_for_state,
    )
    checkpoint_hash = file_sha256(Path(checkpoint_path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_digest": artifact_digest,
        "identity": identity,
        "checkpoint_sha256": checkpoint_hash,
        "normalization_sha256": file_sha256(normalization_path),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "curves": list(curves),
        "device": device,
        "eval_mode_restored": True,
    }
    temporary = manifest_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return ModelResult(
        train_prediction=prediction_train,
        test_prediction=prediction_test,
        train_probability=_sigmoid(prediction_train) if spec.objective == BINARY else None,
        test_probability=_sigmoid(prediction_test) if spec.objective == BINARY else None,
        train_correction=correction_train,
        test_correction=correction_test,
        checkpoint_path=Path(checkpoint_path),
        manifest_path=manifest_path,
        normalization_path=normalization_path,
        checkpoint_sha256=checkpoint_hash,
        artifact_digest=artifact_digest,
        best_epoch=int(best_epoch),
        best_validation_loss=float(best_loss),
        curves=tuple(curves),
        cache_hit=False,
        device=device,
    )


def refit_scalar_model_fixed_epochs(
    *,
    train_x: np.ndarray,
    train_target: np.ndarray,
    train_loss_mask: np.ndarray,
    spec: ModelSpec,
    seed: int,
    epochs: int,
    checkpoint_path: Path,
    artifact_identity: Mapping[str, Any],
    frozen_train_offset: np.ndarray | None = None,
    batch_size: int = 8192,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
) -> ModelResult:
    """Retrain from scratch for a frozen epoch count on every eligible row."""

    device = require_mlx_gpu()
    if not isinstance(epochs, int) or epochs < 1:
        raise Veatic21ModelingError("fixed epochs must be a positive integer")
    spec.validate()
    x = np.asarray(train_x, dtype=np.float32)
    target = np.asarray(train_target, dtype=np.float32)
    mask = np.asarray(train_loss_mask)
    if x.ndim != 2 or x.shape[1] != spec.input_dim or not np.isfinite(x).all():
        raise Veatic21ModelingError("invalid all-data refit feature matrix")
    if target.shape != (len(x),) or mask.shape != (len(x),) or mask.dtype != np.bool_:
        raise Veatic21ModelingError("invalid all-data refit target/mask")
    eligible = np.flatnonzero(mask & np.isfinite(target)).astype(np.int64)
    if len(eligible) < 2:
        raise Veatic21ModelingError("not enough eligible rows for all-data refit")
    if spec.objective == BINARY:
        labels = target[eligible]
        if not np.all(np.isin(labels, (0.0, 1.0))) or len(np.unique(labels)) < 2:
            raise Veatic21ModelingError("binary all-data refit requires both classes")
    offset = (
        np.zeros(len(x), dtype=np.float32)
        if frozen_train_offset is None
        else np.asarray(frozen_train_offset, dtype=np.float32)
    )
    if offset.shape != (len(x),) or not np.isfinite(offset).all():
        raise Veatic21ModelingError("invalid all-data frozen offset")
    if spec.condition_on_frozen_offset != (frozen_train_offset is not None):
        raise Veatic21ModelingError(
            "all-data offset conditioning and frozen offset presence must agree"
        )
    if not isinstance(artifact_identity, Mapping) or not artifact_identity:
        raise Veatic21ModelingError("artifact_identity must be a non-empty mapping")
    manifest_path, normalization_path = _paths(Path(checkpoint_path))
    identity = {
        "schema_version": SCHEMA_VERSION,
        "artifact_identity": dict(artifact_identity),
        "spec": asdict(spec),
        "seed": int(seed),
        "train_shape": list(x.shape),
        "target_digest": array_digest(target),
        "loss_mask_digest": array_digest(mask),
        "eligible_digest": array_digest(eligible),
        "frozen_offset_digest": array_digest(offset),
        "fixed_epoch_all_data_refit": True,
        "fixed_epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip": float(grad_clip),
    }
    artifact_digest = canonical_digest(identity)
    if Path(checkpoint_path).exists() or manifest_path.exists() or normalization_path.exists():
        if not (Path(checkpoint_path).exists() and manifest_path.exists() and normalization_path.exists()):
            raise Veatic21ModelingError("incomplete all-data model artifact set")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_digest") != artifact_digest or manifest.get("identity") != identity:
            raise Veatic21ModelingError("all-data model resume identity mismatch")
        if manifest.get("checkpoint_sha256") != file_sha256(Path(checkpoint_path)):
            raise Veatic21ModelingError("all-data checkpoint checksum drift")
        if manifest.get("normalization_sha256") != file_sha256(normalization_path):
            raise Veatic21ModelingError("all-data normalization checksum drift")
        with np.load(normalization_path, allow_pickle=False) as payload:
            state = Standardization(
                np.asarray(payload["mean"], dtype=np.float32),
                np.asarray(payload["std"], dtype=np.float32),
                array_digest(np.asarray(payload["fit_rows"], dtype=np.int64)),
            )
        standardized = _apply_standardization(x, state)
        model_input = _model_input(
            standardized, offset, condition_on_offset=spec.condition_on_frozen_offset
        )
        model = Veatic21ScalarHead(spec)
        _ = model(mlx_base.mx.array(model_input[:2], dtype=mlx_base.mx.float32))
        model.load_weights(str(checkpoint_path))
        correction = _predict(model, model_input, batch_size=batch_size)
        prediction = correction + offset
        curves = tuple(manifest.get("curves", ()))
        return ModelResult(
            train_prediction=prediction,
            test_prediction=np.zeros(0, dtype=np.float32),
            train_probability=_sigmoid(prediction) if spec.objective == BINARY else None,
            test_probability=(np.zeros(0, dtype=np.float32) if spec.objective == BINARY else None),
            train_correction=correction,
            test_correction=np.zeros(0, dtype=np.float32),
            checkpoint_path=Path(checkpoint_path),
            manifest_path=manifest_path,
            normalization_path=normalization_path,
            checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            artifact_digest=artifact_digest,
            best_epoch=int(epochs),
            best_validation_loss=math.nan,
            curves=curves,
            cache_hit=True,
            device=str(manifest["device"]),
        )

    state = fit_standardization(x, eligible)
    standardized = _apply_standardization(x, state)
    model_input = _model_input(
        standardized, offset, condition_on_offset=spec.condition_on_frozen_offset
    )
    model, curves, _last_epoch, _last_loss = _train_loop(
        spec=spec,
        x=model_input,
        target=target,
        offset=offset,
        fit_rows=eligible,
        validation_rows=None,
        seed=int(seed),
        epochs=int(epochs),
        patience=None,
        batch_size=int(batch_size),
        checkpoint_path=Path(checkpoint_path),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        grad_clip=float(grad_clip),
    )
    correction = _predict(model, model_input, batch_size=batch_size)
    prediction = correction + offset
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(normalization_path, mean=state.mean, std=state.std, fit_rows=eligible)
    checkpoint_hash = file_sha256(Path(checkpoint_path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_digest": artifact_digest,
        "identity": identity,
        "checkpoint_sha256": checkpoint_hash,
        "normalization_sha256": file_sha256(normalization_path),
        "best_epoch": int(epochs),
        "best_validation_loss": None,
        "curves": list(curves),
        "device": device,
        "eval_mode_restored": True,
        "in_sample_metrics_promotable": False,
    }
    temporary = manifest_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return ModelResult(
        train_prediction=prediction,
        test_prediction=np.zeros(0, dtype=np.float32),
        train_probability=_sigmoid(prediction) if spec.objective == BINARY else None,
        test_probability=(np.zeros(0, dtype=np.float32) if spec.objective == BINARY else None),
        train_correction=correction,
        test_correction=np.zeros(0, dtype=np.float32),
        checkpoint_path=Path(checkpoint_path),
        manifest_path=manifest_path,
        normalization_path=normalization_path,
        checkpoint_sha256=checkpoint_hash,
        artifact_digest=artifact_digest,
        best_epoch=int(epochs),
        best_validation_loss=math.nan,
        curves=tuple(curves),
        cache_hit=False,
        device=device,
    )


__all__ = [
    "ALLOWED_HEADS",
    "ALLOWED_OBJECTIVES",
    "AR_HEAD",
    "BINARY",
    "CONTINUOUS",
    "ModelResult",
    "ModelSpec",
    "SCHEMA_VERSION",
    "Standardization",
    "VIDEO_HEADS",
    "Veatic21ModelingError",
    "Veatic21ScalarHead",
    "array_digest",
    "canonical_digest",
    "fit_standardization",
    "refit_scalar_model_fixed_epochs",
    "require_mlx_gpu",
    "train_scalar_model",
]
