"""MLX-GPU randomized PCA and accuracy audits for VEATIC 2.1 Phase 04."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from neural_bridge.veatic21.contracts import (
    PCA_JACOBI_MAX_SWEEPS,
    PCA_JACOBI_TOLERANCE,
    PCA_ORTHOGONALITY_TOLERANCE,
    PCA_POWER_ITERATIONS,
)


@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    total_variance: float
    power_iterations: int
    oversampling: int
    seed: int
    orthogonality_max_abs: float
    reconstruction_residual_fraction: float
    device: str


def _round_robin_pairs(size: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if size < 2 or size % 2:
        raise ValueError("Jacobi size must be even and at least two")
    players = list(range(size))
    rounds = []
    for _ in range(size - 1):
        left = np.asarray(players[: size // 2], dtype=np.int32)
        right = np.asarray(players[size - 1 : size // 2 - 1 : -1], dtype=np.int32)
        rounds.append((left, right))
        players = [players[0], players[-1], *players[1:-1]]
    return tuple(rounds)


def jacobi_eigh_mlx(
    matrix: mx.array,
    *,
    max_sweeps: int = PCA_JACOBI_MAX_SWEEPS,
    tolerance: float = PCA_JACOBI_TOLERANCE,
) -> tuple[mx.array, mx.array, dict[str, float | int | bool]]:
    """Symmetric eigendecomposition using parallel Jacobi rotations on MLX GPU."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Jacobi eigendecomposition requires a square matrix")
    size = int(matrix.shape[0])
    pairs = _round_robin_pairs(size)
    mx.set_default_device(mx.gpu)
    value = matrix.astype(mx.float32)
    vectors = mx.eye(size, dtype=mx.float32)
    converged = False
    relative_offdiagonal = float("inf")
    sweeps = 0
    for sweep in range(1, max_sweeps + 1):
        for round_index, (left_np, right_np) in enumerate(pairs):
            left = mx.array(left_np)
            right = mx.array(right_np)
            diagonal = mx.diag(value)
            app = mx.take(diagonal, left)
            aqq = mx.take(diagonal, right)
            apq = value[left, right]
            safe = mx.abs(apq) > 1e-12
            tau = (aqq - app) / mx.where(safe, 2.0 * apq, 1.0)
            sign = mx.where(tau >= 0.0, 1.0, -1.0)
            tangent = mx.where(
                safe,
                sign / (mx.abs(tau) + mx.sqrt(1.0 + mx.square(tau))),
                0.0,
            )
            cosine = 1.0 / mx.sqrt(1.0 + mx.square(tangent))
            sine = tangent * cosine
            order = np.concatenate((left_np, right_np))
            inverse = mx.array(np.argsort(order).astype(np.int32))

            rows_left = mx.take(value, left, axis=0)
            rows_right = mx.take(value, right, axis=0)
            rotated_rows = mx.concatenate(
                (
                    cosine[:, None] * rows_left - sine[:, None] * rows_right,
                    sine[:, None] * rows_left + cosine[:, None] * rows_right,
                ),
                axis=0,
            )
            value = mx.take(rotated_rows, inverse, axis=0)
            columns_left = mx.take(value, left, axis=1)
            columns_right = mx.take(value, right, axis=1)
            rotated_columns = mx.concatenate(
                (
                    columns_left * cosine[None, :] - columns_right * sine[None, :],
                    columns_left * sine[None, :] + columns_right * cosine[None, :],
                ),
                axis=1,
            )
            value = mx.take(rotated_columns, inverse, axis=1)
            vectors_left = mx.take(vectors, left, axis=1)
            vectors_right = mx.take(vectors, right, axis=1)
            rotated_vectors = mx.concatenate(
                (
                    vectors_left * cosine[None, :] - vectors_right * sine[None, :],
                    vectors_left * sine[None, :] + vectors_right * cosine[None, :],
                ),
                axis=1,
            )
            vectors = mx.take(rotated_vectors, inverse, axis=1)
            if round_index % 32 == 31:
                mx.eval(value, vectors)
        mx.eval(value, vectors)
        diagonal = mx.diag(value)
        offdiagonal = value - mx.diag(diagonal)
        numerator = mx.sqrt(mx.sum(mx.square(offdiagonal)))
        denominator = mx.maximum(mx.sqrt(mx.sum(mx.square(diagonal))), 1e-12)
        relative_offdiagonal = float((numerator / denominator).item())
        sweeps = sweep
        if relative_offdiagonal <= tolerance:
            converged = True
            break
    eigenvalues = mx.diag(value)
    order = mx.argsort(-eigenvalues)
    eigenvalues = mx.take(eigenvalues, order)
    vectors = mx.take(vectors, order, axis=1)
    mx.eval(eigenvalues, vectors)
    return (
        eigenvalues,
        vectors,
        {
            "converged": converged,
            "sweeps": sweeps,
            "relative_offdiagonal": relative_offdiagonal,
        },
    )


def orthonormalize_mlx(
    matrix: mx.array,
    *,
    block_width: int = 32,
) -> mx.array:
    """Twice-reorthogonalized block Gram-Schmidt on GPU."""

    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("orthonormalization requires rows >= columns")
    if block_width < 1:
        raise ValueError("block width must be positive")
    rows, columns = map(int, matrix.shape)
    q = mx.zeros((rows, 0), dtype=mx.float32)
    for start in range(0, columns, block_width):
        stop = min(start + block_width, columns)
        candidate = matrix[:, start:stop]
        if q.shape[1]:
            for _ in range(2):
                candidate = candidate - q @ (q.T @ candidate)
        block = mx.zeros((rows, 0), dtype=mx.float32)
        for offset in range(stop - start):
            column = candidate[:, offset : offset + 1]
            if block.shape[1]:
                for _ in range(2):
                    column = column - block @ (block.T @ column)
            norm = mx.sqrt(mx.sum(mx.square(column)))
            column = column / mx.maximum(norm, 1e-12)
            block = mx.concatenate((block, column), axis=1)
            mx.eval(block)
        if q.shape[1]:
            for _ in range(2):
                block = block - q @ (q.T @ block)
            block = orthonormalize_mlx(block, block_width=block.shape[1])
        q = mx.concatenate((q, block), axis=1)
        mx.eval(q)
    return q


def _fit_standardized_basis(
    standardized: mx.array,
    *,
    rank: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> tuple[mx.array, mx.array, dict[str, float | int | bool]]:
    rows, width = map(int, standardized.shape)
    sketch_width = rank + oversampling
    if not 0 < rank < sketch_width <= min(rows, width):
        raise ValueError("invalid randomized PCA rank/oversampling")
    key = mx.random.key(seed)
    omega = mx.random.normal((width, sketch_width), key=key, dtype=mx.float32)
    q = orthonormalize_mlx(standardized @ omega)
    for _ in range(power_iterations):
        right = orthonormalize_mlx(standardized.T @ q)
        q = orthonormalize_mlx(standardized @ right)
    compressed = q.T @ standardized
    covariance = compressed @ compressed.T / max(1, rows - 1)
    eigenvalues, left_vectors, jacobi = jacobi_eigh_mlx(covariance)
    eigenvalues = mx.maximum(eigenvalues[:rank], 0.0)
    left_vectors = left_vectors[:, :rank]
    singular = mx.sqrt(mx.maximum(eigenvalues * max(1, rows - 1), 1e-12))
    components = (compressed.T @ left_vectors / singular[None, :]).T
    mx.eval(components, eigenvalues)
    return components, eigenvalues, jacobi


def fit_pca_mlx(
    features: np.ndarray,
    *,
    rank: int,
    oversampling: int,
    seed: int,
    power_iterations: int = PCA_POWER_ITERATIONS,
) -> tuple[PCAModel, dict[str, float | int | bool]]:
    """Fit standardized randomized PCA entirely with MLX operations on GPU."""

    features = np.asarray(features)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("PCA features must be a finite matrix")
    mx.set_default_device(mx.gpu)
    x = mx.array(features).astype(mx.float32)
    mean = mx.mean(x, axis=0)
    variance = mx.maximum(mx.mean(mx.square(x), axis=0) - mx.square(mean), 0.0)
    scale = mx.maximum(mx.sqrt(variance), 1e-6)
    standardized = (x - mean) / scale
    components, explained_variance, jacobi = _fit_standardized_basis(
        standardized,
        rank=rank,
        oversampling=oversampling,
        power_iterations=power_iterations,
        seed=seed,
    )
    gram = components @ components.T
    identity = mx.eye(rank, dtype=mx.float32)
    orthogonality = float(mx.max(mx.abs(gram - identity)).item())
    total_variance = float(mx.sum(mx.mean(mx.square(standardized), axis=0)).item())
    captured = float(mx.sum(explained_variance).item())
    residual = max(0.0, 1.0 - captured / max(total_variance, 1e-12))
    mx.eval(mean, scale, components, explained_variance)
    model = PCAModel(
        mean=np.asarray(mean, dtype=np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        components=np.asarray(components, dtype=np.float32),
        explained_variance=np.asarray(explained_variance, dtype=np.float32),
        total_variance=total_variance,
        power_iterations=power_iterations,
        oversampling=oversampling,
        seed=seed,
        orthogonality_max_abs=orthogonality,
        reconstruction_residual_fraction=residual,
        device="gpu:0",
    )
    mx.clear_cache()
    audit = {
        **jacobi,
        "orthogonality_max_abs": orthogonality,
        "orthogonality_pass": orthogonality <= PCA_ORTHOGONALITY_TOLERANCE,
        "explained_variance_nonincreasing": bool(
            np.all(np.diff(model.explained_variance.astype(np.float64)) <= 1e-6)
        ),
        "cumulative_explained_variance_monotonic": bool(
            np.all(np.diff(np.cumsum(model.explained_variance.astype(np.float64))) >= 0.0)
        ),
        "reconstruction_residual_fraction": residual,
    }
    return model, audit


def transform_pca_mlx(model: PCAModel, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != len(model.mean):
        raise ValueError("PCA transform width mismatch")
    mx.set_default_device(mx.gpu)
    x = mx.array(features).astype(mx.float32)
    standardized = (x - mx.array(model.mean)) / mx.array(model.scale)
    scores = standardized @ mx.array(model.components).T
    mx.eval(scores)
    output = np.asarray(scores, dtype=np.float32)
    mx.clear_cache()
    if not np.isfinite(output).all():
        raise ValueError("PCA scores are nonfinite")
    return output


def subspace_overlap(primary: np.ndarray, secondary: np.ndarray, width: int) -> float:
    """Mean squared cosine overlap between two leading component subspaces."""

    left = np.asarray(primary[:width], dtype=np.float64)
    right = np.asarray(secondary[:width], dtype=np.float64)
    if left.shape != right.shape or len(left) != width:
        raise ValueError("subspace components do not support requested width")
    cross = left @ right.T
    return float(np.clip(np.sum(np.square(cross)) / width, 0.0, 1.0))
