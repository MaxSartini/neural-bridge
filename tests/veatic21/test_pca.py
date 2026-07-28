from __future__ import annotations

import mlx.core as mx
import numpy as np

from neural_bridge.veatic21.pca import (
    _round_robin_pairs,
    fit_pca_mlx,
    jacobi_eigh_mlx,
    subspace_overlap,
    transform_pca_mlx,
)


def test_round_robin_jacobi_schedule_covers_each_pair_once() -> None:
    pairs = _round_robin_pairs(8)
    observed = {
        tuple(sorted((int(left), int(right))))
        for lefts, rights in pairs
        for left, right in zip(lefts, rights, strict=True)
    }
    assert len(pairs) == 7
    assert len(observed) == 8 * 7 // 2


def test_gpu_jacobi_matches_numpy_symmetric_eigenvalues() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(8, 8)).astype(np.float32)
    symmetric = source @ source.T
    values, vectors, audit = jacobi_eigh_mlx(mx.array(symmetric))
    mx.eval(values, vectors)

    expected = np.linalg.eigvalsh(symmetric)[::-1]
    assert audit["converged"]
    assert np.allclose(np.asarray(values), expected, rtol=2e-4, atol=2e-4)
    assert np.max(np.abs(np.asarray(vectors).T @ np.asarray(vectors) - np.eye(8))) < 1e-4


def test_randomized_pca_is_ordered_orthogonal_and_reconstructive() -> None:
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(160, 6))
    mixing = rng.normal(size=(6, 24))
    features = (latent @ mixing + 0.01 * rng.normal(size=(160, 24))).astype(np.float32)
    model, audit = fit_pca_mlx(features, rank=6, oversampling=4, seed=7, power_iterations=3)
    scores = transform_pca_mlx(model, features)

    assert model.device == "gpu:0"
    assert scores.shape == (160, 6)
    assert audit["orthogonality_pass"]
    assert audit["explained_variance_nonincreasing"]
    assert audit["cumulative_explained_variance_monotonic"]
    assert model.reconstruction_residual_fraction < 0.01


def test_subspace_overlap_is_rotation_invariant_within_prefix() -> None:
    identity = np.eye(6)
    rotated = identity.copy()
    rotated[:2] = np.asarray(((0.0, 1.0, 0, 0, 0, 0), (-1.0, 0.0, 0, 0, 0, 0)))
    assert subspace_overlap(identity, rotated, 2) == 1.0
