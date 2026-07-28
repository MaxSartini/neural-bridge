from __future__ import annotations

import numpy as np

from neural_bridge.veatic21.residual import (
    derive_phase05_seed,
    derive_residual_recipes,
    fit_residual_checkpoint_mlx,
    predict_residual_mlx,
)


def test_phase05_seeds_are_deterministic_and_label_separated() -> None:
    digest = "f" * 64
    assert derive_phase05_seed(digest, "real") == derive_phase05_seed(digest, "real")
    assert derive_phase05_seed(digest, "real") != derive_phase05_seed(digest, "control")


def test_residual_recipes_are_derived_from_sealed_width() -> None:
    recipes = derive_residual_recipes(64)
    assert [(recipe.name, recipe.hidden_width) for recipe in recipes] == [
        ("linear", 0),
        ("relu-bottleneck-8", 8),
        ("relu-bottleneck-16", 16),
    ]
    assert all(recipe.learning_rate == 1.0 / 64 for recipe in recipes)
    assert all(recipe.max_steps == 128 for recipe in recipes)
    assert all(recipe.checkpoint_interval == 4 for recipe in recipes)


def test_suppressed_residual_returns_frozen_ar_bit_exactly() -> None:
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(96, 16)).astype(np.float32)
    validation_x = rng.normal(size=(48, 16)).astype(np.float32)
    train_y = np.tile(np.asarray((0, 1), dtype=np.int8), 48)
    validation_y = np.tile(np.asarray((0, 1), dtype=np.int8), 24)
    train_ar = np.where(train_y == 1, 0.9, 0.1).astype(np.float64)
    validation_ar = np.where(validation_y == 1, 0.9, 0.1).astype(np.float64)
    checkpoint, _ = fit_residual_checkpoint_mlx(
        train_x,
        train_y,
        train_ar,
        validation_x,
        validation_y,
        validation_ar,
        recipe=derive_residual_recipes(16)[0],
        seed=12,
    )

    output = predict_residual_mlx(checkpoint, validation_x, validation_ar)

    assert not checkpoint.active
    assert checkpoint.best_step == 0
    assert np.array_equal(output, validation_ar)


def test_residual_checkpoint_learns_signal_missing_from_ar_floor() -> None:
    rng = np.random.default_rng(8)
    train_x = rng.normal(size=(256, 16)).astype(np.float32)
    validation_x = rng.normal(size=(128, 16)).astype(np.float32)
    train_y = (train_x[:, 0] + 0.2 * train_x[:, 1] > 0).astype(np.int8)
    validation_y = (validation_x[:, 0] + 0.2 * validation_x[:, 1] > 0).astype(np.int8)
    train_ar = np.full(len(train_x), 0.5, dtype=np.float64)
    validation_ar = np.full(len(validation_x), 0.5, dtype=np.float64)
    checkpoint, audit = fit_residual_checkpoint_mlx(
        train_x,
        train_y,
        train_ar,
        validation_x,
        validation_y,
        validation_ar,
        recipe=derive_residual_recipes(16)[0],
        seed=23,
    )

    output = predict_residual_mlx(checkpoint, validation_x, validation_ar)

    assert checkpoint.active
    assert checkpoint.best_step > 0
    assert audit["restored_eval_mode"] is True
    assert checkpoint.best_validation_pr_auc > checkpoint.baseline_validation_pr_auc
    assert np.std(output) > 0.1
