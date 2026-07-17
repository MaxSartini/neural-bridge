from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from backend.scripts import veatic21_modeling as modeling


def test_model_spec_contracts() -> None:
    modeling.ModelSpec(
        head="neural_ar7", objective=modeling.CONTINUOUS, input_dim=7, pca_width=None
    ).validate()
    modeling.ModelSpec(
        head="short_temporal_conv_residual",
        objective=modeling.BINARY,
        input_dim=5 * 64 + 60,
        pca_width=64,
        condition_on_frozen_offset=True,
    ).validate()
    with pytest.raises(modeling.Veatic21ModelingError):
        modeling.ModelSpec(
            head="neural_ar7", objective=modeling.CONTINUOUS, input_dim=8, pca_width=None
        ).validate()
    with pytest.raises(modeling.Veatic21ModelingError):
        modeling.ModelSpec(
            head="short_temporal_conv_residual",
            objective=modeling.CONTINUOUS,
            input_dim=10,
            pca_width=64,
        ).validate()


def test_standardization_is_fit_only_on_supplied_rows() -> None:
    x = np.asarray([[0.0, 1.0], [2.0, 3.0], [100.0, 200.0]], dtype=np.float32)
    state = modeling.fit_standardization(x, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(state.mean, [1.0, 2.0])
    np.testing.assert_allclose(state.std, [1.0, 1.0])


def _tiny_continuous() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    train_x = rng.normal(size=(48, 7)).astype(np.float32)
    test_x = rng.normal(size=(12, 7)).astype(np.float32)
    target = (0.4 * train_x[:, 0] - 0.2 * train_x[:, 2]).astype(np.float32)
    mask = np.ones(len(train_x), dtype=bool)
    return train_x, test_x, target, mask


def test_continuous_training_resume_and_identity(tmp_path) -> None:
    train_x, test_x, target, mask = _tiny_continuous()
    kwargs = dict(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_loss_mask=mask,
        inner_train_idx=np.arange(0, 36, dtype=np.int64),
        inner_val_idx=np.arange(36, 48, dtype=np.int64),
        spec=modeling.ModelSpec(
            head="neural_ar7",
            objective=modeling.CONTINUOUS,
            input_dim=7,
            pca_width=None,
            hidden_dim=16,
        ),
        seed=123,
        checkpoint_path=tmp_path / "continuous.npz",
        artifact_identity={"contract": "sealed", "ownership": "fold1"},
        refit_after_selection=True,
        batch_size=32,
        max_epochs=3,
        patience=3,
    )
    first = modeling.train_scalar_model(**kwargs)
    assert first.cache_hit is False
    assert first.train_prediction.shape == (48,)
    assert first.test_prediction.shape == (12,)
    assert first.best_epoch in (1, 2, 3)
    assert np.isfinite(first.train_prediction).all()
    resumed = modeling.train_scalar_model(**kwargs)
    assert resumed.cache_hit is True
    np.testing.assert_allclose(first.test_prediction, resumed.test_prediction, atol=1e-6)
    with pytest.raises(modeling.Veatic21ModelingError, match="identity mismatch"):
        modeling.train_scalar_model(
            **{**kwargs, "artifact_identity": {"contract": "changed"}}
        )


def test_binary_bce_with_frozen_logit_offset(tmp_path) -> None:
    rng = np.random.default_rng(22)
    train_x = rng.normal(size=(60, 7)).astype(np.float32)
    test_x = rng.normal(size=(10, 7)).astype(np.float32)
    offset_train = (0.3 * train_x[:, 0]).astype(np.float32)
    offset_test = (0.3 * test_x[:, 0]).astype(np.float32)
    target = (train_x[:, 1] + offset_train > 0).astype(np.float32)
    result = modeling.train_scalar_model(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_loss_mask=np.ones(60, dtype=bool),
        inner_train_idx=np.arange(0, 45, dtype=np.int64),
        inner_val_idx=np.arange(45, 60, dtype=np.int64),
        spec=modeling.ModelSpec(
            head="flat_mlp_residual",
            objective=modeling.BINARY,
            input_dim=7,
            pca_width=64,
            hidden_dim=16,
            condition_on_frozen_offset=True,
        ),
        seed=456,
        checkpoint_path=tmp_path / "binary.npz",
        artifact_identity={"endpoint": "binary", "ar": "same-frozen-logit"},
        frozen_train_offset=offset_train,
        frozen_test_offset=offset_test,
        batch_size=32,
        max_epochs=2,
        patience=2,
    )
    assert result.train_probability is not None
    assert result.test_probability is not None
    assert np.all((result.test_probability >= 0) & (result.test_probability <= 1))
    np.testing.assert_allclose(
        result.train_prediction, result.train_correction + offset_train, atol=1e-6
    )


def test_overlap_and_offset_contracts_fail_closed(tmp_path) -> None:
    train_x, test_x, target, mask = _tiny_continuous()
    base = dict(
        train_x=train_x,
        test_x=test_x,
        train_target=target,
        train_loss_mask=mask,
        inner_train_idx=np.arange(0, 36, dtype=np.int64),
        inner_val_idx=np.arange(35, 48, dtype=np.int64),
        spec=modeling.ModelSpec(
            head="neural_ar7",
            objective=modeling.CONTINUOUS,
            input_dim=7,
            pca_width=None,
        ),
        seed=1,
        checkpoint_path=tmp_path / "bad.npz",
        artifact_identity={"x": 1},
        max_epochs=1,
        patience=1,
    )
    with pytest.raises(modeling.Veatic21ModelingError, match="overlap"):
        modeling.train_scalar_model(**base)
    conditioned = replace(
        base["spec"],
        head="flat_mlp_residual",
        input_dim=7,
        pca_width=64,
        condition_on_frozen_offset=True,
    )
    with pytest.raises(modeling.Veatic21ModelingError, match="requires frozen offsets"):
        modeling.train_scalar_model(
            **{
                **base,
                "inner_val_idx": np.arange(36, 48, dtype=np.int64),
                "spec": conditioned,
            }
        )


def test_fixed_epoch_all_data_refit_is_exact_and_resumable(tmp_path) -> None:
    train_x, _test_x, target, mask = _tiny_continuous()
    kwargs = dict(
        train_x=train_x,
        train_target=target,
        train_loss_mask=mask,
        spec=modeling.ModelSpec(
            head="neural_ar7",
            objective=modeling.CONTINUOUS,
            input_dim=7,
            pca_width=None,
            hidden_dim=16,
        ),
        seed=999,
        epochs=3,
        checkpoint_path=tmp_path / "all_data.npz",
        artifact_identity={"selection_frozen": True, "videos": 124},
        batch_size=32,
    )
    result = modeling.refit_scalar_model_fixed_epochs(**kwargs)
    assert result.best_epoch == 3
    assert len(result.curves) == 3
    assert result.test_prediction.size == 0
    resumed = modeling.refit_scalar_model_fixed_epochs(**kwargs)
    assert resumed.cache_hit is True
    np.testing.assert_allclose(result.train_prediction, resumed.train_prediction, atol=1e-6)

