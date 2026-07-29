from __future__ import annotations

import pytest

from neural_bridge.veatic21.phase02_stage_a_saturated_verify import (
    _ids_digest,
    _matrix_shape,
    _strict_json_bytes,
)


def test_verifier_rejects_nonfinite_json() -> None:
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        _strict_json_bytes(b'{"value": NaN}')


def test_verifier_configuration_identity_is_order_sensitive() -> None:
    assert _ids_digest(["a", "b"]) != _ids_digest(["b", "a"])


def test_verifier_rejects_changed_solver_matrix_shape() -> None:
    with pytest.raises(ValueError, match="solver column count changed"):
        _matrix_shape([[True], [False]], 2, 2, "solver")
