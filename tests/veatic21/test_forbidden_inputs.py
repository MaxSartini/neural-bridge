from __future__ import annotations

from pathlib import Path

import pytest

from neural_bridge.veatic21.contracts import (
    FORBIDDEN_AGAIN_RUNTIME_ROOTS,
    FORBIDDEN_HIDDEN_STATE_FILENAME,
    LABEL_ARRAY_NAMES,
    PHASE00_ACCESSED_TRIBE_ARRAYS,
    VJEPA_ALLOWED_FILENAMES,
)
from neural_bridge.veatic21.data import reject_forbidden_runtime_path, sha256_file


def test_hidden_state_is_rejected_before_open_or_hash(tmp_path: Path) -> None:
    hidden = tmp_path / FORBIDDEN_HIDDEN_STATE_FILENAME
    hidden.write_bytes(b"must never be read")

    with pytest.raises(ValueError, match="forbidden V-JEPA hidden-state"):
        reject_forbidden_runtime_path(hidden)
    with pytest.raises(ValueError, match="forbidden V-JEPA hidden-state"):
        sha256_file(hidden)


def test_hidden_state_never_enters_vjepa_allowlist() -> None:
    assert FORBIDDEN_HIDDEN_STATE_FILENAME not in VJEPA_ALLOWED_FILENAMES


def test_phase00_feature_path_cannot_request_label_arrays() -> None:
    assert PHASE00_ACCESSED_TRIBE_ARRAYS.isdisjoint(LABEL_ARRAY_NAMES)


@pytest.mark.parametrize("root", FORBIDDEN_AGAIN_RUNTIME_ROOTS)
def test_again_runtime_roots_are_rejected(root: Path) -> None:
    with pytest.raises(ValueError, match="forbidden AGAIN runtime path"):
        reject_forbidden_runtime_path(root / "anything.npz")
