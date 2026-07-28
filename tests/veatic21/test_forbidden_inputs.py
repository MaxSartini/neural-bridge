from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import (
    FORBIDDEN_HIDDEN_STATE_FILENAME,
    PHASE00_FEATURE_ARRAYS,
    VJEPA_ALLOWED_FILENAMES,
    InputBoundaryError,
    reject_forbidden_runtime_path,
    validate_runtime_manifest_paths,
)
from neural_bridge.veatic21.data import load_phase00_tribe_arrays, safe_sha256_file


def test_hidden_state_is_excluded_from_every_phase00_allowlist() -> None:
    assert FORBIDDEN_HIDDEN_STATE_FILENAME not in VJEPA_ALLOWED_FILENAMES
    assert FORBIDDEN_HIDDEN_STATE_FILENAME not in PHASE00_FEATURE_ARRAYS


def test_hidden_state_hash_is_rejected_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forbidden = tmp_path / FORBIDDEN_HIDDEN_STATE_FILENAME
    opened = False

    def fail_open(*args: object, **kwargs: object) -> None:
        nonlocal opened
        opened = True
        raise AssertionError("forbidden file was opened")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(InputBoundaryError, match="hidden-state"):
        safe_sha256_file(forbidden)
    assert opened is False


def test_hidden_state_load_is_rejected_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = False

    def fail_load(*args: object, **kwargs: object) -> None:
        nonlocal loaded
        loaded = True
        raise AssertionError("forbidden file was loaded")

    monkeypatch.setattr(np, "load", fail_load)
    with pytest.raises(InputBoundaryError, match="hidden-state"):
        load_phase00_tribe_arrays(
            tmp_path / FORBIDDEN_HIDDEN_STATE_FILENAME, ("cortical_prediction",)
        )
    assert loaded is False


def test_phase00_loader_cannot_request_label_arrays(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot request label arrays"):
        load_phase00_tribe_arrays(tmp_path / "tribe.npz", ("arousal",))


def test_again_code_study_output_and_artifact_paths_are_rejected() -> None:
    forbidden = (
        "/repo/src/neural_bridge/again/data.py",
        "/repo/studies/again/phase-00",
        "/artifacts/runs/again/phase-05",
        "/artifacts/features/again/fitted.npz",
    )
    for path in forbidden:
        with pytest.raises(InputBoundaryError, match="AGAIN runtime path"):
            reject_forbidden_runtime_path(path)
    validate_runtime_manifest_paths(
        (
            "/artifacts/features/veatic-2.1/tribe-v2",
            "/artifacts/runs/veatic-2.1/again-method-restart-20260723",
        )
    )


def test_symlink_to_forbidden_target_is_rejected_before_open(tmp_path: Path) -> None:
    forbidden = tmp_path / FORBIDDEN_HIDDEN_STATE_FILENAME
    alias = tmp_path / "apparently-safe.npz"
    alias.symlink_to(forbidden.name)

    with pytest.raises(InputBoundaryError, match="hidden-state"):
        safe_sha256_file(alias)
