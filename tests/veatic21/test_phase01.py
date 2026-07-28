from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from neural_bridge.veatic21.contracts import PHASE01_MANDATORY_CHECK_NAMES, PHASE01_ROOT
from neural_bridge.veatic21.data import SupervisedRows
from neural_bridge.veatic21.phase01 import (
    _build_result,
    build_target_substrate,
    derive_candidate_pairs,
    future_maximum_increase,
    run_phase01,
    verify_phase01_output,
)


def _video(video_id: str, values: tuple[float, ...]) -> SupervisedRows:
    rows = len(values)
    return SupervisedRows(
        video_id=video_id,
        row_index=np.arange(rows, dtype=np.int32),
        time_seconds=np.arange(rows, dtype=np.float64) * 0.5,
        native_label_frame_count=np.full(rows, 100, dtype=np.int32),
        source_frame_position=np.arange(rows, dtype=np.float64) * 12.5,
        source_floor_frame_index=np.floor(np.arange(rows) * 12.5).astype(np.int32),
        source_ceil_frame_index=np.ceil(np.arange(rows) * 12.5).astype(np.int32),
        source_interp_alpha=np.mod(np.arange(rows) * 12.5, 1.0),
        source_match_quality=tuple("native_exact" for _ in values),
        arousal=np.asarray(values, dtype=np.float64),
        valence=np.asarray(values, dtype=np.float64),
    )


def test_candidate_registry_is_complete_lattice_from_shortest_video() -> None:
    candidates = derive_candidate_pairs((4, 6))
    assert candidates == ((1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3))
    assert len([pair for pair in candidates if pair[0] == 1]) == 3


def test_future_maximum_increase_uses_exact_inclusive_window() -> None:
    arousal = np.asarray([0.0, 1.0, 0.5, 2.0], dtype=np.float64)
    values, mask = future_maximum_increase(arousal, 1, 2)
    assert np.allclose(values[:2], [1.0, 1.0])
    assert np.isnan(values[2:]).all()
    assert np.array_equal(mask, [True, True, False, False])


def test_target_substrate_stores_continuous_values_and_masks_only() -> None:
    videos = (_video("0", (0.0, 1.0, 0.5, 2.0)), _video("1", (1.0, 0.0, 1.5, 1.0)))
    candidates = derive_candidate_pairs([video.row_count for video in videos])
    substrate, slices = build_target_substrate(videos, candidates)

    assert substrate["continuous_future_maximum_increase"].shape == (6, 8)
    assert substrate["valid_mask"].dtype == np.bool_
    assert len(slices) == 2
    assert not any("binary" in key or "event_label" in key for key in substrate)


def test_phase02_authorization_requires_every_phase01_control() -> None:
    checks = dict.fromkeys(PHASE01_MANDATORY_CHECK_NAMES, True)
    result = _build_result(
        checks,
        code_sha256="a" * 64,
        input_sha256="b" * 64,
        row_digest="c" * 64,
        label_digest="d" * 64,
        candidate_count=231,
        active_count=21,
        prospective_count=210,
    )
    assert result["phase01_pass"] is True
    assert result["single_next_authorized_action"] is not None

    checks[PHASE01_MANDATORY_CHECK_NAMES[-1]] = False
    failed = _build_result(
        checks,
        code_sha256="a" * 64,
        input_sha256="b" * 64,
        row_digest="c" * 64,
        label_digest="d" * 64,
        candidate_count=231,
        active_count=21,
        prospective_count=210,
    )
    assert failed["phase01_pass"] is False
    assert failed["single_next_authorized_action"] is None


def test_phase01_refuses_noncanonical_output(tmp_path: Path) -> None:
    assert tmp_path != PHASE01_ROOT
    with pytest.raises(ValueError, match="canonical root"):
        run_phase01(tmp_path)


def test_phase01_source_never_opens_cortical_payload() -> None:
    source = inspect.getsource(run_phase01)
    assert 'payload["cortical_prediction"]' not in source
    assert "np.load" not in source
    assert "vjepa21_hidden_states.npz" not in source


def test_phase01_verifier_refuses_nonexistent_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_phase01_output(tmp_path)
