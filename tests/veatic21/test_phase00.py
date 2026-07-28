from __future__ import annotations

from pathlib import Path

import pytest

from neural_bridge.veatic21.contracts import MANDATORY_CHECK_NAMES, PHASE00_ROOT
from neural_bridge.veatic21.phase00 import build_result, run_phase00


def _result(checks: dict[str, bool]) -> dict[str, object]:
    return build_result(
        checks=checks,
        code_sha256="a" * 64,
        test_sha256="b" * 64,
        input_identity_sha256="c" * 64,
        tribe_tree_sha256="d" * 64,
        vjepa_tree_sha256="e" * 64,
        video_count=124,
        row_count=20_657,
        quality_counts={},
        source_match_counts={},
        runtime_firewall={},
    )


def test_phase01_authorization_requires_all_27_controls() -> None:
    checks = dict.fromkeys(MANDATORY_CHECK_NAMES, True)
    assert len(checks) == 27
    passed = _result(checks)
    assert passed["phase00_pass"] is True
    assert passed["single_next_authorized_action"] is not None

    checks[MANDATORY_CHECK_NAMES[-1]] = False
    failed = _result(checks)
    assert failed["phase00_pass"] is False
    assert failed["single_next_authorized_action"] is None


def test_phase00_refuses_noncanonical_output(tmp_path: Path) -> None:
    assert tmp_path != PHASE00_ROOT
    with pytest.raises(ValueError, match="canonical lifecycle root"):
        run_phase00(tmp_path)


def test_phase00_refuses_nonempty_output_before_reading_inputs(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("do not resume", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        run_phase00(tmp_path, enforce_canonical_output=False)
