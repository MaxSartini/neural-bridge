from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.phase01 import PHASE01_CHECKS

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "studies" / "veatic-2.1" / "phase-01-label-alignment"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase01_compact_record_is_control_complete_and_hash_consistent() -> None:
    result = _load_json(RECORD / "result.json")
    target = _load_json(RECORD / "target-registration.json")
    provenance = _load_json(RECORD / "provenance.json")
    manifest = _load_json(RECORD / "artifact-manifest.json")
    checks = result["checks"]
    external_hashes = provenance["external_hashes"]

    assert set(checks) == set(PHASE01_CHECKS)
    assert all(checks.values())
    assert result["status"] == "pass"
    assert result["phase02_authorized"] is True
    assert result["global_binary_label_stored"] is False
    assert result["outer_split_created"] is False
    assert target["initial_no_washout"]["start_row"] == 1
    assert target["initial_no_washout"]["end_row"] == 6
    assert target["prospective_washout"]["activated"] is False
    assert target["prospective_washout"]["candidate_starts"] == [5, 6]
    assert _sha256(RECORD / "result.json") == external_hashes["result_sha256"]
    assert _sha256(RECORD / "artifact-manifest.json") == external_hashes["artifact_manifest_sha256"]

    artifacts = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    for filename in ("result.json", "target-registration.json", "report.md"):
        assert _sha256(RECORD / filename) == artifacts[filename]


def test_live_handoff_retains_phase01_hashes_after_phase02_progression() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    provenance = _load_json(RECORD / "provenance.json")

    assert all(value in current for value in provenance["external_hashes"].values())
    assert "## Concluded Phase 02 evidence" in current
    assert "Current Phase 01 execution: PASS" in current
    assert "Authorized phase: Phase 03 raw cortical benchmark only" in current
    assert "## Exact next action" in current
