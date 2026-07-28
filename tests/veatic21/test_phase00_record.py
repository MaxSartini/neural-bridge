from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.contracts import MANDATORY_CHECKS

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "studies" / "veatic-2.1" / "phase-00-dense-foundation"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase00_compact_record_is_self_consistent_and_control_complete() -> None:
    result = _load_json(RECORD / "result.json")
    provenance = _load_json(RECORD / "provenance.json")
    artifact_manifest = _load_json(RECORD / "artifact-manifest.json")
    checks = result["checks"]
    external_hashes = provenance["external_hashes"]

    assert isinstance(checks, dict)
    assert set(checks) == set(MANDATORY_CHECKS)
    assert all(checks.values())
    assert result["status"] == "pass"
    assert result["phase01_authorized"] is True
    assert isinstance(external_hashes, dict)
    assert _sha256(RECORD / "result.json") == external_hashes["result_sha256"]
    assert _sha256(RECORD / "artifact-manifest.json") == external_hashes["artifact_manifest_sha256"]

    artifacts = artifact_manifest["artifacts"]
    assert isinstance(artifacts, list)
    hashes = {item["path"]: item["sha256"] for item in artifacts}
    for filename in ("result.json", "report.md", "veatic-derivation-ledger.json"):
        assert _sha256(RECORD / filename) == hashes[filename]


def test_live_handoff_retains_concluded_phase00_hashes_after_progression() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    provenance = _load_json(RECORD / "provenance.json")
    external_hashes = provenance["external_hashes"]
    assert isinstance(external_hashes, dict)

    assert all(value in current for value in external_hashes.values())
    assert "## Concluded Phase 00 evidence" in current
    assert "Current Phase 00 execution: PASS" in current
    assert "## Exact next action" in current
