from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.phase02 import PHASE02_CHECKS

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "studies" / "veatic-2.1" / "phase-02-ar-baseline"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase02_compact_record_is_control_complete_and_hash_consistent() -> None:
    result = _load_json(RECORD / "result.json")
    decomposition = _load_json(RECORD / "ar-dominance-decomposition.json")
    provenance = _load_json(RECORD / "provenance.json")
    manifest = _load_json(RECORD / "artifact-manifest.json")

    assert set(result["checks"]) == set(PHASE02_CHECKS)
    assert all(result["checks"].values())
    assert result["status"] == "pass"
    assert result["phase03_authorized"] is True
    assert result["washout_activated"] is False
    assert result["protocol_cells"] == {"grouped_video": 5, "blocked_temporal": 1}
    assert result["eligible_rows"] == 19_169
    assert decomposition["target_history_overlap_rows"] == 0
    assert decomposition["washout_decision"]["activated"] is False
    assert all(
        cell["ar_delta_vs_chance"] > 0.0 and cell["ar_delta_vs_strongest_simple"] > 0.0
        for cell in decomposition["cells"]
    )

    external = provenance["external_hashes"]
    assert _sha256(RECORD / "result.json") == external["result_sha256"]
    assert _sha256(RECORD / "artifact-manifest.json") == external["artifact_manifest_sha256"]
    artifacts = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    for filename in (
        "request.json",
        "result.json",
        "report.md",
        "selected-hyperparameters.json",
        "ar-dominance-decomposition.json",
        "veatic-derivation-ledger.json",
    ):
        assert _sha256(RECORD / filename) == artifacts[filename]
    assert len([path for path in artifacts if path.startswith("predictions/")]) == 6
    assert len([path for path in artifacts if path.startswith("models/")]) == 6


def test_live_handoff_retains_phase02_hashes_after_phase03_progression() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    provenance = _load_json(RECORD / "provenance.json")

    assert all(value in current for value in provenance["external_hashes"].values())
    assert "## Concluded Phase 03 evidence" in current
    assert "Current Phase 02 execution: PASS" in current
    assert "Authorized phase: Phase 05 learned frozen-AR bridge only" in current
    assert "## Exact next action" in current
