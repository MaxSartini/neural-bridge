from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.phase03 import PHASE03_CHECKS

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "studies" / "veatic-2.1" / "phase-03-raw-cortical"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase03_compact_record_is_control_complete_and_hash_consistent() -> None:
    result = _load_json(RECORD / "result.json")
    summary = _load_json(RECORD / "summary.json")
    control = _load_json(RECORD / "control-matrix.json")
    provenance = _load_json(RECORD / "provenance.json")
    manifest = _load_json(RECORD / "artifact-manifest.json")

    assert set(result["checks"]) == set(PHASE03_CHECKS)
    assert all(result["checks"].values())
    assert result["status"] == "pass"
    assert result["phase04_authorized"] is True
    assert result["direct_raw_fusion_claim_pass"] is False
    assert result["lanes_per_cell"] == 17
    assert result["declared_raw_width"] == 20_484
    assert summary["promotion"]["direct_fusion_promoted_by_default"] is False
    assert control["declared_width"] == 20_484
    assert control["roles"]["no_video_architecture_ablation"]["applicable"] is False

    external = provenance["external_hashes"]
    assert _sha256(RECORD / "result.json") == external["result_sha256"]
    assert _sha256(RECORD / "artifact-manifest.json") == external["artifact_manifest_sha256"]
    artifacts = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    for filename in (
        "request.json",
        "control-matrix.json",
        "result.json",
        "summary.json",
        "primary-deltas.json",
        "report.md",
        "veatic-derivation-ledger.json",
    ):
        assert _sha256(RECORD / filename) == artifacts[filename]
    assert len([path for path in artifacts if path.startswith("predictions/")]) == 6
    assert len([path for path in artifacts if path.startswith("models/")]) == 6


def test_live_handoff_records_phase03_hashes_and_only_authorizes_phase04() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    provenance = _load_json(RECORD / "provenance.json")

    assert all(value in current for value in provenance["external_hashes"].values())
    assert "Authorized phase: Phase 04 fold-owned PCA bridge only" in current
    assert "Phase 05 learned bridge" in current
    assert "## Exact next action" in current
