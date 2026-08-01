from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "studies/veatic-2.1/phase-00-protected-input-foundation"


def test_phase00_compact_record_is_complete_and_nonpredictive() -> None:
    result = json.loads((RECORD / "result.json").read_text())
    provenance = json.loads((RECORD / "provenance.json").read_text())
    artifacts = json.loads((RECORD / "artifact-manifest.json").read_text())
    assert result["status"] == "pass"
    assert result["video_count"] == 124
    assert result["total_rows"] == 20_657
    assert result["mismatch_count"] == 0
    assert result["forbidden_hidden_state_count"] == 0
    assert result["writable_path_count"] == 0
    assert result["predictive_claim"] is False
    assert provenance["pca_fit"] is False
    assert provenance["predictive_model_fit"] is False
    assert provenance["again_runtime_or_artifact_used"] is False
    assert artifacts["bundle_manifest_sha256"] == provenance["bundle_manifest_sha256"]


def test_current_state_records_exact_phase00_hashes() -> None:
    state = (ROOT / "internal/handoff/CURRENT_STATE.md").read_text()
    artifacts = json.loads((RECORD / "artifact-manifest.json").read_text())
    provenance = json.loads((RECORD / "provenance.json").read_text())
    assert artifacts["bundle_manifest_sha256"] in state
    assert artifacts["external_artifact_manifest_sha256"] in state
    assert provenance["executed_commit"] in state
