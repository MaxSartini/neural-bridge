from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "fresh-method-rebuild-20260728/phase-02-target-specific-ar/registration"
)
COMPACT = ROOT / "internal" / "active" / "veatic21-phase02-registration"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"

REGISTRATION_SHA256 = (
    "ab21c9b971fc0cf8aa18f4d77f585b8236db08bf810b1f83c79a476dfde44815"
)
RESULT_SHA256 = "0f3064d7ed7207879b195d8ed8ddb57fd440ed62f3922ee793159faee58a016e"
ARTIFACT_MANIFEST_SHA256 = (
    "39d5e587f4d1c1529039f8aae59137d3e07f6e265dda484bbf5797ccc6942a7c"
)
CHECKSUMS_SHA256 = "286db04959d3f8b1bd68b361115ed43772be1f3d6ae2e8a79f384a220d71754b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase02_registration_external_evidence_is_sealed() -> None:
    result = json.loads((EXTERNAL / "result.json").read_text(encoding="utf-8"))
    assert _sha256(EXTERNAL / "experiment-registration.json") == REGISTRATION_SHA256
    assert _sha256(EXTERNAL / "result.json") == RESULT_SHA256
    assert _sha256(EXTERNAL / "artifact-manifest.json") == ARTIFACT_MANIFEST_SHA256
    assert _sha256(EXTERNAL / "checksums.sha256") == CHECKSUMS_SHA256
    assert result["registration_pass"] is True
    assert result["active_target_count"] == 21
    assert result["grouped_outer_cells"] == 40
    assert result["blocked_outer_cells"] == 2
    assert result["outer_model_scores_opened"] is False
    assert result["outer_test_labels_opened"] is False
    assert result["cortical_values_opened"] is False
    assert not any(result["operations"].values())


def test_phase02_registration_split_and_support_coverage() -> None:
    splits = json.loads((EXTERNAL / "split-registry.json").read_text(encoding="utf-8"))
    support = json.loads((EXTERNAL / "support-audit.json").read_text(encoding="utf-8"))
    assert len(splits["grouped"]) == 40
    for repeat in range(4):
        test_videos = [
            video
            for row in splits["grouped"]
            if row["repeat"] == repeat
            for video in row["test_videos"]
        ]
        assert Counter(test_videos) == Counter({video: 1 for video in range(124)})
    assert [row["test_block_index"] for row in splits["blocked"]] == [2, 3]
    assert len(support["grouped"]) == 840
    assert len(support["blocked"]) == 42
    assert min(row["test_rows"] for row in support["grouped"]) == 1219
    assert min(row["outer_test_rows"] for row in support["blocked"]) == 2672
    assert min(row["inner_validation_rows"] for row in support["blocked"]) == 2630
    assert all(
        not row["outer_test_labels_opened"]
        for row in [*support["grouped"], *support["blocked"]]
    )


def test_phase02_registration_repo_freeze_matches_external() -> None:
    for filename in ("experiment-registration.json", "result.json", "report.md"):
        assert (COMPACT / filename).read_bytes() == (EXTERNAL / filename).read_bytes()
    provenance = json.loads((COMPACT / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["registration_sha256"] == REGISTRATION_SHA256
    assert provenance["result_sha256"] == RESULT_SHA256
    assert provenance["artifact_manifest_sha256"] == ARTIFACT_MANIFEST_SHA256
    assert provenance["checksums_sha256"] == CHECKSUMS_SHA256


def test_current_state_authorizes_only_frozen_phase02_execution() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    for digest in (
        REGISTRATION_SHA256,
        RESULT_SHA256,
        ARTIFACT_MANIFEST_SHA256,
        CHECKSUMS_SHA256,
    ):
        assert digest in current
    assert "Frozen Phase 02 registration evidence" in current
    assert "execute the frozen Phase 02 comprehensive target-specific AR benchmark" in current
    assert "Do not add, remove, or tune a target, split, history family" in current
    assert "Do not score cortical values" in current
