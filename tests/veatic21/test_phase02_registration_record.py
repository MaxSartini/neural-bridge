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
    "af1d35027c9b38758ead506efeefb6c0d313eb6af17a2837307076f360b24adf"
)
RESULT_SHA256 = "a9950066e87501fc444b2d46c01f52fde10a47149d83bdb04fc43140f9f23c1b"
ARTIFACT_MANIFEST_SHA256 = (
    "b4778af27016692bb670ac7ddd7e8a468e5baa59114de4641fc92f469c622945"
)
CHECKSUMS_SHA256 = "eae2c245e3fce49a18916312e3b6e88425eee3c9c5c70df3114324c3cfa5a9aa"


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
