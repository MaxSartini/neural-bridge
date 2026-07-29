from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "fresh-method-rebuild-20260728/phase-01-label-alignment"
)
COMPACT = ROOT / "studies" / "veatic-2.1" / "phase-01-label-alignment"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"

RESULT_SHA256 = "feaae3f2f9b954786457dd816dd22f911d30c31492da42be682558ed84182710"
ARTIFACT_MANIFEST_SHA256 = (
    "ea7257732a6de79b67448bebfca75267242cd17b5e7cf6a8b984cebe0c6a551e"
)
CHECKSUMS_SHA256 = "0abe69e14a7d156cd5b950bcbcb7f44616965320e1c4bac70979522fa8ea1348"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase01_external_evidence_is_complete_and_sealed() -> None:
    result = json.loads((EXTERNAL / "result.json").read_text(encoding="utf-8"))
    assert _sha256(EXTERNAL / "result.json") == RESULT_SHA256
    assert _sha256(EXTERNAL / "artifact-manifest.json") == ARTIFACT_MANIFEST_SHA256
    assert _sha256(EXTERNAL / "checksums.sha256") == CHECKSUMS_SHA256
    assert result["phase01_pass"] is True
    assert result["mandatory_controls_passed"] == 28
    assert len(result["checks"]) == 28 and all(result["checks"].values())
    assert result["video_count"] == 124
    assert result["row_count"] == 20_657
    assert result["candidate_count"] == 231
    assert result["active_no_washout_candidate_count"] == 21
    assert result["prospective_washout_candidate_count"] == 210
    assert result["selected_target"] is None
    assert result["global_binary_label_stored"] is False
    assert result["outer_split_created"] is False
    assert result["tribe_cortical_values_loaded"] is False
    assert not any(result["operations"].values())


def test_phase01_arrays_preserve_alignment_quality_and_continuous_targets() -> None:
    with np.load(EXTERNAL / "aligned-labels.npz", allow_pickle=False) as payload:
        assert all(payload[key].shape == (20_657,) for key in payload.files)
        assert int(payload["quality_black_frame_flag"].sum()) == 76
        assert int(payload["quality_duplicate_frame_flag"].sum()) == 871
        assert int(payload["quality_exclusion_flag"].sum()) == 923
        assert set(np.unique(payload["source_match_quality_code"])) == {0, 1}
    with np.load(EXTERNAL / "target-substrate.npz", allow_pickle=False) as payload:
        values = payload["continuous_future_maximum_increase"]
        mask = payload["valid_mask"]
        assert values.shape == mask.shape == (231, 20_657)
        assert np.isfinite(values[mask]).all()
        assert np.isnan(values[~mask]).all()
        assert not any("binary" in key.lower() for key in payload.files)


def test_phase01_candidate_registry_is_complete_and_unselected() -> None:
    registry = json.loads((EXTERNAL / "candidate-registry.json").read_text(encoding="utf-8"))
    candidates = registry["candidates"]
    actual = {(row["future_start_rows"], row["future_end_rows"]) for row in candidates}
    expected = {(start, end) for end in range(1, 22) for start in range(1, end + 1)}
    assert len(candidates) == len(actual) == 231
    assert actual == expected
    assert sum(row["phase02_active"] for row in candidates) == 21
    assert sum(row["prospective_only"] for row in candidates) == 210
    assert all(row["eligible"] for row in candidates)
    assert all(row["eligible_videos"] == 124 for row in candidates)
    assert registry["selected_candidate"] is None
    assert registry["global_binary_label_stored"] is False


def test_phase01_compact_record_matches_external_evidence() -> None:
    for filename in (
        "result.json",
        "report.md",
        "artifact-manifest.json",
        "alignment-schema.json",
        "veatic-derivation-ledger.json",
    ):
        assert (COMPACT / filename).read_bytes() == (EXTERNAL / filename).read_bytes()
    provenance = json.loads((COMPACT / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["result_sha256"] == RESULT_SHA256
    assert provenance["artifact_manifest_sha256"] == ARTIFACT_MANIFEST_SHA256
    assert provenance["checksums_sha256"] == CHECKSUMS_SHA256


def test_current_state_records_phase01_and_authorizes_only_phase02() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    assert RESULT_SHA256 in current
    assert ARTIFACT_MANIFEST_SHA256 in current
    assert CHECKSUMS_SHA256 in current
    assert "Phase 01 execution and independent verification: PASS, 28/28" in current
    assert "Authorized action: commit and push the verified Stage B systems-backtest" in current
    assert "outer outcomes or cortical values" in current
