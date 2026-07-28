from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "fresh-method-rebuild-20260728/phase-00-dense-foundation"
)
COMPACT = ROOT / "studies" / "veatic-2.1" / "phase-00-dense-foundation"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"

RESULT_SHA256 = "76667bc439af70b4ed212fe114922f0453415280fb64acf2910955d688333ffb"
ARTIFACT_MANIFEST_SHA256 = (
    "7a5e8dab2d442536eadd8b0d23491333c43a45355e74b361402c86abb7cc7e0e"
)
CHECKSUMS_SHA256 = "fea767403cf56919697aa228eb9053587d268cbe8a3f56143b7f08dac359ea8c"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase00_external_evidence_is_complete_and_sealed() -> None:
    result = json.loads((EXTERNAL / "result.json").read_text(encoding="utf-8"))
    assert _sha256(EXTERNAL / "result.json") == RESULT_SHA256
    assert _sha256(EXTERNAL / "artifact-manifest.json") == ARTIFACT_MANIFEST_SHA256
    assert _sha256(EXTERNAL / "checksums.sha256") == CHECKSUMS_SHA256
    assert result["phase00_pass"] is True
    assert result["mandatory_controls_passed"] == 27
    assert len(result["checks"]) == 27 and all(result["checks"].values())
    assert result["video_count"] == 124
    assert result["row_count"] == 20_657
    assert result["all_video_predictions_considered"] is True
    assert result["all_canonical_rows_considered"] is True
    assert result["vjepa_hidden_states_loaded"] is False
    assert result["vjepa_hidden_states_hashed"] is False
    assert result["vjepa_hidden_states_copied"] is False
    assert result["vjepa_hidden_states_inspected"] is False
    assert not any(result["operations"].values())


def test_phase00_row_inventory_proves_complete_2hz_coverage() -> None:
    with (EXTERNAL / "row-inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["video_id"] for row in rows] == [str(index) for index in range(124)]
    assert sum(int(row["row_count"]) for row in rows) == 20_657
    assert all(float(row["row_hz"]) == 2.0 for row in rows)
    assert all(float(row["time_step_seconds"]) == 0.5 for row in rows)
    assert all(float(row["time_start_seconds"]) == 0.0 for row in rows)
    assert all(int(row["cortical_width"]) == 20_484 for row in rows)
    assert all(row["cortical_dtype"] == "float16" for row in rows)
    assert sum(int(row["black_rows"]) for row in rows) == 76
    assert sum(int(row["duplicate_rows"]) for row in rows) == 871
    assert sum(int(row["both_rows"]) for row in rows) == 24
    assert sum(int(row["quality_union_rows"]) for row in rows) == 923


def test_phase00_compact_record_matches_external_evidence() -> None:
    for filename in (
        "result.json",
        "report.md",
        "artifact-manifest.json",
        "veatic-derivation-ledger.json",
    ):
        assert (COMPACT / filename).read_bytes() == (EXTERNAL / filename).read_bytes()
    provenance = json.loads((COMPACT / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["result_sha256"] == RESULT_SHA256
    assert provenance["artifact_manifest_sha256"] == ARTIFACT_MANIFEST_SHA256
    assert provenance["checksums_sha256"] == CHECKSUMS_SHA256
    assert provenance["video_count"] == 124
    assert provenance["row_count"] == 20_657


def test_current_state_retains_phase00_while_authorizing_only_phase02() -> None:
    current = CURRENT.read_text(encoding="utf-8")
    assert RESULT_SHA256 in current
    assert ARTIFACT_MANIFEST_SHA256 in current
    assert CHECKSUMS_SHA256 in current
    assert "Phase 00 execution: PASS, 27/27 mandatory controls" in current
    assert "Authorized action: execute the frozen Phase 02 comprehensive" in current
    assert "Cortical benchmark, PCA, head search" in current
