from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
MASTER = ROOT / "internal" / "active" / "veatic21-master-scientific-specification.md"
CHECKLIST = ROOT / "internal" / "active" / "veatic21-rebuild-protocol.md"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"
PROJECT_README = ROOT / "README.md"
METHODS_README = ROOT / "docs" / "README.md"
STUDY = ROOT / "studies" / "veatic-2.1"
PACKAGE = ROOT / "src" / "neural_bridge" / "veatic21"

MASTER_ABSOLUTE = str(MASTER)
CHECKLIST_ABSOLUTE = str(CHECKLIST)
CURRENT_ABSOLUTE = str(CURRENT)

TRIBE_PER_VIDEO_ROOT = (
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/"
    "veatic 2.1 raw cortical predictions/per_video"
)
VJEPA_ROOT = (
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/"
    "veatic 2.1 v jepa 2.1 stuff"
)
FRESH_LIFECYCLE_ROOT = (
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "fresh-method-rebuild-20260728"
)
DISALLOWED_ACTIVE_TEXT = (
    "tribe-v2/compact-",
    "vjepa-2.1/compact-",
    "/again-method-restart-",
    "Final TRIBE",
    "final-TRIBE",
    "sole real Neural Bridge representation",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"required authority document is missing: {path}"
    return path.read_text(encoding="utf-8")


def test_durable_master_retains_comprehensive_scientific_contract() -> None:
    master = _read(MASTER)

    required_sections = (
        "## Purpose and change control",
        "### Hard method-only transfer firewall",
        "## Canonical VEATIC input roots",
        "## Study-local AGAIN research map",
        "#### VEATIC-specific washout procedure",
        "## Control matrix required from the first applicable cell",
        "## Comprehensive VEATIC experiment-sufficiency contract",
        "## Metric contract",
        "## PCA strategy and accuracy safeguards",
        "## Phase 00 implementation contract",
        "## Phase 01 exact next-stage contract",
        "## Phase 02 through zero-label execution sequence",
    )
    missing = [section for section in required_sections if section not in master]
    assert not missing, f"master scientific specification lost required sections: {missing}"
    assert "Specification version: 2.0" in master
    assert len(master.splitlines()) >= 1_000, "master specification was unexpectedly shortened"


def test_live_handoff_preserves_mandatory_authority_anchors() -> None:
    current = _read(CURRENT)

    assert "## Mandatory authority anchors" in current
    assert MASTER_ABSOLUTE in current
    assert CHECKLIST_ABSOLUTE in current
    assert "## Live scientific state" in current
    assert "## Active execution contract" in current
    assert "## Exact next action" in current
    assert "Do not copy the master specification back into" in current


def test_repository_contract_protects_document_roles() -> None:
    agents = _read(AGENTS)

    assert MASTER_ABSOLUTE in agents
    assert CHECKLIST_ABSOLUTE in agents
    assert CURRENT_ABSOLUTE in agents
    assert "Never delete, replace wholesale, or shorten it" in agents
    assert "derived operational checklist and navigation aid" in agents
    assert "Every replacement must retain its `Mandatory authority anchors` section" in agents


def test_derived_checklist_defers_to_master_and_live_state() -> None:
    checklist = _read(CHECKLIST)

    assert "derived operational checklist and navigation aid" in checklist
    assert "not an independent source of scientific truth" in checklist
    assert "the master wins" in checklist
    assert "internal/active/veatic21-master-scientific-specification.md" in checklist
    assert "internal/handoff/CURRENT_STATE.md" in checklist


def test_all_active_authorities_use_exact_renamed_input_roots() -> None:
    master = _read(MASTER)
    checklist = _read(CHECKLIST)
    current = _read(CURRENT)

    for document in (master, checklist, current):
        assert TRIBE_PER_VIDEO_ROOT in document
        assert VJEPA_ROOT in document
        for disallowed in DISALLOWED_ACTIVE_TEXT:
            assert disallowed not in document

    assert FRESH_LIFECYCLE_ROOT in master
    assert FRESH_LIFECYCLE_ROOT in current


def test_complete_per_video_2hz_boundary_is_unambiguous() -> None:
    master = _read(MASTER)
    checklist = _read(CHECKLIST)
    current = _read(CURRENT)

    for document in (master, checklist, current):
        assert "all 124" in document.lower()
        assert "2 Hz" in document
        assert "cortical_prediction" in document
        assert "vjepa21_hidden_states.npz" in document

    assert "There is no single pooled or privileged" in master
    assert "There is no single pooled or privileged" in checklist
    assert "complete collection of all 124 per-video arrays" in current
    assert "851d55ccaac7c587495f65cdfbfbcf6bfe22a66a7ab3da2a048d0422e4087a60" in master
    assert "851d55ccaac7c587495f65cdfbfbcf6bfe22a66a7ab3da2a048d0422e4087a60" in current


def test_every_claim_bearing_phase_requires_fresh_comprehensive_search() -> None:
    master = _read(MASTER)
    checklist = _read(CHECKLIST)
    normalized_master = " ".join(master.split())

    required_master_text = (
        "Every phase and subphase",
        "one representation, one head, one optimizer, one training budget, or one seed",
        "Every attempted configuration",
        "search-sufficiency gate",
        "learning curves",
        "not negative",
    )
    for text in required_master_text:
        assert text in normalized_master

    assert "## Comprehensive experiment and search-sufficiency checklist" in checklist
    assert "full append-only experiment ledger" in checklist
    assert "one convenient implementation" in checklist
    assert "derive and justify the split proportions" in checklist
    assert "Do not import an AGAIN head" in checklist


def test_live_state_authorizes_only_phase02_after_fresh_phase01() -> None:
    current = _read(CURRENT)

    assert "fresh Phases 00 and 01 concluded; Phase 02 experiment registration" in current
    assert "Phase 00 implementation: complete" in current
    assert "Phase 00 execution: PASS, 27/27 mandatory controls" in current
    assert "Phase 01 execution and independent verification: PASS, 28/28" in current
    assert "Authorized action: execute the frozen Phase 02 comprehensive" in current
    assert "comprehensive target-specific AR" in current
    assert "all 21 active no-washout candidates" in current
    assert "210 prospective washout candidates" in current
    assert "Concluded Phase 00 evidence" in current
    assert "Concluded Phase 01 evidence" in current
    assert "Frozen Phase 02 registration evidence" in current
    assert PACKAGE.is_dir(), "concluded Phase 00 must retain its fresh implementation"
    phase_directories = {path.name for path in STUDY.glob("phase-*") if path.is_dir()}
    assert phase_directories == {"phase-00-dense-foundation", "phase-01-label-alignment"}


def test_surrounding_veatic_documentation_matches_fresh_authority() -> None:
    project = _read(PROJECT_README)
    methods = _read(METHODS_README)
    study = _read(STUDY / "README.md")

    assert TRIBE_PER_VIDEO_ROOT in project
    assert VJEPA_ROOT in project
    assert "Phase 00 independently passed all 27 mandatory controls" in project
    assert "Phase 01 independently passed all 28 mandatory controls" in project
    assert "all 21 no-washout" in project
    assert "one projection, one head, one optimizer, one budget, or one seed" in project
    assert "complete collection of 124 per-video predicted-cortical payloads" in methods
    assert "None of those counts or settings transfers to VEATIC 2.1" in methods
    assert "Fresh Phases 00 and 01 are concluded" in study
    assert "Phase 01 passed 28/28 controls" in study

    stale_claims = (
        "90 arousal-spike target hypotheses",
        "five fold-owned cortical PCA bases",
        "The immediate task is the full resumable fresh-AR benchmark",
    )
    for claim in stale_claims:
        assert claim not in project
