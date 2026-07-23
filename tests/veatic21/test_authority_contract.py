from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
MASTER = ROOT / "internal" / "active" / "veatic21-master-scientific-specification.md"
CHECKLIST = ROOT / "internal" / "active" / "veatic21-rebuild-protocol.md"
CURRENT = ROOT / "internal" / "handoff" / "CURRENT_STATE.md"

MASTER_ABSOLUTE = str(MASTER)
CHECKLIST_ABSOLUTE = str(CHECKLIST)
CURRENT_ABSOLUTE = str(CURRENT)


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
        "## Metric contract",
        "## PCA strategy and accuracy safeguards",
        "## Phase 00 implementation contract",
        "## Phase 01 exact next-stage contract",
        "## Phase 02 through zero-label execution sequence",
    )
    missing = [section for section in required_sections if section not in master]
    assert not missing, f"master scientific specification lost required sections: {missing}"
    assert len(master.splitlines()) >= 900, "master specification was unexpectedly shortened"


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
