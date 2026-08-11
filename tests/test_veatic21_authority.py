from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
STATE = ROOT / "internal/handoff/CURRENT_STATE.md"
ACTIVE = ROOT / "internal/active"
VEATIC_CODE = ROOT / "src/neural_bridge/veatic21"
VEATIC_TESTS = ROOT / "tests/veatic21"
VEATIC_STUDY = ROOT / "studies/veatic-2.1"

PROTECTED_ROOTS = (
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/veatic-2.1/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/",
)


def _authored(name: str) -> list[Path]:
    """Every file with this name that the repository itself authors.

    Vendored dependencies are excluded: an npm package is free to ship its own
    AGENTS.md (recharts does), and that says nothing about this repository's
    contract. Without the filter the assertion below fails on any machine that
    has run `npm install`, which trains people to ignore a governance test.
    """
    return sorted(
        path
        for path in ROOT.rglob(name)
        if "node_modules" not in path.parts and ".venv" not in path.parts
    )


def test_one_contract_and_one_live_state() -> None:
    assert _authored("AGENTS.md") == [AGENTS]
    assert _authored("CURRENT_STATE.md") == [STATE]


def test_no_active_veatic_plan_code_test_or_study_record() -> None:
    assert not any(path.is_file() for path in ACTIVE.rglob("*"))
    assert not VEATIC_CODE.exists()
    assert not VEATIC_TESTS.exists()
    assert not VEATIC_STUDY.exists()


def test_reset_and_autonomous_rebuild_are_authorized() -> None:
    agents = AGENTS.read_text()
    state = STATE.read_text()
    normalized_state = " ".join(state.split())
    assert "autonomously design, implement, train, infer, benchmark" in agents
    assert "autonomous evidence-led rebuild authorized" in state
    assert "no further user authorization is required" in state
    assert "/Users/maxsartini/Neural Bridge/studies/again/" in state
    assert "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/" in state
    assert "all five Phase 02 AR iterations" in state
    assert "Do not start from repository `src/`" in state
    assert "No VEATIC 2.1 target" in state
    assert "mandatory and exclusive source of VEATIC 2.1 cortical predictions" in state
    assert "not a general read restriction" in state
    assert "It must not be deleted, rebuilt, reassembled" in state
    assert "`per_video/0` through `per_video/123`" in state
    assert "The `tribe-v2` and `vjepa-2.1` roots are retired" in state
    assert "must not be invoked, enumerated, read, hashed, copied from" in normalized_state
    assert (
        "Training, inference, benchmarking, controls, and compute-intensive search are authorized"
        in normalized_state
    )


def test_protected_roots_are_explicit_in_both_authorities() -> None:
    for protected_root in PROTECTED_ROOTS:
        assert protected_root in AGENTS.read_text()
        assert protected_root in STATE.read_text()
