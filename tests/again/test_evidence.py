from pathlib import Path

from neural_bridge.again.evidence import verify_evidence

ROOT = Path(__file__).resolve().parents[2]


def test_concluded_evidence_recomputes_from_compact_rows() -> None:
    phase5 = verify_evidence(
        "phase5-selected",
        ROOT
        / "studies/again/phase-05-learned-bridge/evidence"
        / "phase_5_5_selected_head_420_confirmation_20260714_124953",
    )
    blocked = verify_evidence(
        "phase7-blocked",
        ROOT / "studies/again/phase-07-continuous/blocked-confirmation",
    )
    grouped = verify_evidence(
        "phase7-grouped",
        ROOT / "studies/again/phase-07-continuous/grouped-confirmation",
    )

    assert phase5["verification_pass"] and phase5["scientific_pass"]
    assert blocked["verification_pass"] and not blocked["scientific_pass"]
    assert grouped["verification_pass"] and grouped["scientific_pass"]
