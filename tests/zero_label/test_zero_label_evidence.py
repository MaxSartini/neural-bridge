from pathlib import Path

from neural_bridge.zero_label import verify_locked_confirmation

ROOT = Path(__file__).resolve().parents[2]


def test_locked_confirmation_recomputes_from_compact_evidence() -> None:
    result = verify_locked_confirmation(
        ROOT / "studies/again/zero-label/locked-confirmation"
    )

    assert result["verification_pass"]
    assert result["scientific_pass"]
    assert result["rows"] == 140
    assert result["locked_videos"] == 299
    assert all(result["checks"].values())
