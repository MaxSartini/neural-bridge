from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "internal/active/veatic21-master-scientific-specification.md"
PROTOCOL = ROOT / "internal/active/veatic21-rebuild-protocol.md"
STATE = ROOT / "internal/handoff/CURRENT_STATE.md"

PROTECTED_ROOTS = (
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/",
)


def test_required_authority_files_exist() -> None:
    for path in (MASTER, PROTOCOL, STATE):
        assert path.is_file(), path
        assert path.read_text().strip(), path


def test_current_state_retains_mandatory_authority_anchors() -> None:
    text = STATE.read_text()
    assert "## Mandatory authority anchors" in text
    assert str(MASTER) in text
    assert str(PROTOCOL) in text
    assert str(STATE) in text


def test_phase00_is_the_only_authorized_scientific_phase() -> None:
    master = MASTER.read_text()
    state = STATE.read_text()
    assert "The only scientific phase authorized" in master
    assert "fresh Phase 00" in state
    assert "fit PCA" in state
    assert "may not" in state


def test_protected_roots_are_explicit_in_all_live_authorities() -> None:
    texts = [MASTER.read_text(), PROTOCOL.read_text(), STATE.read_text()]
    for protected_root in PROTECTED_ROOTS:
        for text in texts:
            assert protected_root in text


def test_hidden_states_are_forbidden_and_rows_are_the_alignment_authority() -> None:
    master = MASTER.read_text()
    protocol = PROTOCOL.read_text()
    for text in (master, protocol):
        assert "vjepa21_hidden_states.npz" in text
        assert "rows.csv" in text
        assert "cortical_prediction" in text
    assert "opened, memory-mapped, hashed, copied" in master
    assert "Join `rows.csv` row index `i` only to TRIBE payload position `i`" in protocol


def test_again_transfer_is_method_only() -> None:
    master = MASTER.read_text()
    assert "What transfers" in master
    assert "What never transfers" in master
    assert "must not import, execute, copy, adapt in place, or load any AGAIN" in master
    assert "AGAIN numerical answers are hypotheses, not VEATIC settings" in master


def test_phase_order_reaches_valence_then_zero_label() -> None:
    master = MASTER.read_text()
    valence = master.index("## Phase 08 — VEATIC-specific valence programme")
    zero_label = master.index("## Phase 09 — genuine zero-label-at-inference lane")
    assert valence < zero_label


def test_hardware_contract_requires_measurement_not_a_fixed_worker_count() -> None:
    master = MASTER.read_text()
    assert "fastest valid topology is measured per workload" in master
    assert "benchmark representative real cells" in master
    assert "Changing worker count" in master
