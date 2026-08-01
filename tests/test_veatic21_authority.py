from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
MASTER = ROOT / "internal/active/veatic21-master-scientific-specification.md"
PROTOCOL = ROOT / "internal/active/veatic21-rebuild-protocol.md"
COMBINATION = ROOT / "internal/active/veatic21-supervised-spike-continuous-combination.md"
STATE = ROOT / "internal/handoff/CURRENT_STATE.md"
ACTIVE = ROOT / "internal/active"

PROTECTED_ROOTS = (
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/",
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/",
)


def test_required_authority_files_exist() -> None:
    for path in (AGENTS, MASTER, COMBINATION, PROTOCOL, STATE):
        assert path.is_file(), path
        assert path.read_text().strip(), path


def test_repository_contract_names_one_current_route_and_input() -> None:
    text = AGENTS.read_text()
    assert "The active VEATIC route is exactly" in text
    assert "`per_video/0` through `per_video/123`" in text
    assert "The only downstream VEATIC input" in text
    assert "Work only on `main`" in text


def test_repository_has_one_contract_and_one_live_state() -> None:
    assert list(ROOT.rglob("AGENTS.md")) == [AGENTS]
    assert list(ROOT.rglob("CURRENT_STATE.md")) == [STATE]


def test_active_tree_contains_only_current_authority_and_registration() -> None:
    files = {path.relative_to(ACTIVE).as_posix() for path in ACTIVE.rglob("*") if path.is_file()}
    assert files == {
        "veatic21-master-scientific-specification.md",
        "veatic21-rebuild-protocol.md",
        "veatic21-supervised-spike-continuous-combination.md",
        "veatic21-phase01-registration/README.md",
        "veatic21-phase01-registration/experiment-registration.json",
        "veatic21-phase01-registration/phase01-data-contract.json",
    }


def test_current_state_retains_mandatory_authority_anchors() -> None:
    text = STATE.read_text()
    assert "## Mandatory authority anchors" in text
    assert str(MASTER) in text
    assert str(COMBINATION) in text
    assert str(PROTOCOL) in text
    assert str(STATE) in text


def test_phase01_is_the_only_new_authorized_scientific_phase() -> None:
    master = MASTER.read_text()
    state = STATE.read_text()
    assert "The only new scientific phase authorized is Phase 01" in master
    assert "Phase 00 passed" in state
    assert "Phase 01 registration and implementation are the only active work" in state
    assert "fit a projection" in state
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


def test_combination_uses_every_consolidated_per_video_cortical_payload() -> None:
    text = COMBINATION.read_text()
    assert (
        "/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/"
    ) in text
    assert "`per_video/0` through `per_video/123`" in text
    assert "tribe_v2_cortical_predictions.npz:cortical_prediction" in text
    assert "partial video subset" in text
    assert "maximum positive arousal movement after a nonzero washout" in text


def test_again_transfer_is_method_only() -> None:
    master = MASTER.read_text()
    assert "Method-transfer firewall" in master
    assert "must not import, execute, copy, adapt in place, or load any AGAIN" in master
    assert "No source-dataset numeric value is a VEATIC setting" in master


def test_phase_order_reaches_valence_then_zero_label() -> None:
    master = MASTER.read_text()
    combination = master.index("## Phase 02 — supervised spike-and-continuous combination")
    valence = master.index("## Phase 03 — valence")
    zero_label = master.index("## Phase 04 — zero-label at inference")
    assert combination < valence
    assert valence < zero_label


def test_protocol_contains_no_competing_phase_ladder() -> None:
    protocol = PROTOCOL.read_text()
    headings = [line for line in protocol.splitlines() if line.startswith("## Phase ")]
    assert headings == [
        "## Phase 00 — protected-input foundation",
        "## Phase 01 — VEATIC targets, geometry, and ownership",
        "## Phase 02 — supervised spike + continuous combination",
        "## Phase 03 — valence",
        "## Phase 04 — zero-label at inference",
        "## Phase 05 — paper and product",
        "## Phase transition",
    ]


def test_hardware_contract_requires_measurement_not_a_fixed_worker_count() -> None:
    master = MASTER.read_text()
    assert "fastest valid topology is measured per workload" in master
    assert "benchmark representative real cells" in master
    assert "Changing worker count" in master
