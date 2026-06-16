"""Verify standalone neuro imports do not initialize unrelated app stacks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import neuro_core  # noqa: E402


def main() -> None:
    assert neuro_core.NeuroResponseIRBuilder
    assert neuro_core.NeuroCalibrationModel
    unexpected = [name for name in ("flask", "flask_cors", "openai", "neo4j") if name in sys.modules]
    assert not unexpected, f"Standalone neuro imports initialized unrelated dependencies: {unexpected}"
    print({"neuro_core_imports_ok": True, "unexpected_imports": unexpected})


if __name__ == "__main__":
    main()
