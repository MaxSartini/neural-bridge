"""Smoke-test isolated sequential execution and artifact validation."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.sequential_stage_runner import SequentialStageRunner, StageSpec  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        stages = [
            StageSpec(
                name="first",
                command=[sys.executable, "-c", "from pathlib import Path; Path('first.json').write_text('{}')"],
                required_outputs=["first.json"],
            ),
            StageSpec(
                name="second",
                command=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert Path('first.json').exists(); Path('second.json').write_text('{}')",
                ],
                required_outputs=["second.json"],
            ),
        ]
        result = SequentialStageRunner(temporary).run(stages)
        assert result["completed"]
        assert len(result["stages"]) == 2
        status = json.loads((Path(temporary) / "stage_status.json").read_text())
        assert status["completed"]
        print(json.dumps({"sequential_stage_runner_ok": True, "stages": 2}))


if __name__ == "__main__":
    main()
