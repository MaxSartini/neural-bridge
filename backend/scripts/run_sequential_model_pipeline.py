"""Run a JSON-defined sequence of isolated model stages."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.sequential_stage_runner import SequentialStageRunner, StageSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="JSON manifest containing run_dir and stages")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = Path(manifest.get("run_dir", manifest_path.parent / "sequential_run"))
    if not run_dir.is_absolute():
        run_dir = manifest_path.parent / run_dir

    stages = [
        StageSpec(
            name=item["name"],
            command=[str(value) for value in item["command"]],
            required_outputs=[str(value) for value in item.get("required_outputs", [])],
            environment={str(key): str(value) for key, value in item.get("environment", {}).items()},
            timeout_seconds=item.get("timeout_seconds"),
        )
        for item in manifest["stages"]
    ]
    result = SequentialStageRunner(str(run_dir)).run(stages)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
