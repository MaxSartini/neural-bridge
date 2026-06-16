"""Run heavy inference stages in isolated processes with artifact handoffs."""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: List[str]
    required_outputs: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None


class SequentialStageRunner:
    """Execute one model stage at a time and require durable output artifacts."""

    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "stage_status.json"

    def run(self, stages: List[StageSpec]) -> Dict[str, object]:
        status: Dict[str, object] = {
            "run_dir": str(self.run_dir),
            "started_at": time.time(),
            "completed": False,
            "stages": [],
        }
        self._write_status(status)

        for stage in stages:
            result = self._run_stage(stage)
            status["stages"].append(result)
            self._write_status(status)
            if result["returncode"] != 0 or not result["outputs_complete"]:
                status["failed_stage"] = stage.name
                status["finished_at"] = time.time()
                self._write_status(status)
                raise RuntimeError(f"Sequential stage failed: {stage.name}")

        status["completed"] = True
        status["finished_at"] = time.time()
        self._write_status(status)
        return status

    def _run_stage(self, stage: StageSpec) -> Dict[str, object]:
        if not stage.command:
            raise ValueError(f"Stage {stage.name} has no command")

        log_path = self.run_dir / f"{stage.name}.log"
        started = time.time()
        env = os.environ.copy()
        env.update(stage.environment)
        env["NEURAL_BRIDGE_STAGE_RUN_DIR"] = str(self.run_dir)
        env["NEURAL_BRIDGE_STAGE_NAME"] = stage.name

        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                stage.command,
                cwd=self.run_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=stage.timeout_seconds,
                check=False,
            )

        outputs = [self._output_info(path) for path in stage.required_outputs]
        return {
            "name": stage.name,
            "command": stage.command,
            "returncode": completed.returncode,
            "started_at": started,
            "finished_at": time.time(),
            "log_path": str(log_path),
            "outputs": outputs,
            "outputs_complete": all(item["complete"] for item in outputs),
        }

    def _output_info(self, configured_path: str) -> Dict[str, object]:
        path = Path(configured_path)
        if not path.is_absolute():
            path = self.run_dir / path
        size = path.stat().st_size if path.exists() and path.is_file() else 0
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": size,
            "complete": path.exists() and (path.is_dir() or size > 0),
        }

    def _write_status(self, status: Dict[str, object]) -> None:
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(self.status_path)
