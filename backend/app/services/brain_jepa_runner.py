"""Guarded subprocess runner for the Brain-JEPA Rust/Metal encoder."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from app.services.brain_jepa_adapter import BrainJepaAdapter


class BrainJepaRunner:
    """Run Brain-JEPA only when translation fidelity is explicit."""

    def __init__(
        self,
        binary_path: str,
        weights_path: str,
        gradient_path: str,
    ):
        self.binary_path = str(Path(binary_path).expanduser().resolve())
        self.weights_path = str(Path(weights_path).expanduser().resolve())
        self.gradient_path = str(Path(gradient_path).expanduser().resolve())

    def run(
        self,
        ir_path: str,
        output_path: str,
        work_dir: str,
        allow_incomplete_research_input: bool = False,
    ) -> Dict[str, Any]:
        work = Path(work_dir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)
        translated_npz = work / "brain_jepa_input.npz"
        translated_safe = work / "brain_jepa_input.safetensors"
        metadata = BrainJepaAdapter().build(
            ir_path,
            str(translated_npz),
            str(translated_safe),
        )
        if not metadata["production_eligible"] and not allow_incomplete_research_input:
            reasons = "; ".join(metadata["ineligibility_reasons"])
            raise ValueError(f"Refusing incomplete Brain-JEPA translation: {reasons}")

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary_path,
            "--weights",
            self.weights_path,
            "--gradient",
            self.gradient_path,
            "--input",
            str(translated_safe),
            "--output",
            str(target),
            "--model",
            "vit_base",
            "--verbose",
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        runtime_seconds = time.perf_counter() - started
        result = {
            **metadata,
            "research_ablation": not metadata["production_eligible"],
            "command": command,
            "returncode": completed.returncode,
            "runtime_seconds": runtime_seconds,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "embedding_path": str(target),
            "embedding_exists": target.exists() and target.stat().st_size > 0,
        }
        Path(str(target) + ".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if completed.returncode != 0 or not result["embedding_exists"]:
            raise RuntimeError(f"Brain-JEPA failed with return code {completed.returncode}")
        return result
