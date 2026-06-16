"""Orchestrate loss-aware TRIBE translation and one compatible encoder stage."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

from app.services.neuro_encoder_router import NeuroEncoderRouter
from app.services.neuro_response_ir import NeuroResponseIRBuilder
from app.services.neuro_translation_report import NeuroTranslationReport
from app.services.sequential_stage_runner import SequentialStageRunner, StageSpec


class NeuroBridgePipeline:
    """Build translation artifacts, route safely, then run one heavy encoder."""

    def __init__(self, backend_root: str, moment_model_dir: str):
        self.backend_root = Path(backend_root).expanduser().resolve()
        self.moment_model_dir = str(Path(moment_model_dir).expanduser().resolve())

    def run(
        self,
        raw_tribe_path: str,
        output_dir: str,
        sampling_hz: float = 2.0,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        ir_metadata = NeuroResponseIRBuilder(sampling_hz=sampling_hz).build_from_npz(
            raw_tribe_path,
            str(target),
        )
        report_path = target / "neuro_translation_report.json"
        report = NeuroTranslationReport().write(str(target / "neuro_response_ir.json"), str(report_path))
        routing = report["encoder_routing"]
        stage = self._encoder_stage(routing["selected_encoder"], target)

        plan = {
            "raw_tribe_path": str(Path(raw_tribe_path).expanduser().resolve()),
            "output_dir": str(target),
            "translation_report_path": str(report_path),
            "routing": routing,
            "encoder_stage": {
                "name": stage.name,
                "command": stage.command,
                "required_outputs": stage.required_outputs,
            },
            "dry_run": dry_run,
        }
        (target / "neuro_bridge_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        if not dry_run:
            plan["stage_status"] = SequentialStageRunner(str(target / "encoder_stage")).run([stage])
            (target / "neuro_bridge_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        plan["ir_schema_version"] = ir_metadata["schema_version"]
        return plan

    def _encoder_stage(self, selected_encoder: str, target: Path) -> StageSpec:
        scripts = self.backend_root / "scripts"
        ir_path = target / "neuro_response_ir.npz"
        if selected_encoder == "moment_1_small":
            output = target / "moment_small_embedding.npz"
            return StageSpec(
                name="moment_1_small",
                command=[
                    sys.executable,
                    str(scripts / "run_moment_tribe_encoder.py"),
                    str(ir_path),
                    str(output),
                    "--model-dir",
                    self.moment_model_dir,
                ],
                required_outputs=[str(output), str(output.with_suffix(".json"))],
                timeout_seconds=600,
            )
        if selected_encoder == "brain_jepa":
            output = target / "brain_jepa_embedding.safetensors"
            return StageSpec(
                name="brain_jepa",
                command=[
                    sys.executable,
                    str(scripts / "run_brain_jepa_encoder.py"),
                    str(ir_path),
                    str(output),
                    "--work-dir",
                    str(target / "brain_jepa_work"),
                ],
                required_outputs=[str(output), str(Path(str(output) + ".json"))],
                timeout_seconds=600,
            )
        raise ValueError(f"Unsupported selected encoder: {selected_encoder}")
