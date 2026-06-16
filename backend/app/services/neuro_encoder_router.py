"""Choose a compatible neuro-response encoder without fabricating input."""

from pathlib import Path
from typing import Any, Dict

import numpy as np


class NeuroEncoderRouter:
    """Apply geometry/fidelity gates before accuracy-based model promotion."""

    def inspect(self, ir_path: str) -> Dict[str, Any]:
        source = Path(ir_path).expanduser().resolve()
        with np.load(source) as bundle:
            if "schaefer400_trajectories" not in bundle:
                raise ValueError("Canonical IR is missing schaefer400_trajectories")
            cortical = np.asarray(bundle["schaefer400_trajectories"])
            has_tian50 = "tian50_trajectories" in bundle
            tian_shape = list(bundle["tian50_trajectories"].shape) if has_tian50 else None
            has_aal424 = "aal424_trajectories" in bundle
            aal_shape = list(bundle["aal424_trajectories"].shape) if has_aal424 else None

        time_steps = int(cortical.shape[0])
        brain_jepa_compatible = bool(
            cortical.shape[1] == 400
            and has_tian50
            and tian_shape == [time_steps, 50]
            and time_steps >= 160
            and time_steps % 160 == 0
        )
        brain_dit_compatible = bool(has_aal424 and aal_shape == [200, 424])
        brainlm_compatible = bool(
            has_aal424
            and aal_shape is not None
            and aal_shape[1] == 424
            and aal_shape[0] > 0
        )
        return {
            "source_ir_path": str(source),
            "time_steps": time_steps,
            "has_schaefer400": cortical.shape[1] == 400,
            "has_tian50": has_tian50,
            "has_aal424": has_aal424,
            "brain_jepa_compatible": brain_jepa_compatible,
            "brain_dit_compatible": brain_dit_compatible,
            "brainlm_compatible": brainlm_compatible,
            "selected_encoder": "brain_jepa" if brain_jepa_compatible else "moment_1_small",
            "selection_reason": (
                "complete native Brain-JEPA geometry is available"
                if brain_jepa_compatible
                else "MOMENT is mask-aware; Brain-JEPA would require fabricated or unmasked values"
            ),
            "ineligible_candidates": {
                "brain_dit": (
                    None
                    if brain_dit_compatible
                    else "Requires a complete native 200 x AAL-424 trajectory."
                ),
                "brainlm": (
                    None
                    if brainlm_compatible
                    else "Requires a complete native AAL-424 trajectory and its model-specific preprocessing."
                ),
            },
            "accuracy_gate": "Selection remains provisional until paired held-out benchmarks promote a candidate.",
        }
