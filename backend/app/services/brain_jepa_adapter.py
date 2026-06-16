"""Translate canonical TRIBE IR into an explicitly masked Brain-JEPA input."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from safetensors.numpy import save_file


class BrainJepaAdapter:
    """Build 450 ROI x 160 time inputs without fabricating missing values."""

    CORTICAL_ROIS = 400
    SUBCORTICAL_ROIS = 50
    WINDOW_LENGTH = 160

    def build(
        self,
        ir_path: str,
        output_path: str,
        safetensors_path: str | None = None,
    ) -> Dict[str, Any]:
        source = Path(ir_path).expanduser().resolve()
        with np.load(source) as bundle:
            cortical = np.asarray(bundle["schaefer400_trajectories"], dtype=np.float32)
            subcortical = (
                np.asarray(bundle["tian50_trajectories"], dtype=np.float32)
                if "tian50_trajectories" in bundle
                else None
            )
        if cortical.ndim != 2 or cortical.shape[1] != self.CORTICAL_ROIS:
            raise ValueError(f"Expected time x 400 Schaefer trajectories, got {cortical.shape}")
        if subcortical is not None and subcortical.shape != (cortical.shape[0], self.SUBCORTICAL_ROIS):
            raise ValueError(f"Expected time x 50 Tian trajectories, got {subcortical.shape}")

        windows = []
        temporal_masks = []
        for start in range(0, max(1, cortical.shape[0]), self.WINDOW_LENGTH):
            chunk = cortical[start:start + self.WINDOW_LENGTH]
            values = np.zeros((self.CORTICAL_ROIS + self.SUBCORTICAL_ROIS, self.WINDOW_LENGTH), dtype=np.float32)
            values[: self.CORTICAL_ROIS, : chunk.shape[0]] = chunk.T
            if subcortical is not None:
                values[self.CORTICAL_ROIS :, : chunk.shape[0]] = subcortical[
                    start : start + self.WINDOW_LENGTH
                ].T
            mask = np.zeros(self.WINDOW_LENGTH, dtype=bool)
            mask[: chunk.shape[0]] = True
            windows.append(values)
            temporal_masks.append(mask)

        roi_mask = np.zeros(self.CORTICAL_ROIS + self.SUBCORTICAL_ROIS, dtype=bool)
        roi_mask[: self.CORTICAL_ROIS] = True
        if subcortical is not None:
            roi_mask[self.CORTICAL_ROIS :] = True
        stacked_windows = np.stack(windows, axis=0)
        stacked_temporal_masks = np.stack(temporal_masks, axis=0)
        temporal_coverage = float(stacked_temporal_masks.mean())
        roi_coverage = float(roi_mask.mean())
        production_eligible = bool(stacked_temporal_masks.all() and roi_mask.all())
        ineligibility_reasons = []
        if not stacked_temporal_masks.all():
            ineligibility_reasons.append("Brain-JEPA has no inference-time temporal availability mask")
        if not roi_mask.all():
            ineligibility_reasons.append("Brain-JEPA has no inference-time ROI availability mask")

        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            values=stacked_windows,
            roi_available_mask=roi_mask,
            temporal_available_mask=stacked_temporal_masks,
        )

        exported_safetensors = None
        if safetensors_path is not None:
            safe_target = Path(safetensors_path).expanduser().resolve()
            safe_target.parent.mkdir(parents=True, exist_ok=True)
            save_file({"fmri": stacked_windows[:, None, :, :]}, str(safe_target))
            exported_safetensors = str(safe_target)

        metadata = {
            "source_ir_path": str(source),
            "output_path": str(target),
            "safetensors_path": exported_safetensors,
            "target_model": "Brain-JEPA",
            "shape": list(stacked_windows.shape),
            "translation": "Schaefer-400 cortical channels copied without temporal rescaling; short windows zero-padded with explicit masks",
            "available_cortical_rois": self.CORTICAL_ROIS,
            "available_subcortical_rois": self.SUBCORTICAL_ROIS if subcortical is not None else 0,
            "missing_subcortical_rois": 0 if subcortical is not None else self.SUBCORTICAL_ROIS,
            "roi_coverage": roi_coverage,
            "temporal_coverage": temporal_coverage,
            "production_eligible": production_eligible,
            "ineligibility_reasons": ineligibility_reasons,
            "warning": "Brain-JEPA inference does not consume availability masks. Incomplete inputs are research ablations, not valid neuro-conditioning.",
        }
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
