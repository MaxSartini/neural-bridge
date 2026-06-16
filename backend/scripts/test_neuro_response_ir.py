"""Smoke-test loss-aware TRIBE response translation."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_response_ir import NeuroResponseIRBuilder  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(11)
    predictions = rng.normal(size=(8, 20484)).astype(np.float32)
    subcortical = rng.normal(size=(8, 8808)).astype(np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        result = NeuroResponseIRBuilder().build(
            predictions,
            temporary,
            subcortical_predictions=subcortical,
        )
        arrays = np.load(Path(temporary) / "neuro_response_ir.npz")
        assert arrays["parcel_trajectories"].shape[0] == 8
        assert arrays["schaefer400_trajectories"].shape == (8, 400)
        assert arrays["global_trajectories"].shape == (8, 4)
        assert arrays["subcortical_roi_trajectories"].shape == (8, 16)
        assert arrays["subcortical_summary_features"].shape == (80,)
        assert result["source"]["space"] == "fsaverage5"
        assert result["schema_version"] == "neuro_response_ir_v2"
        assert result["translations"]["harvard_oxford_subcortical"]["voxel_count"] == 8808
        assert result["translations"]["destrieux_parcels"]["reconstruction_rmse"] >= 0
        print(
            json.dumps(
                {
                    "neuro_response_ir_ok": True,
                    "parcel_shape": list(arrays["parcel_trajectories"].shape),
                    "schaefer400_shape": list(arrays["schaefer400_trajectories"].shape),
                    "explained_variance": result["translations"]["destrieux_parcels"]["explained_variance"],
                    "schaefer400_explained_variance": result["translations"]["schaefer400_cortical"]["explained_variance"],
                }
            )
        )


if __name__ == "__main__":
    main()
