"""Smoke-test the exact TRIBE subcortical Harvard-Oxford mapping."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.subcortical_roi_adapter import (  # noqa: E402
    SubcorticalRoiAdapter,
    TRIBE_SUBCORTICAL_VOXELS,
)


def main() -> None:
    predictions = np.zeros((3, TRIBE_SUBCORTICAL_VOXELS), dtype=np.float32)
    predictions[1] = 1.0
    projection = SubcorticalRoiAdapter().project(predictions)
    features, names = SubcorticalRoiAdapter().feature_vector(projection)

    assert projection["voxel_count"] == TRIBE_SUBCORTICAL_VOXELS
    assert projection["region_trajectories"].shape == (3, 16)
    assert sum(row["voxel_count"] for row in projection["region_metrics"]) == TRIBE_SUBCORTICAL_VOXELS
    assert "Left Amygdala" in projection["region_labels"]
    assert "Left Lateral Ventricle" in projection["interpretation_exclusions"]
    assert features.shape == (16 * 5,)
    assert len(names) == features.size
    print(
        {
            "subcortical_roi_adapter_ok": True,
            "voxel_count": projection["voxel_count"],
            "regions": len(projection["region_labels"]),
            "features": features.size,
        }
    )


if __name__ == "__main__":
    main()
