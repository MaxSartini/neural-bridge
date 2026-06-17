"""Cortical ROI helpers for TRIBE fsaverage5 predictions."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..utils.logger import get_logger

logger = get_logger("neural_bridge.cortical_roi_mapper")

FSAVERAGE5_VERTICES_PER_HEMI = 10242
TRIBE_CORTICAL_VERTICES = FSAVERAGE5_VERTICES_PER_HEMI * 2


class CorticalRoiMapper:
    """Load the Destrieux surface atlas used by the cache viewer."""

    def __init__(self, atlas_dir: Optional[str] = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        self.atlas_dir = atlas_dir or os.path.join(project_root, "models", "neuro_atlases")

    def load_destrieux_atlas(self) -> Optional[dict[str, Any]]:
        try:
            from nilearn import datasets

            atlas = datasets.fetch_atlas_surf_destrieux(data_dir=self.atlas_dir, verbose=0)
            return {
                "labels": list(atlas.labels),
                "left": np.asarray(atlas.map_left),
                "right": np.asarray(atlas.map_right),
            }
        except Exception as exc:
            logger.warning("Failed to load Destrieux surface atlas: %s", exc)
            return None
