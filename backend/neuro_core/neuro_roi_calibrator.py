"""Dependency-light compatibility exports for cortical ROI calibration."""

from app.services.neuro_roi_calibrator import NeuroRoiCalibrator, TRIBE_CORTICAL_VERTICES

__all__ = ["NeuroRoiCalibrator", "TRIBE_CORTICAL_VERTICES"]
