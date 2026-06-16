"""Dependency-light compatibility exports for validation scoring."""

from app.services.neuro_validation_harness import NeuroValidationHarness, ValidationRecord

__all__ = ["NeuroValidationHarness", "ValidationRecord"]
