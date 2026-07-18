"""Canonical AGAIN scientific implementation.

The public API is intentionally small: prepare dense targets, then call the
single benchmark runner. Historical phase scripts are evidence, not imports.
"""

from .data import add_targets_and_ar_features
from .engine import RunConfig, evaluate_prepared_fold, run_sanity_benchmark
from .models import ResidualConfig

__all__ = [
    "ResidualConfig",
    "RunConfig",
    "add_targets_and_ar_features",
    "evaluate_prepared_fold",
    "run_sanity_benchmark",
]
