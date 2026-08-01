"""Fresh VEATIC 2.1 implementation boundary."""

from neural_bridge.veatic21.benchmark import benchmark_bundle_topologies
from neural_bridge.veatic21.bundle import (
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_TRIBE_ROOT,
    DEFAULT_VJEPA_ROOT,
    assemble_bundle,
    assert_safe_delete_target,
    verify_bundle,
)
from neural_bridge.veatic21.phase00 import DEFAULT_PHASE00_ROOT, run_phase00, verify_phase00

__all__ = [
    "DEFAULT_BUNDLE_ROOT",
    "DEFAULT_TRIBE_ROOT",
    "DEFAULT_VJEPA_ROOT",
    "DEFAULT_PHASE00_ROOT",
    "assemble_bundle",
    "assert_safe_delete_target",
    "benchmark_bundle_topologies",
    "run_phase00",
    "verify_bundle",
    "verify_phase00",
]
