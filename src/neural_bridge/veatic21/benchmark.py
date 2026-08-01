"""Reproducible real-input topology benchmark for the Phase 00 bundle workload."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.bundle import (
    DEFAULT_BUNDLE_ROOT,
    DEFAULT_TRIBE_ROOT,
    DEFAULT_VJEPA_ROOT,
    EXPECTED_VIDEO_IDS,
    _assemble_video,
    _verify_video,
)


def _map[T, R](function: Callable[[T], R], values: list[T], workers: int) -> list[R]:
    if workers == 1:
        return [function(value) for value in values]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, values))


def benchmark_bundle_topologies(
    *,
    worker_counts: tuple[int, ...],
    repeats: int,
    tribe_root: Path = DEFAULT_TRIBE_ROOT,
    vjepa_root: Path = DEFAULT_VJEPA_ROOT,
    scratch_parent: Path = DEFAULT_BUNDLE_ROOT.parent,
    expected_ids: tuple[str, ...] = EXPECTED_VIDEO_IDS,
) -> dict[str, Any]:
    """Time complete real per-video assembly and source-rehash verification workloads."""

    if repeats < 1 or not worker_counts or min(worker_counts) < 1:
        raise ValueError("repeats and every worker count must be positive")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for repeat in range(repeats):
        for workers in worker_counts:
            temporary = Path(tempfile.mkdtemp(prefix=".bundle-benchmark-", dir=scratch_parent))
            per_video = temporary / "per_video"
            per_video.mkdir()
            try:
                assembly_arguments = [
                    (str(tribe_root), str(vjepa_root), str(per_video), video_id)
                    for video_id in expected_ids
                ]
                started = time.perf_counter()
                assembled = _map(_assemble_video, assembly_arguments, workers)
                assembly_seconds = time.perf_counter() - started

                verification_arguments = [(str(temporary), video_id) for video_id in expected_ids]
                started = time.perf_counter()
                verified = _map(_verify_video, verification_arguments, workers)
                verification_seconds = time.perf_counter() - started
                runs.append(
                    {
                        "repeat": repeat,
                        "workers": workers,
                        "video_count": len(assembled),
                        "verified_video_count": len(verified),
                        "assembly_seconds": assembly_seconds,
                        "verification_seconds": verification_seconds,
                        "end_to_end_seconds": assembly_seconds + verification_seconds,
                    }
                )
            finally:
                shutil.rmtree(temporary)
    return {
        "worker_counts": list(worker_counts),
        "repeats": repeats,
        "video_ids": list(expected_ids),
        "runs": runs,
    }
