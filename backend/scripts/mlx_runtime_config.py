"""Runtime memory configuration helpers for MLX-heavy scripts."""

from __future__ import annotations

import os
from typing import Any


def _parse_limit_mb(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be a numeric megabyte value, got {raw!r}") from exc
    if value <= 0:
        return None
    return value


def configure_mlx_memory_from_env(mx_module: Any, *, label: str) -> dict[str, Any]:
    """Apply verified MLX memory limits from env vars.

    `MLX_MAX_MAPPED_MEM_MB` is not parsed by upstream MLX in this local install.
    We support it as a Neural Bridge compatibility alias by translating it to
    MLX's real `set_wired_limit(bytes)` API.
    """

    limit_mb = _parse_limit_mb("MLX_MAX_MAPPED_MEM_MB")
    source = "MLX_MAX_MAPPED_MEM_MB"
    if limit_mb is None:
        limit_mb = _parse_limit_mb("NEURAL_BRIDGE_MLX_WIRED_LIMIT_MB")
        source = "NEURAL_BRIDGE_MLX_WIRED_LIMIT_MB"
    if limit_mb is None:
        return {"mlx_memory_config_applied": False, "label": label}

    limit_bytes = int(limit_mb * 1024 * 1024)
    info = mx_module.device_info(mx_module.gpu)
    memory_size = int(info.get("memory_size", 0))
    recommended = int(info.get("max_recommended_working_set_size", 0))
    if memory_size and limit_bytes >= memory_size:
        raise ValueError(
            f"{source}={limit_mb} MB is >= total unified memory "
            f"{memory_size / 1024**2:.0f} MB; wired limit must remain below total memory"
        )
    if recommended and limit_bytes > recommended:
        raise ValueError(
            f"{source}={limit_mb} MB is above MLX device max_recommended_working_set_size "
            f"{recommended / 1024**2:.0f} MB. Raise the macOS system limit first with "
            f"`sudo sysctl iogpu.wired_limit_mb={limit_mb}`, then rerun."
        )
    previous = int(mx_module.set_wired_limit(limit_bytes))
    return {
        "mlx_memory_config_applied": True,
        "label": label,
        "source": source,
        "wired_limit_mb": limit_mb,
        "wired_limit_bytes": limit_bytes,
        "previous_wired_limit_bytes": previous,
        "device_name": info.get("device_name", ""),
        "memory_size_bytes": memory_size,
        "max_recommended_working_set_size_bytes": recommended,
    }
