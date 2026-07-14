"""Fail-closed Apple accelerator detection shared by research integrations."""

from __future__ import annotations

from dataclasses import dataclass

from ._optional import require_upstream


@dataclass(frozen=True)
class AcceleratorStatus:
    backend: str
    available: bool
    detail: str


def accelerator_status(backend: str) -> AcceleratorStatus:
    normalized = backend.lower()
    if normalized == "mlx":
        mx = require_upstream("mlx.core")
        try:
            mx.set_default_device(mx.gpu)
            device = str(mx.default_device())
            probe = mx.asarray([1.0]) + 1.0
            mx.eval(probe)
            available = "gpu" in device.lower()
            return AcceleratorStatus("mlx", available, device)
        except Exception as exc:  # pragma: no cover - hardware/runtime specific
            return AcceleratorStatus("mlx", False, f"{type(exc).__name__}: {exc}")
    if normalized == "mps":
        torch = require_upstream("torch")
        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
        detail = f"built={built},available={available}"
        if available:
            try:
                probe = torch.ones(1, device="mps") + 1
                torch.mps.synchronize()
                detail += f",device={probe.device}"
            except Exception as exc:  # pragma: no cover - hardware/runtime specific
                return AcceleratorStatus("mps", False, f"{detail},{type(exc).__name__}: {exc}")
        return AcceleratorStatus("mps", available, detail)
    raise ValueError("backend must be 'mlx' or 'mps'")


def require_accelerator(backend: str) -> AcceleratorStatus:
    status = accelerator_status(backend)
    if not status.available:
        raise RuntimeError(f"Required Apple accelerator {backend!r} is unavailable: {status.detail}")
    return status
