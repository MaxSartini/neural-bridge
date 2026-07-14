"""Shared optional-dependency loading."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


class MissingResearchToolingError(RuntimeError):
    """Raised when an explicitly requested integration is not installed."""


def require_upstream(module_name: str) -> ModuleType:
    try:
        return import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only when the extra is absent
        raise MissingResearchToolingError(
            f"{module_name!r} is required; run `uv sync --extra research-tooling` "
            "from backend/."
        ) from exc
