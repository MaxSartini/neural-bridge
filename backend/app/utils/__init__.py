"""Utilities exposed through lazy imports."""

from importlib import import_module

__all__ = ['FileParser', 'LLMClient']

_MODULES = {"FileParser": "file_parser", "LLMClient": "llm_client"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
