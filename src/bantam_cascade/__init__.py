"""Bantam Cascade: certified small-model + LLM-fallback cascades."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bantam-cascade")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "0.1.1"

__all__ = ["__version__"]
