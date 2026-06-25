"""CLI wrapper for dense AGAIN 2Hz label alignment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.scripts.again_dense_2hz_benchmark import build_labels_cli


if __name__ == "__main__":
    build_labels_cli()
