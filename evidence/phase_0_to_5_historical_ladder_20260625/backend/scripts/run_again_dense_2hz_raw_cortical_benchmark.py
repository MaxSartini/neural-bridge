"""CLI wrapper for dense AGAIN 2Hz raw cortical/temporal benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.scripts.again_dense_2hz_benchmark import raw_cortical_cli


if __name__ == "__main__":
    raw_cortical_cli()
