"""CLI wrapper for the dense AGAIN 2Hz AR baseline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.scripts.again_dense_2hz_benchmark import ar_baseline_cli


if __name__ == "__main__":
    ar_baseline_cli()
