"""CLI wrapper for dense AGAIN 2Hz Phase 4 PCA bridge summaries."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.again_dense_2hz_phase4_pca_bridge import summarize_cli


if __name__ == "__main__":
    summarize_cli()
