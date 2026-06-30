"""CLI wrapper for dense AGAIN 2Hz train-only PCA feature build."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.again_dense_2hz_phase4_pca_bridge import build_features_cli


if __name__ == "__main__":
    build_features_cli()
