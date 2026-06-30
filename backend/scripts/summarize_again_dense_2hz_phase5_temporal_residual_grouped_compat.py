"""Summarize a temporal residual grouped compatibility output root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_grouped_compat as run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--pca-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    manifest = json.loads((output_root / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
    pca_root = Path(args.pca_root or manifest["grouped_pca_root"])
    finalized = run.finalize_output(output_root, pca_root, Path(args.reports_dir))
    print(json.dumps(fr.clean_json({"output_root": str(output_root), **finalized}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
