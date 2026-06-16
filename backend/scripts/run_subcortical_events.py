"""Run the TRIBE subcortical branch in an isolated process."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.tribe_adapter import TribeAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    events = pd.read_pickle(args.events)
    predictions = TribeAdapter()._predict_subcortical_events(events)
    if predictions is None:
        raise SystemExit(1)
    np.savez_compressed(args.output, predictions=np.asarray(predictions))


if __name__ == "__main__":
    main()
