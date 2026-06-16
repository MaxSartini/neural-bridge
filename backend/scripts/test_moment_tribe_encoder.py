"""Smoke-test deterministic TRIBE trajectory preprocessing."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.moment_tribe_encoder import MomentTribeEncoder  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(7)
    predictions = rng.normal(size=(17, 20484)).astype(np.float32)
    encoder = MomentTribeEncoder("/tmp/not-loaded", device="cpu", channels=32)
    first, first_mask, metadata = encoder.prepare_series(predictions)
    second, second_mask, _ = encoder.prepare_series(predictions)
    amplitude = encoder.amplitude_features(predictions)
    assert first.shape == (1, 32, 512)
    assert np.array_equal(first, second)
    assert np.array_equal(first_mask, second_mask)
    assert first_mask.sum() == 17
    assert np.isfinite(first).all()
    assert metadata["cortical_vertices"] == 20484
    assert amplitude["mean"].shape == (20484,)
    assert amplitude["std"].shape == (20484,)
    assert amplitude["peak_abs"].shape == (20484,)
    print(json.dumps({"moment_tribe_preprocessing_ok": True, **metadata}))


if __name__ == "__main__":
    main()
