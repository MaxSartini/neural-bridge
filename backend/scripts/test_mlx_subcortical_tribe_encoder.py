"""Smoke-test the released subcortical checkpoint through the MLX runtime."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.mlx_subcortical_tribe_encoder import MlxSubcorticalTribeEncoder  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models" / "tribe" / "loganf26-tribev2-subcortical"
    encoder = MlxSubcorticalTribeEncoder(str(model_dir))
    features = {"text": np.zeros((1, 1, 1024, 4), dtype=np.float32)}
    mean, std = encoder.predict_with_uncertainty(features)
    assert mean.shape == (1, 8808, 100)
    assert std.shape == mean.shape
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()
    print(json.dumps({
        "mlx_subcortical_ok": True,
        "mean_shape": list(mean.shape),
        "mean_abs": float(np.mean(np.abs(mean))),
        "subject_disagreement_mean": float(np.mean(std)),
    }))


if __name__ == "__main__":
    main()
