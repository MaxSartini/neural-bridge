"""Check MLX TRIBE encoder parity against the released PyTorch checkpoint."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.mlx_tribe_encoder import MlxTribeEncoder  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rng = np.random.default_rng(7)
    features = {"text": rng.normal(size=(1, 2, 3072, 4)).astype(np.float32)}

    sys.path.insert(0, str(root / "external_models/tribev2-apple-silicon"))
    from tribev2 import TribeModel  # noqa: E402

    model = TribeModel.from_pretrained(
        str(root / "models/tribe/facebook-tribev2"),
        cache_folder=str(root / "models/cache/tribev2"),
        device="cpu",
        config_update={"data.num_workers": 0, "data.batch_size": 1},
    )
    batch = SimpleNamespace(
        data={"text": torch.from_numpy(features["text"]), "subject_id": torch.zeros(1, dtype=torch.long)}
    )
    started = time.perf_counter()
    with torch.no_grad():
        expected = model._model(batch).cpu().numpy()
    torch_seconds = time.perf_counter() - started

    started = time.perf_counter()
    encoder = MlxTribeEncoder(str(root / "models/tribe-mlx/zimengxiong-tribev2-mlx"))
    mlx_load_seconds = time.perf_counter() - started
    encoder.predict(features)
    started = time.perf_counter()
    actual = encoder.predict(features)
    mlx_warm_seconds = time.perf_counter() - started
    max_diff = float(np.max(np.abs(expected - actual)))
    mean_diff = float(np.mean(np.abs(expected - actual)))
    assert max_diff < 1e-4, (max_diff, mean_diff)
    print({
        "tribe_mlx_parity_ok": True,
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "torch_seconds": torch_seconds,
        "mlx_load_seconds": mlx_load_seconds,
        "mlx_warm_seconds": mlx_warm_seconds,
    })


if __name__ == "__main__":
    main()
