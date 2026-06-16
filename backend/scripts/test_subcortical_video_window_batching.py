"""Tests for no-loss subcortical V-JEPA window batching controls."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.config import Config
from app.services.tribe_adapter import TribeAdapter


def test_isolated_subcortical_process_receives_video_window_batch_size() -> None:
    events = pd.DataFrame(
        [
            {
                "type": "Video",
                "start": 0.0,
                "duration": 1.0,
                "timeline": "default",
                "subject": "default",
                "filepath": "clip.webm",
            }
        ]
    )
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs["env"]
        output_index = args[0].index("--output") + 1
        Path(args[0][output_index]).parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(args[0][output_index], predictions=np.zeros((1, 8808)))
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    with patch.object(Config, "TRIBE_ENABLE_SUBCORTICAL", True), patch.object(
        Config, "TRIBE_SUBCORTICAL_VIDEO_WINDOW_BATCH_SIZE", 4
    ), patch("subprocess.run", side_effect=fake_run):
        result = TribeAdapter()._predict_subcortical_events_isolated(events)

    assert result.shape == (1, 8808)
    assert captured["env"]["TRIBE_VIDEO_WINDOW_BATCH_SIZE"] == "4"
    assert os.environ.get("TRIBE_VIDEO_WINDOW_BATCH_SIZE") is None


def test_neuralset_video_extractor_has_window_batching_patch() -> None:
    import neuralset.extractors.video as video

    source = Path(video.__file__).read_text(encoding="utf-8")
    assert "TRIBE_VIDEO_WINDOW_BATCH_SIZE" in source
    assert "np.stack([item[1] for item in batch_items], axis=0)" in source
