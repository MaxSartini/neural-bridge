import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import Config
from backend.app.services.mlx_vjepa21_cortical import _preprocess_video_batch
from backend.app.services.tribe_adapter import TribeAdapter


def test_tribe_adapter_selects_vjepa21_mlx_video_for_vjepa21_layout(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "vjepa21_mlx" / "vitg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "tensor_layout": "vjepa2_1_mlx_port",
                "hidden_size": 1408,
                "num_hidden_layers": 40,
                "num_attention_heads": 22,
                "image_size": 384,
                "frames_per_clip": 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "mlx")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_LOCAL_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_ID", "vjepa2_1_vit_giant_384")
    monkeypatch.setattr(Config, "TRIBE_VJEPA21_IMAGE_SIZE", 64)

    update = TribeAdapter()._config_update()

    assert update["data.video_feature.name"] == "MlxVjepa21Video"
    assert update["data.video_feature.mlx_weights_dir"] == str(model_dir)
    assert update["data.video_feature.image_size"] == 64
    assert "vjepa21" in update["data.video_feature.cache_model_name"]


def test_tribe_adapter_can_override_feature_frequency(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "vjepa21_mlx" / "vitg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")
    (model_dir / "config.json").write_text(
        json.dumps({"tensor_layout": "vjepa2_1_mlx_port", "hidden_size": 1408}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "mlx")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_FEATURE_FREQUENCY_HZ", 1.0)

    update = TribeAdapter()._config_update()

    assert update["data.frequency"] == 1.0


def test_tribe_adapter_can_configure_data_workers(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "vjepa21_mlx" / "vitg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")
    (model_dir / "config.json").write_text(
        json.dumps({"tensor_layout": "vjepa2_1_mlx_port", "hidden_size": 1408}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "mlx")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_DATA_NUM_WORKERS", 2)

    update = TribeAdapter()._config_update()

    assert update["data.num_workers"] == 2


def test_tribe_adapter_can_configure_video_frame_sampler(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "vjepa21_mlx" / "vitg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")
    (model_dir / "config.json").write_text(
        json.dumps({"tensor_layout": "vjepa2_1_mlx_port", "hidden_size": 1408}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "mlx")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_VIDEO_FRAME_SAMPLER", "ffmpeg")

    update = TribeAdapter()._config_update()

    assert update["data.video_feature.frame_sampler"] == "ffmpeg"


def test_tribe_adapter_can_configure_vjepa21_compile_and_cache_policy(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "vjepa21_mlx" / "vitg"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")
    (model_dir / "config.json").write_text(
        json.dumps({"tensor_layout": "vjepa2_1_mlx_port", "hidden_size": 1408}),
        encoding="utf-8",
    )
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "mlx")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_VJEPA21_COMPILE_ENCODER", True)
    monkeypatch.setattr(Config, "TRIBE_MLX_CLEAR_CACHE_EACH_WINDOW", False)
    monkeypatch.setattr(Config, "TRIBE_MLX_CLEAR_CACHE_EACH_VIDEO", True)

    update = TribeAdapter()._config_update()

    assert update["data.video_feature.compile_encoder"] is True
    assert update["data.video_feature.clear_cache_each_window"] is False
    assert update["data.video_feature.clear_cache_each_video"] is True


def test_tribe_adapter_clear_mlx_runtime_cache_calls_mlx_and_gc(monkeypatch):
    import mlx.core as mx
    import backend.app.services.tribe_adapter as tribe_adapter_module

    calls = []
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append("mlx"))
    monkeypatch.setattr(tribe_adapter_module.gc, "collect", lambda: calls.append("gc"))

    TribeAdapter._clear_mlx_runtime_cache("test")

    assert calls == ["mlx", "gc"]


def test_vjepa21_preprocess_skips_resize_for_square_model_sized_frames():
    frames = np.zeros((2, 3, 64, 64, 3), dtype=np.uint8)

    batch = _preprocess_video_batch(frames, image_size=64)

    assert batch.shape == (2, 3, 3, 64, 64)
    assert np.isfinite(batch).all()


def test_tribe_adapter_keeps_existing_vjepa2_mlx_video_for_old_layout(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "upstream-encoders-mlx" / "facebook-vjepa2-vitg-fpc64-256"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"placeholder")
    (model_dir / "config.json").write_text(json.dumps({"hidden_size": 1408}), encoding="utf-8")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "mlx")
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", str(model_dir))
    monkeypatch.setattr(Config, "TRIBE_VIDEO_ENCODER_LOCAL_DIR", str(model_dir))

    update = TribeAdapter()._config_update()

    assert update["data.video_feature.name"] == "MlxVjepa2Video"
    assert update["data.video_feature.mlx_weights_dir"] == str(model_dir)
    assert "data.video_feature.processor_model_name" in update


def test_tribe_adapter_coalesces_contiguous_direct_video_chunks():
    events = pd.DataFrame(
        [
            {
                "type": "Video",
                "filepath": "/tmp/again_video.webm",
                "start": 0.0,
                "offset": 0.0,
                "duration": 60.0,
                "stop": 60.0,
            },
            {
                "type": "Video",
                "filepath": "/tmp/again_video.webm",
                "start": 60.0,
                "offset": 60.0,
                "duration": 62.23,
                "stop": 122.23,
            },
        ]
    )

    coalesced = TribeAdapter._coalesce_direct_video_chunks(events, "/tmp/again_video.webm")

    assert len(coalesced) == 1
    assert float(coalesced.iloc[0]["start"]) == 0.0
    assert float(coalesced.iloc[0]["offset"]) == 0.0
    assert float(coalesced.iloc[0]["duration"]) == 122.23
    assert coalesced.attrs["neural_bridge_video_chunks_coalesced"]["original_video_event_count"] == 2


def test_tribe_adapter_does_not_coalesce_noncontiguous_video_chunks():
    events = pd.DataFrame(
        [
            {"type": "Video", "filepath": "/tmp/v.webm", "start": 0.0, "duration": 10.0, "stop": 10.0},
            {"type": "Video", "filepath": "/tmp/v.webm", "start": 12.0, "duration": 10.0, "stop": 22.0},
        ]
    )

    coalesced = TribeAdapter._coalesce_direct_video_chunks(events, "/tmp/v.webm")

    assert len(coalesced) == 2
