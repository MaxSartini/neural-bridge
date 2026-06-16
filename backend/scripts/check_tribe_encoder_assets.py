"""Check whether local TRIBE upstream encoder directories look usable.

This is intentionally a static check. It does not load multi-GB model weights.
"""

from pathlib import Path

from app.services.tribe_adapter import TribeAdapter
from app.config import Config


def check(label: str, path: str) -> dict[str, object]:
    adapter = TribeAdapter()
    resolved = Path(adapter._resolve_path(path))
    files = [p.name for p in resolved.iterdir()] if resolved.is_dir() else []
    return {
        "label": label,
        "path": str(resolved),
        "exists": resolved.is_dir(),
        "hf_encoder_layout": adapter._looks_like_encoder_model_dir(str(resolved)),
        "hf_transformers_layout": adapter._looks_like_transformers_model_dir(str(resolved)),
        "mlx_layout": adapter._looks_like_mlx_model_dir(str(resolved)),
        "file_count": len(files),
        "sample_files": files[:8],
    }


def main() -> None:
    for result in [
        check("text", Config.TRIBE_TEXT_ENCODER_LOCAL_DIR),
        check("text_mlx", Config.TRIBE_TEXT_ENCODER_MLX_DIR),
        check("audio", Config.TRIBE_AUDIO_ENCODER_LOCAL_DIR),
        check("video", Config.TRIBE_VIDEO_ENCODER_LOCAL_DIR),
    ]:
        print(result)


if __name__ == "__main__":
    main()
