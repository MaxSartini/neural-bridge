"""Run real TRIBE text inference when the configured text encoder is ready."""

from argparse import ArgumentParser
import multiprocessing as mp

from app.config import Config
from app.services.tribe_adapter import TribeAdapter


def text_backend_ready(adapter: TribeAdapter) -> tuple[bool, str]:
    local_hf = adapter._resolve_path(Config.TRIBE_TEXT_ENCODER_LOCAL_DIR)
    local_mlx = adapter._resolve_path(Config.TRIBE_TEXT_ENCODER_MLX_DIR)
    if adapter._looks_like_transformers_model_dir(local_hf):
        return True, f"HF Transformers text encoder ready: {local_hf}"
    if adapter._looks_like_mlx_model_dir(local_mlx):
        return True, f"MLX text encoder ready: {local_mlx}"
    return False, f"No complete text encoder found. Checked HF={local_hf} MLX={local_mlx}"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail if text encoder is incomplete")
    parser.add_argument("--text", default="Clean water safety notice.")
    args = parser.parse_args()

    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    adapter = TribeAdapter()
    ready, reason = text_backend_ready(adapter)
    if not ready:
        if args.strict:
            raise SystemExit(reason)
        print({"skipped": True, "reason": reason})
        return

    result = adapter.predict(
        stimulus_text=args.text,
        stimulus_type="text",
        output_dir="backend/uploads/tribe_text_smoke_test",
        backend="apple_silicon_tribe",
    )
    print(result)
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
