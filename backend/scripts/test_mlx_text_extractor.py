"""Smoke-test the MLX text feature extractor used by TRIBE on Apple Silicon."""

from argparse import ArgumentParser
from dataclasses import dataclass

from app.config import Config
from app.services.tribe_adapter import TribeAdapter
from app.services.mlx_text_extractor import MlxText


@dataclass
class WordEvent:
    text: str
    context: str


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="fail if the MLX model is incomplete")
    args = parser.parse_args()

    adapter = TribeAdapter()
    model_dir = adapter._resolve_path(Config.TRIBE_TEXT_ENCODER_MLX_DIR)
    if not adapter._looks_like_mlx_model_dir(model_dir):
        message = f"MLX text model is not complete yet: {model_dir}"
        if args.strict:
            raise SystemExit(message)
        print({"skipped": True, "reason": message})
        return

    extractor = MlxText(
        model_name=model_dir,
        event_types="Word",
        aggregation="sum",
        allow_missing=True,
        frequency=2.0,
        layers=[0.5, 0.75, 1.0],
        layer_aggregation="group_mean",
        token_aggregation="mean",
        cache_n_layers=20,
        contextualized=True,
        batch_size=1,
    )
    event = WordEvent(text="water", context="Clean water safety notice")
    feature = next(extractor._get_data([event]))
    print({
        "success": True,
        "model_dir": model_dir,
        "feature_shape": tuple(feature.shape),
        "feature_dtype": str(feature.dtype),
    })


if __name__ == "__main__":
    main()
