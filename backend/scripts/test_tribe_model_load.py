"""Smoke-test local TRIBE checkpoint loading without running text/audio/video encoders."""

from app.services.tribe_adapter import TribeAdapter


def main() -> None:
    adapter = TribeAdapter()
    source = adapter._model_source()
    device = adapter._resolve_device()

    from tribev2 import TribeModel  # type: ignore

    model = TribeModel.from_pretrained(
        source,
        cache_folder=adapter._resolve_path("./models/cache/tribev2"),
        device=device,
    )
    print({
        "loaded": model.__class__.__name__,
        "source": source,
        "device": device,
    })


if __name__ == "__main__":
    main()
