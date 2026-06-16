"""Smoke-test TRIBE adapter availability without forcing model downloads."""

from app.services.tribe_adapter import TribeAdapter


def main() -> None:
    adapter = TribeAdapter()
    print({"available": adapter.is_available()})


if __name__ == "__main__":
    main()
