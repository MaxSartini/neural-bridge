"""Write a machine-readable TRIBE translation-quality report."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.neuro_translation_report import NeuroTranslationReport  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    print(json.dumps(NeuroTranslationReport().write(args.metadata_path, args.output_path), indent=2))


if __name__ == "__main__":
    main()
