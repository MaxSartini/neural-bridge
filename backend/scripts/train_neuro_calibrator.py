"""Train a grouped CatBoost calibrator from canonical IR manifest rows."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuro_core.neuro_calibration_model import NeuroCalibrationModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="JSONL rows with feature_path, group, and targets")
    parser.add_argument("output_dir")
    parser.add_argument("--model-version", default="openlav_v1")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    features = []
    feature_names = None
    groups = []
    targets = {}
    for row in rows:
        feature_path = Path(row["feature_path"]).expanduser()
        with np.load(feature_path) as bundle:
            features.append(np.asarray(bundle["calibration_feature_vector"], dtype=np.float32))
        metadata_path = feature_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_names = metadata["feature_contract"]["feature_names"]
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise ValueError(f"Feature contract mismatch in {feature_path}")
        groups.append(str(row["group"]))
        for axis, value in row["targets"].items():
            targets.setdefault(axis, []).append(float(value))
    if any(len(values) != len(rows) for values in targets.values()):
        raise ValueError("Every manifest row must provide every target axis")

    report = NeuroCalibrationModel().train(
        np.stack(features),
        {axis: np.asarray(values, dtype=np.float32) for axis, values in targets.items()},
        groups,
        args.output_dir,
        feature_names=feature_names,
        model_version=args.model_version,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
