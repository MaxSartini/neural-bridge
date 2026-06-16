"""Ensure OpenLAV cache reuse rejects legacy feature schemas."""

import json
import tempfile
from pathlib import Path

from run_openlav_tribe_cache import feature_schema


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        assert feature_schema(root) is None
        (root / "neuro_response_ir.json").write_text(
            json.dumps({"feature_contract": {"schema_version": "neuro_calibration_features_v1"}}),
            encoding="utf-8",
        )
        assert feature_schema(root) == "neuro_calibration_features_v1"
        (root / "neuro_response_ir.json").write_text(
            json.dumps({"feature_contract": {"schema_version": "neuro_calibration_features_v2"}}),
            encoding="utf-8",
        )
        assert feature_schema(root) == "neuro_calibration_features_v2"
    print(json.dumps({"openlav_cache_skip_contract_ok": True}))


if __name__ == "__main__":
    main()
