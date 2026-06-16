"""Check optional pretrained neuro-behavior component assets."""

import json
import os
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "behavior_component_registry.json"


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _resolve(component: Dict[str, Any], external_root: Path) -> Path:
    base = ROOT if component["location"] == "repository" else external_root
    return base / component["path"]


def _component_size(component: Dict[str, Any], path: Path) -> tuple[int, int]:
    pattern = component.get("glob")
    if not pattern:
        return _size_bytes(path), 1 if path.exists() else 0
    matches = [item for item in path.glob(pattern) if item.is_file()]
    return sum(item.stat().st_size for item in matches), len(matches)


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    env_name = registry["external_asset_root_env"]
    external_root = Path(os.environ.get(env_name, registry["external_asset_root_default"]))

    checks = []
    for component in registry["components"]:
        path = _resolve(component, external_root)
        size, matched_files = _component_size(component, path)
        minimum = int(component.get("minimum_bytes", 1))
        minimum_files = int(component.get("minimum_files", 1))
        checks.append(
            {
                "id": component["id"],
                "role": component["role"],
                "live_policy": component["live_policy"],
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": size,
                "minimum_bytes": minimum,
                "matched_files": matched_files,
                "minimum_files": minimum_files,
                "complete": path.exists() and size >= minimum and matched_files >= minimum_files,
            }
        )

    print(
        json.dumps(
            {
                "registry": str(REGISTRY_PATH),
                "external_asset_root": str(external_root),
                "components": checks,
            },
            indent=2,
        )
    )

    required = [item for item in checks if item["live_policy"] == "enabled"]
    if not all(item["complete"] for item in required):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
