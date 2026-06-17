"""Remove generated local artifacts without touching benchmark/model caches.

Default mode is a dry run. Use --apply to remove the listed files/directories.
The script deliberately protects video/TRIBE cache and benchmark data paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PROTECTED_TOP_LEVEL_PARTS = {
    ".git",
    ".venv",
    "models",
    "external_models",
    "benchmarks",
    "uploads",
}

PROTECTED_ANYWHERE_PARTS = {
    ".venv",
    "uploads",
    "tribe_cache",
    "video_windows",
}

EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(ROOT / "external_assets"))).expanduser()

PROTECTED_PREFIXES = (EXTERNAL_ROOT,)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def is_protected(path: Path) -> bool:
    resolved = path.resolve()
    if any(is_relative_to(resolved, prefix) for prefix in PROTECTED_PREFIXES):
        return True
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return True
    if relative.parts and relative.parts[0] in PROTECTED_TOP_LEVEL_PARTS:
        return True
    return any(part in PROTECTED_ANYWHERE_PARTS for part in relative.parts)


def collect_targets(args: argparse.Namespace) -> list[Path]:
    targets: set[Path] = set()

    for path in ROOT.rglob("__pycache__"):
        if path.is_dir() and "node_modules" not in path.parts and not is_protected(path):
            targets.add(path)

    for pattern in ("*.pyc", ".DS_Store"):
        for path in ROOT.rglob(pattern):
            if path.exists() and "node_modules" not in path.parts and not is_protected(path):
                targets.add(path)

    for path in ROOT.rglob(".pytest_cache"):
        if path.is_dir() and "node_modules" not in path.parts and not is_protected(path):
            targets.add(path)

    if args.include_logs:
        for pattern in ("*.log",):
            for path in ROOT.rglob(pattern):
                if path.exists() and not is_protected(path):
                    targets.add(path)

    if args.include_frontend_build:
        for path in (ROOT / "frontend" / "dist", ROOT / "frontend" / "node_modules"):
            if path.exists() and not is_protected(path):
                targets.add(path)

    return sorted(targets, key=lambda item: (len(item.parts), str(item)))


def remove_target(path: Path) -> None:
    if not path.exists():
        return
    if is_protected(path):
        raise RuntimeError(f"Refusing to remove protected path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually remove targets.")
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Also remove non-protected *.log files. Not enabled by default.",
    )
    parser.add_argument(
        "--include-frontend-build",
        action="store_true",
        help="Also remove frontend/dist and frontend/node_modules when not protected.",
    )
    args = parser.parse_args()

    targets = collect_targets(args)
    action = "remove" if args.apply else "would_remove"
    for target in targets:
        print(f"{action}\t{target.relative_to(ROOT)}")
        if args.apply:
            remove_target(target)
    print(f"{action}_count\t{len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
