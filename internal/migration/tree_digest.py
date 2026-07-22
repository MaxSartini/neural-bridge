#!/usr/bin/env python3
"""Print a deterministic content digest for one file or directory tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

IGNORED_NAMES = {".DS_Store", ".pytest_cache", "__pycache__"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = 0
    symlinks = 0
    size_bytes = 0
    paths = [root] if root.is_file() or root.is_symlink() else sorted(root.rglob("*"))
    for path in paths:
        if any(part in IGNORED_NAMES for part in path.relative_to(root).parts):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        relative = "." if path == root else path.relative_to(root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            digest.update(f"L\0{relative}\0{target}\n".encode())
            symlinks += 1
            continue
        size = path.stat().st_size
        content = file_digest(path)
        digest.update(f"F\0{relative}\0{size}\0{content}\n".encode())
        files += 1
        size_bytes += size
    return {
        "path": str(root.resolve()),
        "sha256_tree": digest.hexdigest(),
        "files": files,
        "symlinks": symlinks,
        "size_bytes": size_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if not args.path.exists() and not args.path.is_symlink():
        parser.error(f"path does not exist: {args.path}")
    print(json.dumps(tree_digest(args.path), sort_keys=True))


if __name__ == "__main__":
    main()
