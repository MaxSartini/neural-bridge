"""Neutral, deterministic byte-level provenance utilities."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

IGNORED_NAMES = frozenset({".DS_Store", ".pytest_cache", "__pycache__"})


class TreeDigest(TypedDict):
    path: str
    sha256_tree: str
    files: int
    symlinks: int
    size_bytes: int


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> TreeDigest:
    """Match the canonical migration tree-identity algorithm byte for byte."""

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


def verify_tree_digest(
    root: Path, expected: Mapping[str, object], *, source: Path | str
) -> TreeDigest:
    """Return a tree identity only when it matches its recorded provenance."""

    verified = tree_digest(root)
    for field in ("sha256_tree", "files", "size_bytes"):
        expected_value = expected.get(field)
        if verified[field] != expected_value:
            raise ValueError(
                f"{source}: expected {field}={expected_value!r}, found {verified[field]!r}"
            )
    return verified
