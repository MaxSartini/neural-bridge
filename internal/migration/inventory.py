#!/usr/bin/env python3
"""Build a read-only, hash-backed inventory for one migration slice."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


def sha256(path: Path, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        return "pending-large-file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, label: str, pattern: re.Pattern[str], max_bytes: int):
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        current_path = Path(current)

        for name in list(dirs):
            path = current_path / name
            if path.is_symlink():
                rel = path.relative_to(root)
                if pattern.search(str(rel)):
                    yield [label, str(path), "symlink", 0, "", os.readlink(path)]

        for name in sorted(files):
            path = current_path / name
            rel = path.relative_to(root)
            if not pattern.search(str(rel)):
                continue
            try:
                stat = path.lstat()
                if path.is_symlink():
                    yield [label, str(path), "symlink", 0, "", os.readlink(path)]
                else:
                    yield [
                        label,
                        str(path),
                        "file",
                        stat.st_size,
                        sha256(path, max_bytes),
                        "",
                    ]
            except OSError as error:
                yield [label, str(path), "error", 0, "", str(error)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", required=True)
    parser.add_argument(
        "--root", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True
    )
    parser.add_argument("--max-hash-mib", type=int, default=64)
    args = parser.parse_args()

    pattern = re.compile(args.pattern, re.IGNORECASE)
    rows = []
    for label, raw_root in args.root:
        root = Path(raw_root).resolve()
        rows.extend(inventory(root, label, pattern, args.max_hash_mib * 1024 * 1024))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["root", "path", "kind", "size_bytes", "sha256", "detail"])
        writer.writerows(rows)

    by_root = {label: 0 for label, _ in args.root}
    total_bytes = 0
    pending = 0
    for row in rows:
        by_root[row[0]] += 1
        total_bytes += int(row[3])
        pending += row[4] == "pending-large-file"
    print(f"rows={len(rows)} bytes={total_bytes} pending_large_hashes={pending}")
    print(" ".join(f"{label}={count}" for label, count in by_root.items()))


if __name__ == "__main__":
    main()
