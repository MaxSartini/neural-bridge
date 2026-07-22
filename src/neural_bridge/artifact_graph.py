"""Build an exact Graphify-compatible catalogue for external artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

DEFAULT_EXCLUDES = frozenset({"archive", "indexes", "quarantine", "scratch"})
DEFAULT_REPOSITORY = Path("/Users/maxsartini/Neural Bridge")
DEFAULT_ARTIFACTS = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
DEFAULT_INDEX_ROOT = DEFAULT_ARTIFACTS / "indexes" / "graphify"


def _artifact_id(relative_path: str) -> str:
    digest = hashlib.sha256(relative_path.encode()).hexdigest()
    return f"artifact::{digest}"


def iter_artifacts(root: Path, excluded_roots: frozenset[str]) -> Iterable[Path]:
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        if current == root:
            names[:] = sorted(name for name in names if name not in excluded_roots)
        else:
            names.sort()
        linked_directories = [name for name in names if (current / name).is_symlink()]
        names[:] = [name for name in names if name not in linked_directories]
        for name in linked_directories:
            yield current / name
        for name in sorted(files):
            if name not in {".DS_Store", ".graphifyignore"}:
                yield current / name


def build_artifact_graph(
    root: Path,
    output: Path,
    excluded_roots: frozenset[str] = DEFAULT_EXCLUDES,
) -> dict[str, object]:
    root = root.resolve(strict=True)
    nodes: list[dict[str, object]] = []
    total_bytes = 0
    for path in iter_artifacts(root, excluded_roots):
        try:
            stat = path.stat(follow_symlinks=False)
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        link_target = str(path.resolve(strict=False)) if path.is_symlink() else None
        total_bytes += stat.st_size
        nodes.append(
            {
                "id": _artifact_id(relative),
                "label": relative,
                "file_type": "artifact_alias" if link_target else "artifact",
                "source_file": str(path),
                "source_location": "",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "link_target": link_target,
                "suffix": path.suffix.lower(),
                "_origin": "artifact_catalog",
            }
        )

    graph = {
        "nodes": nodes,
        "edges": [],
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
        json.dump(graph, handle, separators=(",", ":"))
        temporary = Path(handle.name)
    temporary.replace(output)
    return {
        "artifact_count": len(nodes),
        "catalogued_bytes": total_bytes,
        "output": str(output),
        "root": str(root),
    }


def refresh_index(repository: Path, artifacts: Path, index_root: Path) -> dict[str, object]:
    graphify = shutil.which("graphify")
    if graphify is None:
        raise RuntimeError("graphify is not installed")

    repository = repository.resolve(strict=True)
    artifacts = artifacts.resolve(strict=True)
    index_root = index_root.resolve()
    repository_output = index_root / "repository"
    repository_graph = repository_output / "graphify-out" / "graph.json"
    catalog_graph = index_root / "catalog" / "graph.json"
    merged_graph = index_root / "merged" / "graph.json"

    catalog_summary = build_artifact_graph(artifacts, catalog_graph)
    subprocess.run(
        [
            graphify,
            "extract",
            str(repository),
            "--code-only",
            "--no-cluster",
            "--force",
            "--out",
            str(repository_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    merged_graph.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            graphify,
            "merge-graphs",
            str(repository_graph),
            str(catalog_graph),
            "--out",
            str(merged_graph),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with merged_graph.open() as handle:
        merged = json.load(handle)
    edges = merged.get("edges", merged.get("links"))
    if not isinstance(merged.get("nodes"), list) or not isinstance(edges, list):
        raise RuntimeError("Graphify produced an invalid merged graph")
    return {
        **catalog_summary,
        "merged_edges": len(edges),
        "merged_graph": str(merged_graph),
        "merged_nodes": len(merged["nodes"]),
        "repository": str(repository),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("index_root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = refresh_index(args.repository, args.artifacts, args.index_root)
    print(json.dumps(summary, sort_keys=True))
    return 0


def serve_mcp() -> None:
    """Refresh the merged graph before handing stdio directly to Graphify MCP."""
    graphify_mcp = shutil.which("graphify-mcp")
    if graphify_mcp is None:
        raise RuntimeError("graphify-mcp is not installed")
    summary = refresh_index(DEFAULT_REPOSITORY, DEFAULT_ARTIFACTS, DEFAULT_INDEX_ROOT)
    os.execv(graphify_mcp, [graphify_mcp, "--graph", str(summary["merged_graph"])])


if __name__ == "__main__":
    raise SystemExit(main())
