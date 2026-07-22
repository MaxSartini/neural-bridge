"""Build an exact Graphify-compatible catalogue for external artifacts."""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

DEFAULT_EXCLUDES = frozenset({"archive", "indexes", "quarantine", "scratch"})
DEFAULT_REPOSITORY = Path("/Users/maxsartini/Neural Bridge")
DEFAULT_ARTIFACTS = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
DEFAULT_INDEX_ROOT = DEFAULT_ARTIFACTS / "indexes" / "graphify"
REPOSITORY_WATCH_EXCLUDES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "artifacts"}
)


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


def _watch_filter(_change: object, changed: str) -> bool:
    path = Path(changed)
    if path.name == ".DS_Store":
        return False
    try:
        relative = path.relative_to(DEFAULT_REPOSITORY)
    except ValueError:
        relative = None
    if relative is not None:
        return not any(part in REPOSITORY_WATCH_EXCLUDES for part in relative.parts)
    try:
        relative = path.relative_to(DEFAULT_ARTIFACTS)
    except ValueError:
        return False
    return not relative.parts or relative.parts[0] not in DEFAULT_EXCLUDES


def _watch_and_refresh(stop_event: threading.Event) -> None:
    from watchfiles import watch

    for _changes in watch(
        DEFAULT_REPOSITORY,
        DEFAULT_ARTIFACTS,
        watch_filter=_watch_filter,
        debounce=120_000,
        step=5_000,
        stop_event=stop_event,
    ):
        try:
            refresh_index(DEFAULT_REPOSITORY, DEFAULT_ARTIFACTS, DEFAULT_INDEX_ROOT)
        except Exception as error:  # pragma: no cover - operational recovery path
            print(f"Graphify refresh failed: {error}", file=sys.stderr, flush=True)


def _normalise_lookup(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_exact_node(query: str, nodes: list[dict[str, object]]) -> dict[str, object]:
    query = query.strip()
    normalised = _normalise_lookup(query)
    matches = []
    for node in nodes:
        label = str(node.get("label", ""))
        node_id = str(node.get("id", ""))
        source = str(node.get("source_file", ""))
        absolute_source = str(
            Path(source) if Path(source).is_absolute() else DEFAULT_REPOSITORY / source
        )
        exact_values = {node_id.lower(), label.lower(), source.lower(), absolute_source.lower()}
        normalised_values = {
            _normalise_lookup(label.removesuffix("()")),
            _normalise_lookup(source),
            _normalise_lookup(absolute_source),
        }
        if query.lower() in exact_values or normalised in normalised_values:
            matches.append(node)
    if len(matches) != 1:
        raise LookupError(
            f"Expected one exact Graphify node for {query!r}; found {len(matches)}. "
            "Use its exact symbol label, absolute path, repository path, or node ID."
        )
    return matches[0]


def _python_extent(path: Path, start_line: int) -> int | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    return next(
        (
            node.end_lineno
            for node in ast.walk(tree)
            if getattr(node, "lineno", None) == start_line
            and isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
        ),
        None,
    )


def _code_excerpt(
    path: Path,
    node: dict[str, object],
    nodes: list[dict[str, object]],
) -> tuple[int, int, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    location = str(node.get("source_location", ""))
    match = re.search(r"L(\d+)", location)
    start = int(match.group(1)) if match else 1
    end = _python_extent(path, start) if path.suffix == ".py" else None
    if end is None and match:
        following = sorted(
            int(other_match.group(1))
            for other in nodes
            if str(other.get("source_file", "")) == str(node.get("source_file", ""))
            and (other_match := re.search(r"L(\d+)", str(other.get("source_location", ""))))
            and int(other_match.group(1)) > start
        )
        end = following[0] - 1 if following else len(lines)
    end = end or len(lines)
    return start, end, "\n".join(lines[start - 1 : end])


def exact_node_payload(
    query: str,
    graph_path: Path = DEFAULT_INDEX_ROOT / "merged" / "graph.json",
) -> dict[str, object]:
    """Return the exact indexed code block or executable artifact handle."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise RuntimeError(f"Invalid Graphify graph: {graph_path}")
    node = _resolve_exact_node(query, nodes)
    source = Path(str(node.get("source_file", "")))
    path = source if source.is_absolute() else DEFAULT_REPOSITORY / source
    path = path.resolve(strict=True)
    allowed = (DEFAULT_REPOSITORY.resolve(), DEFAULT_ARTIFACTS.resolve())
    if not any(path == root or path.is_relative_to(root) for root in allowed):
        raise RuntimeError(f"Indexed source escaped Neural Bridge roots: {path}")
    payload: dict[str, object] = {
        "id": node.get("id"),
        "label": node.get("label"),
        "path": str(path),
        "type": node.get("file_type"),
    }
    if node.get("file_type") == "code" and path.is_file():
        start, end, content = _code_excerpt(path, node, nodes)
        payload.update({"start_line": start, "end_line": end, "content": content})
    else:
        stat = path.stat()
        payload.update(
            {
                "delivery": "direct_consumer_path",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return payload


async def _serve_exact_mcp(graph_path: Path) -> None:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server("neural-bridge-graphify")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="get_exact_neural_bridge_node",
                description=(
                    "Resolve exactly one current Neural Bridge code symbol or external artifact "
                    "from the merged Graphify index. Returns the actual bounded code block for "
                    "code, or the exact consumer path for large artifacts. Never returns "
                    "candidates."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Exact symbol label, absolute path, repository path, or Graphify ID"
                            ),
                        }
                    },
                    "required": ["query"],
                },
                annotations=types.ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, object]) -> list[types.TextContent]:
        if name != "get_exact_neural_bridge_node":
            raise ValueError(f"Unknown tool: {name}")
        payload = exact_node_payload(str(arguments["query"]), graph_path)
        return [types.TextContent(type="text", text=json.dumps(payload, separators=(",", ":")))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def serve_mcp() -> int:
    """Serve exact retrieval while keeping the combined Graphify index current."""
    summary = refresh_index(DEFAULT_REPOSITORY, DEFAULT_ARTIFACTS, DEFAULT_INDEX_ROOT)
    stop_event = threading.Event()
    watcher = threading.Thread(target=_watch_and_refresh, args=(stop_event,), daemon=True)
    watcher.start()
    try:
        asyncio.run(_serve_exact_mcp(Path(str(summary["merged_graph"]))))
        return 0
    finally:
        stop_event.set()
        watcher.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
