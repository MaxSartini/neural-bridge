"""Build and serve one exact Neural Bridge repository/artifact knowledge graph."""

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
from typing import cast

DEFAULT_EXCLUDES = frozenset({"indexes", "quarantine"})
DEFAULT_REPOSITORY = Path("/Users/maxsartini/Neural Bridge")
DEFAULT_ARTIFACTS = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
DEFAULT_INDEX_ROOT = DEFAULT_ARTIFACTS / "indexes" / "graphify"
REPOSITORY_WATCH_EXCLUDES = frozenset(
    {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "artifacts"}
)
INLINE_TEXT_SUFFIXES = frozenset(
    {".cfg", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
)
INLINE_TEXT_BYTES = 32 * 1024


def _catalog_id(namespace: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{relative_path}".encode()).hexdigest()
    return f"{namespace}::{digest}"


def iter_artifacts(root: Path, excluded_roots: frozenset[str]) -> Iterable[Path]:
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        names[:] = sorted(name for name in names if name not in excluded_roots)
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
    *,
    namespace: str = "artifact",
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
                "id": _catalog_id(namespace, relative),
                "label": relative,
                "file_type": "artifact_alias" if link_target else "artifact",
                "source_file": str(path),
                "source_location": "",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "link_target": link_target,
                "suffix": path.suffix.lower(),
                "_origin": f"{namespace}_catalog",
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
        f"{namespace}_count": len(nodes),
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
    repository_catalog_graph = index_root / "repository-catalog" / "graph.json"
    catalog_graph = index_root / "catalog" / "graph.json"
    merged_graph = index_root / "merged" / "graph.json"

    repository_catalog = build_artifact_graph(
        repository,
        repository_catalog_graph,
        REPOSITORY_WATCH_EXCLUDES,
        namespace="repository_file",
    )
    catalog_summary = build_artifact_graph(
        artifacts,
        catalog_graph,
        DEFAULT_EXCLUDES,
        namespace="artifact",
    )
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
            str(repository_catalog_graph),
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
        "repository_file_count": repository_catalog["repository_file_count"],
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


def _contextual_unique(
    matches: list[dict[str, object]], context: str | None
) -> dict[str, object] | None:
    if not context:
        return None
    terms = [_normalise_lookup(term) for term in re.findall(r"[A-Za-z0-9_.-]+", context)]
    terms = [term for term in terms if term]
    contextual = [
        node
        for node in matches
        if all(
            term
            in _normalise_lookup(
                f"{node.get('source_file', '')} {node.get('label', '')} {node.get('_origin', '')}"
            )
            for term in terms
        )
    ]
    return contextual[0] if len(contextual) == 1 else None


def _resolve_exact_node(
    query: str,
    nodes: list[dict[str, object]],
    context: str | None = None,
) -> dict[str, object]:
    query = query.strip()
    normalised = _normalise_lookup(query)
    query_lower = query.lower()

    id_matches = [node for node in nodes if str(node.get("id", "")).lower() == query_lower]
    if len(id_matches) == 1:
        return id_matches[0]

    path_matches = []
    for node in nodes:
        source = str(node.get("source_file", ""))
        absolute_source = str(
            Path(source) if Path(source).is_absolute() else DEFAULT_REPOSITORY / source
        )
        if query_lower in {source.lower(), absolute_source.lower()}:
            path_matches.append(node)
    catalog_path_matches = [
        node for node in path_matches if str(node.get("_origin", "")).endswith("_catalog")
    ]
    if len(catalog_path_matches) == 1:
        return catalog_path_matches[0]
    if len(path_matches) == 1:
        return path_matches[0]

    label_matches = []
    for node in nodes:
        label = str(node.get("label", ""))
        if query_lower == label.lower() or normalised == _normalise_lookup(
            label.removesuffix("()")
        ):
            label_matches.append(node)
    if len(label_matches) == 1:
        return label_matches[0]
    if contextual := _contextual_unique(label_matches, context):
        return contextual

    # Let callers use stable identities they naturally have (for example a qualified
    # Python name or a unique filename) without exposing search candidates. Every
    # relaxed match still has to resolve to exactly one node.
    path_suffix_matches = []
    query_path = query_lower.replace("\\", "/").removeprefix("./")
    for node in nodes:
        source = str(node.get("source_file", "")).lower().replace("\\", "/")
        if source == query_path or source.endswith(f"/{query_path}"):
            path_suffix_matches.append(node)
    catalog_suffix_matches = [
        node
        for node in path_suffix_matches
        if str(node.get("_origin", "")).endswith("_catalog")
    ]
    if len(catalog_suffix_matches) == 1:
        return catalog_suffix_matches[0]
    if contextual := _contextual_unique(catalog_suffix_matches, context):
        return contextual
    if len(path_suffix_matches) == 1:
        return path_suffix_matches[0]
    if contextual := _contextual_unique(path_suffix_matches, context):
        return contextual

    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query.removesuffix("()"))
    qualified_symbol = _normalise_lookup(identifiers[-1]) if identifiers else ""
    qualified_symbol_matches = []
    for node in nodes:
        if node.get("file_type") != "code":
            continue
        label = _normalise_lookup(str(node.get("label", "")).removesuffix("()"))
        if label and label == qualified_symbol:
            qualified_symbol_matches.append(node)
    if len(qualified_symbol_matches) == 1:
        return qualified_symbol_matches[0]
    if contextual := _contextual_unique(qualified_symbol_matches, context):
        return contextual

    matches = (
        catalog_path_matches
        or path_matches
        or label_matches
        or catalog_suffix_matches
        or path_suffix_matches
        or qualified_symbol_matches
    )
    if len(matches) != 1:
        raise LookupError(
            f"Expected one exact Graphify node for {query!r}; found {len(matches)}. "
            "Use a unique symbol or qualified symbol, path suffix, filename, absolute path, node "
            "ID, or distinguishing provenance context."
        )
    return matches[0]


def _inline_text(path: Path) -> str | None:
    if path.suffix.lower() not in INLINE_TEXT_SUFFIXES or path.stat().st_size > INLINE_TEXT_BYTES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError:
        return None


def _json_pointer_content(path: Path, pointer: str) -> str:
    if path.suffix.lower() != ".json":
        raise ValueError("JSON pointers can only be used with JSON files")
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("JSON pointer must be empty for the root or begin with '/'")
        for raw_token in pointer[1:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and token in value:
                value = cast(dict[str, object], value)[token]
            elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
                value = value[int(token)]
            else:
                raise LookupError(f"JSON pointer {pointer!r} does not exist in {path}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    *,
    context: str | None = None,
    json_pointer: str | None = None,
) -> dict[str, object]:
    """Return the exact indexed code block or executable artifact handle."""
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise RuntimeError(f"Invalid Graphify graph: {graph_path}")
    node = _resolve_exact_node(query, nodes, context)
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
        payload.update(
            {
                "delivery": "inline_content",
                "start_line": start,
                "end_line": end,
                "content": content,
            }
        )
    else:
        stat = path.stat()
        content = (
            _json_pointer_content(path, json_pointer)
            if path.is_file() and json_pointer is not None
            else _inline_text(path) if path.is_file() else None
        )
        payload.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if json_pointer is not None:
            payload["json_pointer"] = json_pointer
        if content is None or len(content.encode("utf-8")) > INLINE_TEXT_BYTES:
            payload["delivery"] = "direct_consumer_path"
        else:
            payload.update({"delivery": "inline_content", "content": content})
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
                    "Universal Neural Bridge retrieval: resolve exactly one current or historical "
                    "code symbol, repository file, script, configuration, document, result, model, "
                    "cache, or external artifact from the combined Graphify index. Returns content "
                    "when model-readable and the exact direct consumer path for heavy payloads. "
                    "An optional JSON pointer returns only the exact requested JSON value. Never "
                    "returns candidates or a routing pointer."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Unique symbol or qualified symbol, path suffix, filename, "
                                "absolute path, or Graphify ID"
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Optional provenance terms such as programme, phase, role, "
                                "lifecycle, or directory, used only to disambiguate repeated names"
                            ),
                        },
                        "json_pointer": {
                            "type": "string",
                            "description": (
                                "Optional RFC 6901 pointer for an exact bounded JSON value; use an "
                                "empty string only when the full JSON document is genuinely needed"
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
        context = arguments.get("context")
        json_pointer = arguments.get("json_pointer")
        payload = exact_node_payload(
            str(arguments["query"]),
            graph_path,
            context=str(context) if context is not None else None,
            json_pointer=str(json_pointer) if json_pointer is not None else None,
        )
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
