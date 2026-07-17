"""Audit the local Codex Mac app workflow and unified Neural Bridge graph."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
WORKSPACE = CODEX_HOME / "workspaces" / "neural-bridge-unified"
GRAPH_DB = WORKSPACE / ".codegraph" / "codegraph.db"

GRAPH_SUFFIXES = {
    ".astro", ".c", ".cc", ".cfc", ".cfm", ".cfs", ".cob", ".cpp",
    ".cs", ".cuh", ".cu", ".dart", ".dpk", ".dpr", ".erl", ".ets",
    ".go", ".h", ".hpp", ".hrl", ".java", ".js", ".jsx", ".kt",
    ".kts", ".liquid", ".lpr", ".lua", ".luau", ".m", ".metal",
    ".mjs", ".mm", ".mts", ".nix", ".pas", ".php", ".py", ".r",
    ".rb", ".rs", ".scala", ".sol", ".svelte", ".swift", ".tf",
    ".tofu", ".ts", ".tsx", ".vue", ".yaml", ".yml",
}
TEXT_SUFFIXES = {
    ".cfg", ".cmake", ".conf", ".css", ".csv", ".env", ".f", ".f90",
    ".html", ".ini", ".j2", ".jinja", ".json", ".jsonl", ".lock",
    ".md", ".rst", ".sh", ".toml", ".tsv", ".txt", ".xml",
}
HEAVY_SUFFIXES = {
    ".ckpt", ".dcm", ".fif", ".gz", ".mat", ".mkv", ".mov", ".mp3",
    ".mp4", ".npy", ".npz", ".parquet", ".pdf", ".pkl", ".pt",
    ".pth", ".safetensors", ".so", ".wav", ".webm", ".zip",
}
DEPENDENCY_DIRS = {
    ".cache", ".git", ".venv", "__pycache__", "build", "dist", "node_modules",
    "site-packages", "venv",
}
REQUIRED_HOOKS = {
    "PreToolUse", "PostToolUse", "SessionStart", "PreCompact",
    "UserPromptSubmit", "Stop",
}
REQUIRED_FORMAT_TOOLS = {
    "duckdb", "ffprobe", "file", "jq", "rg", "sqlite3", "yq",
}


def internal_repository_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [item for item in result.stdout.decode().split("\0") if item]


def graph_files() -> tuple[set[str], list[str]]:
    connection = sqlite3.connect(GRAPH_DB)
    try:
        paths = {row[0] for row in connection.execute("SELECT path FROM files")}
        errors = [
            f"{path}: {detail}"
            for path, detail in connection.execute(
                "SELECT path, errors FROM files "
                "WHERE errors IS NOT NULL AND errors != '[]'"
            )
        ]
        return paths, errors
    finally:
        connection.close()


def owned_graph_sources() -> set[str]:
    expected = {
        f"internal/{path}"
        for path in internal_repository_files()
        if Path(path).suffix.lower() in GRAPH_SUFFIXES
    }
    external = WORKSPACE / "external"
    if external.is_dir():
        for path in external.rglob("*"):
            if not path.is_file() or any(part in DEPENDENCY_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in GRAPH_SUFFIXES:
                expected.add(path.relative_to(WORKSPACE).as_posix())
    return expected


def inventory() -> dict[str, object]:
    counts: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    for root_name in ("internal", "external"):
        root = WORKSPACE / root_name
        for directory, dirs, names in os.walk(root, followlinks=True):
            dirs[:] = [name for name in dirs if name != ".git"]
            for name in names:
                suffix = Path(name).suffix.lower() or "[no-extension]"
                extensions[suffix] += 1
                if suffix in GRAPH_SUFFIXES:
                    counts["graph_source"] += 1
                elif suffix in TEXT_SUFFIXES:
                    counts["searchable_text"] += 1
                elif suffix in HEAVY_SUFFIXES:
                    counts["heavy_or_binary"] += 1
                else:
                    counts["other_metadata"] += 1
    return {
        "classes": dict(sorted(counts.items())),
        "top_extensions": extensions.most_common(20),
        "total_files": sum(counts.values()),
    }


def audit_desktop_config(errors: list[str]) -> None:
    config = tomllib.loads((CODEX_HOME / "config.toml").read_text(encoding="utf-8"))
    features = config.get("features", {})
    if not features.get("hooks") or not features.get("plugin_hooks"):
        errors.append("Codex desktop hook feature flags are not both enabled")

    servers = config.get("mcp_servers", {})
    for required in ("codegraph", "context-mode"):
        if required not in servers or servers[required].get("enabled") is False:
            errors.append(f"Codex desktop MCP is missing or disabled: {required}")
    if "codebase-memory-mcp" in servers:
        errors.append("retired codebase-memory MCP is still registered")

    plugins = config.get("plugins", {})
    if any(name.startswith("mempalace@") for name in plugins):
        errors.append("retired MemPalace plugin still has a desktop config entry")
    ponytail = plugins.get("ponytail@ponytail", {})
    if ponytail.get("enabled") is not True:
        errors.append("Ponytail is not enabled for the desktop app")

    hooks = json.loads((CODEX_HOME / "hooks.json").read_text(encoding="utf-8"))
    missing_hooks = REQUIRED_HOOKS - set(hooks.get("hooks", {}))
    if missing_hooks:
        errors.append(f"missing desktop lifecycle hooks: {sorted(missing_hooks)}")

    for skill in ("karpathy-guidelines", "caveman"):
        if not (CODEX_HOME / "skills" / skill / "SKILL.md").is_file():
            errors.append(f"missing desktop skill: {skill}")

    for tool in sorted(REQUIRED_FORMAT_TOOLS):
        if shutil.which(tool) is None:
            errors.append(f"missing format-routing tool: {tool}")


def main() -> int:
    errors: list[str] = []
    for link in ("internal", "external"):
        path = WORKSPACE / link
        if not path.exists():
            errors.append(f"unified workspace root is unavailable: {path}")
    if not GRAPH_DB.is_file():
        errors.append(f"unified CodeGraph database is unavailable: {GRAPH_DB}")

    graph: set[str] = set()
    expected: set[str] = set()
    graph_errors: list[str] = []
    if not errors:
        graph, graph_errors = graph_files()
        expected = owned_graph_sources()
        missing = sorted(expected - graph)
        if missing:
            errors.append(
                f"{len(missing)} owned source files are absent from CodeGraph: "
                + ", ".join(missing[:10])
            )
        if graph_errors:
            errors.append(
                f"{len(graph_errors)} CodeGraph files have parser errors: "
                + ", ".join(graph_errors[:5])
            )

    audit_desktop_config(errors)
    report = {
        "host": "Codex Mac desktop app",
        "unified_workspace": str(WORKSPACE),
        "owned_graph_coverage": {
            "expected": len(expected),
            "indexed": len(expected & graph),
            "missing": len(expected - graph),
        },
        "graph_total_files": len(graph),
        "graph_parser_errors": len(graph_errors),
        "inventory": inventory() if WORKSPACE.exists() else {},
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
