"""Audit the local Codex desktop app workflow and unified Neural Bridge graph."""

from __future__ import annotations

import json
import os
import select
import shlex
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
EXTERNAL_ROOT = Path(
    os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", WORKSPACE / "external")
).resolve()

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
    "UserPromptSubmit", "PermissionRequest", "Stop",
}
REQUIRED_FORMAT_TOOLS = {
    "codegraph", "context-mode", "duckdb", "ffprobe", "file", "jq", "rg",
    "rtk", "sqlite3", "yq",
}
CONTEXT_HOOK_ACTIONS = {
    "PreToolUse": "pretooluse",
    "PostToolUse": "posttooluse",
    "SessionStart": "sessionstart",
    "PreCompact": "precompact",
    "UserPromptSubmit": "userpromptsubmit",
    "Stop": "stop",
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
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("unified CodeGraph database integrity check failed")
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


def hook_commands(hooks: dict[str, object], event: str) -> list[str]:
    commands: list[str] = []
    for group in hooks.get("hooks", {}).get(event, []):
        commands.extend(
            item["command"]
            for item in group.get("hooks", [])
            if item.get("type") == "command" and item.get("command")
        )
    return commands


def command_exists(command: str) -> bool:
    executable = shlex.split(command)[0]
    return Path(executable).is_file() if Path(executable).is_absolute() else shutil.which(executable) is not None


def discovered_hooks() -> list[dict[str, object]]:
    process = subprocess.Popen(
        [shutil.which("codex") or "codex", "app-server", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None

    def request(payload: dict[str, object]) -> None:
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

    def response(request_id: int) -> dict[str, object]:
        while select.select([process.stdout], [], [], 10)[0]:
            row = json.loads(process.stdout.readline())
            if row.get("id") == request_id:
                return row
        raise RuntimeError(f"Codex app-server timed out waiting for response {request_id}")

    try:
        request({
            "method": "initialize",
            "id": 0,
            "params": {
                "clientInfo": {
                    "name": "neural_bridge_audit",
                    "title": "Neural Bridge Audit",
                    "version": "1.0.0",
                }
            },
        })
        response(0)
        request({"method": "initialized", "params": {}})
        request({
            "method": "hooks/list",
            "id": 1,
            "params": {"cwds": [str(ROOT)]},
        })
        result = response(1)
        return result["result"]["data"][0]["hooks"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def audit_desktop_config(errors: list[str]) -> dict[str, object]:
    config = tomllib.loads((CODEX_HOME / "config.toml").read_text(encoding="utf-8"))
    features = config.get("features", {})
    if not features.get("hooks") or not features.get("plugin_hooks"):
        errors.append("Codex desktop hook feature flags are not both enabled")

    servers = config.get("mcp_servers", {})
    for required in ("codegraph", "context-mode"):
        server = servers.get(required)
        if not server or server.get("enabled") is False:
            errors.append(f"Codex desktop MCP is missing or disabled: {required}")
        elif not command_exists(server.get("command", "")):
            errors.append(f"Codex desktop MCP command is unavailable: {required}")
    if servers.get("context-mode", {}).get("env", {}).get("CONTEXT_MODE_PLATFORM") != "codex":
        errors.append("Context-Mode desktop MCP is not pinned to Codex storage")
    if servers.get("codegraph", {}).get("args") != [
        "serve", "--mcp", "--path", str(WORKSPACE)
    ]:
        errors.append("CodeGraph desktop MCP is not pinned to unified workspace")
    if "codebase-memory-mcp" in servers:
        errors.append("retired codebase-memory MCP is still registered")

    plugins = config.get("plugins", {})
    if any(name.startswith("mempalace@") for name in plugins):
        errors.append("retired MemPalace plugin still has a desktop config entry")
    ponytail = plugins.get("ponytail@ponytail", {})
    if ponytail.get("enabled") is not True:
        errors.append("Ponytail is not enabled for the desktop app")
    if not list((CODEX_HOME / "plugins" / "cache" / "ponytail" / "ponytail").glob("*/skills/ponytail/SKILL.md")):
        errors.append("Ponytail desktop skill bundle is unavailable")

    hooks = json.loads((CODEX_HOME / "hooks.json").read_text(encoding="utf-8"))
    missing_hooks = REQUIRED_HOOKS - set(hooks.get("hooks", {}))
    if missing_hooks:
        errors.append(f"missing desktop lifecycle hooks: {sorted(missing_hooks)}")
    for event, action in CONTEXT_HOOK_ACTIONS.items():
        expected = f"context-mode hook codex {action}"
        commands = hook_commands(hooks, event)
        if not any(command.endswith(expected) for command in commands):
            errors.append(f"Context-Mode desktop hook is miswired: {event}")
        for command in commands:
            if not command_exists(command):
                errors.append(f"desktop hook command is unavailable: {event}: {command}")
    pretool_matchers = [
        group.get("matcher", "") for group in hooks.get("hooks", {}).get("PreToolUse", [])
    ]
    if not any("mcp__" in matcher for matcher in pretool_matchers):
        errors.append("Context-Mode PreToolUse hook does not match Codex MCP tools")
    if not any(
        command.endswith("tokless rtk-hook codex")
        for command in hook_commands(hooks, "PreToolUse")
    ):
        errors.append("RTK Codex desktop hook is missing")
    else:
        rtk_hook = next(
            command
            for command in hook_commands(hooks, "PreToolUse")
            if command.endswith("tokless rtk-hook codex")
        )
        probe = subprocess.run(
            shlex.split(rtk_hook),
            input=json.dumps({
                "session_id": "00000000-0000-4000-8000-000000000000",
                "cwd": str(ROOT),
                "tool_name": "Bash",
                "tool_input": {"command": "git status --short"},
            }),
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        try:
            updated = json.loads(probe.stdout)["hookSpecificOutput"]["updatedInput"]
        except (json.JSONDecodeError, KeyError, TypeError):
            updated = {}
        if probe.returncode or updated.get("command") != "rtk git status --short":
            errors.append("RTK Codex desktop rewrite probe failed")

    runtime_hooks: list[dict[str, object]] = []
    try:
        runtime_hooks = discovered_hooks()
    except (KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        errors.append(f"Codex desktop hook discovery failed: {exc}")
    else:
        untrusted = [
            str(hook.get("key"))
            for hook in runtime_hooks
            if hook.get("enabled") and hook.get("trustStatus") != "trusted"
        ]
        if untrusted:
            errors.append(f"Codex desktop hooks require review: {untrusted}")

    for skill in ("karpathy-guidelines", "caveman"):
        if not (CODEX_HOME / "skills" / skill / "SKILL.md").is_file():
            errors.append(f"missing desktop skill: {skill}")

    for tool in sorted(REQUIRED_FORMAT_TOOLS):
        if shutil.which(tool) is None:
            errors.append(f"missing format-routing tool: {tool}")

    instruction_limits = {
        Path.home() / "AGENTS.md": 10,
        CODEX_HOME / "AGENTS.md": 30,
        ROOT / "AGENTS.md": 50,
    }
    for path, max_lines in instruction_limits.items():
        if not path.is_file():
            errors.append(f"missing instruction file: {path}")
        elif len(path.read_text(encoding="utf-8").splitlines()) > max_lines:
            errors.append(f"instruction file exceeds {max_lines} lines: {path}")

    return {
        "enabled_plugins": sorted(
            name for name, settings in plugins.items() if settings.get("enabled") is True
        ),
        "enabled_mcp_servers": sorted(
            name for name, settings in servers.items() if settings.get("enabled") is not False
        ),
        "runtime_hooks": [
            {
                "event": hook.get("eventName"),
                "source": hook.get("source"),
                "enabled": hook.get("enabled"),
                "trust": hook.get("trustStatus"),
            }
            for hook in runtime_hooks
        ],
    }


def graph_counts_by_area(paths: set[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in paths:
        parts = path.split("/")
        area = f"{parts[0]}/{parts[1]}" if len(parts) > 2 else f"{parts[0]}/[root]"
        counts[area] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    errors: list[str] = []
    expected_roots = {"internal": ROOT, "external": EXTERNAL_ROOT}
    for link, expected_root in expected_roots.items():
        path = WORKSPACE / link
        if not path.is_symlink():
            errors.append(f"unified workspace root is not a symlink: {path}")
        elif not path.exists():
            errors.append(f"unified workspace root is unavailable: {path}")
        elif path.resolve() != expected_root.resolve():
            errors.append(
                f"unified workspace root points at {path.resolve()}, expected {expected_root}"
            )
    if not GRAPH_DB.is_file():
        errors.append(f"unified CodeGraph database is unavailable: {GRAPH_DB}")

    graph: set[str] = set()
    expected: set[str] = set()
    graph_errors: list[str] = []
    if not errors:
        try:
            graph, graph_errors = graph_files()
            expected = owned_graph_sources()
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            errors.append(str(exc))
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
        indexed_roots = {path.split("/", 1)[0] for path in graph}
        if indexed_roots != {"internal", "external"}:
            errors.append(f"unified CodeGraph roots are incomplete: {sorted(indexed_roots)}")

    desktop = audit_desktop_config(errors)
    report = {
        "host": "Codex desktop app for macOS",
        "unified_workspace": str(WORKSPACE),
        "unified_roots": {
            "internal": str((WORKSPACE / "internal").resolve()),
            "external": str((WORKSPACE / "external").resolve()),
        },
        "graph_files_by_root": {
            root: sum(path.startswith(f"{root}/") for path in graph)
            for root in ("internal", "external")
        },
        "graph_files_by_area": graph_counts_by_area(graph),
        "owned_graph_coverage_by_area": {
            area: {
                "expected": count,
                "indexed": sum(
                    graph_path in graph
                    for graph_path in expected
                    if (
                        f"{graph_path.split('/')[0]}/{graph_path.split('/')[1]}"
                        if len(graph_path.split('/')) > 2
                        else f"{graph_path.split('/')[0]}/[root]"
                    ) == area
                ),
            }
            for area, count in graph_counts_by_area(expected).items()
        },
        "owned_graph_coverage": {
            "expected": len(expected),
            "indexed": len(expected & graph),
            "missing": len(expected - graph),
        },
        "graph_total_files": len(graph),
        "graph_parser_errors": len(graph_errors),
        "inventory": inventory() if WORKSPACE.exists() else {},
        "desktop": desktop,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
