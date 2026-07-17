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


def running_commands() -> set[str]:
    result = subprocess.run(
        ["/bin/ps", "-axo", "comm="],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def detect_desktop_app() -> Path:
    override = os.environ.get("CODEX_DESKTOP_APP")
    if override:
        return Path(override).expanduser().resolve()
    commands = running_commands()
    for name in ("ChatGPT", "Codex"):
        candidate = Path(f"/Applications/{name}.app")
        if str(candidate / "Contents" / "MacOS" / name) in commands:
            return candidate
    for name in ("ChatGPT", "Codex"):
        candidate = Path(f"/Applications/{name}.app")
        if candidate.is_dir():
            return candidate
    return Path("/Applications/ChatGPT.app")


ROOT = Path(__file__).resolve().parents[1]
CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
DESKTOP_APP = detect_desktop_app()
DESKTOP_EXECUTABLE = DESKTOP_APP / "Contents" / "MacOS" / DESKTOP_APP.stem
DESKTOP_CODEX = DESKTOP_APP / "Contents" / "Resources" / "codex"
WORKSPACE = CODEX_HOME / "workspaces" / "neural-bridge-unified"
GRAPH_DB = WORKSPACE / ".codegraph" / "codegraph.db"
EXTERNAL_ROOT = Path(
    os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", WORKSPACE / "external")
).resolve()
HOOK_GUARD = CODEX_HOME / "tools" / "context_mode_hook_guard.py"

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
EXPECTED_ENABLED_PLUGINS = {
    "browser@openai-bundled",
    "chrome@openai-bundled",
    "computer-use@openai-bundled",
    "ponytail@ponytail",
}
EXPECTED_ENABLED_MCP_SERVERS = {"codegraph", "context-mode", "node_repl"}
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


def real_root_graph(project_root: Path) -> dict[str, object]:
    """Validate one real repository index without relying on a named smoke file."""
    database = project_root / ".codegraph" / "codegraph.db"
    connection = sqlite3.connect(database)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"CodeGraph database integrity check failed: {database}")
        rows = list(
            connection.execute(
                "SELECT path, node_count, errors FROM files ORDER BY node_count DESC, path"
            )
        )
    finally:
        connection.close()

    paths = {path for path, _, _ in rows}
    missing = sorted(path for path in paths if not (project_root / path).is_file())
    parser_errors = [path for path, _, detail in rows if detail not in (None, "[]")]
    areas = Counter(path.split("/", 1)[0] for path in paths)
    probe_path = next(
        (path for path, nodes, _ in rows if nodes and (project_root / path).is_file()),
        None,
    )
    source_readable = False
    if probe_path:
        probe = subprocess.run(
            [
                "codegraph", "node", "--path", str(project_root), "--file", probe_path,
                "--offset", "1", "--limit", "1",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        output = f"{probe.stdout}\n{probe.stderr}".lower()
        source_readable = (
            probe.returncode == 0
            and bool(probe.stdout.strip())
            and "could not read from disk" not in output
        )
    return {
        "paths": paths,
        "files": len(paths),
        "areas": dict(sorted(areas.items())),
        "missing_on_disk": missing,
        "parser_errors": parser_errors,
        "source_readable": source_readable,
    }


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
        [str(DESKTOP_CODEX), "app-server", "--stdio"],
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
    if not DESKTOP_APP.is_dir() or not os.access(DESKTOP_CODEX, os.X_OK):
        errors.append(f"Codex desktop runtime is unavailable: {DESKTOP_APP}")
    if str(DESKTOP_EXECUTABLE) not in running_commands():
        errors.append(f"Codex desktop app is not running: {DESKTOP_EXECUTABLE}")
    config = tomllib.loads((CODEX_HOME / "config.toml").read_text(encoding="utf-8"))
    features = config.get("features", {})
    if not features.get("hooks") or not features.get("plugin_hooks"):
        errors.append("Codex desktop hook feature flags are not both enabled")

    servers = config.get("mcp_servers", {})
    for required in EXPECTED_ENABLED_MCP_SERVERS:
        server = servers.get(required)
        if not server or server.get("enabled") is False:
            errors.append(f"Codex desktop MCP is missing or disabled: {required}")
        elif not command_exists(server.get("command", "")):
            errors.append(f"Codex desktop MCP command is unavailable: {required}")
    if servers.get("context-mode", {}).get("env", {}).get("CONTEXT_MODE_PLATFORM") != "codex":
        errors.append("Context-Mode desktop MCP is not pinned to Codex storage")

    codex_settings_path = CODEX_HOME / "settings.json"
    try:
        codex_settings = json.loads(codex_settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        codex_settings = {}
    external_read_rule = f"Read({EXTERNAL_ROOT}/**)"
    if external_read_rule not in codex_settings.get("permissions", {}).get("allow", []):
        errors.append("Context-Mode cannot read the complete external Neural Bridge root")
    if servers.get("codegraph", {}).get("args") != [
        "serve", "--mcp", "--path", str(WORKSPACE)
    ]:
        errors.append("CodeGraph desktop MCP is not pinned to unified workspace")
    if servers.get("codegraph", {}).get("env", {}).get("CODEGRAPH_MCP_TOOLS") != "explore,search,node":
        errors.append("CodeGraph exact search/source tools are not enabled")
    enabled_servers = {
        name
        for name, settings in servers.items()
        if settings.get("enabled") is not False
    }
    if enabled_servers != EXPECTED_ENABLED_MCP_SERVERS:
        errors.append(
            "enabled MCP servers differ from intentional coding set: "
            f"{sorted(enabled_servers)}"
        )
    if "codebase-memory-mcp" in servers:
        errors.append("retired codebase-memory MCP is still registered")

    plugins = config.get("plugins", {})
    if any(name.startswith("mempalace@") for name in plugins):
        errors.append("retired MemPalace plugin still has a desktop config entry")
    ponytail = plugins.get("ponytail@ponytail", {})
    if ponytail.get("enabled") is not True:
        errors.append("Ponytail is not enabled for the desktop app")
    enabled_plugins = {
        name
        for name, settings in plugins.items()
        if settings.get("enabled") is True
    }
    if enabled_plugins != EXPECTED_ENABLED_PLUGINS:
        errors.append(
            "enabled desktop plugins differ from intentional set: "
            f"{sorted(enabled_plugins)}"
        )
    plugin_probe = subprocess.run(
        [str(DESKTOP_CODEX), "plugin", "list", "--json"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    try:
        runtime_plugins = {
            item["pluginId"]
            for item in json.loads(plugin_probe.stdout)["installed"]
            if item.get("enabled")
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        runtime_plugins = set()
    if plugin_probe.returncode or runtime_plugins != EXPECTED_ENABLED_PLUGINS:
        errors.append(
            "Codex runtime plugin set differs from intentional set: "
            f"{sorted(runtime_plugins)}"
        )
    if not list((CODEX_HOME / "plugins" / "cache" / "ponytail" / "ponytail").glob("*/skills/ponytail/SKILL.md")):
        errors.append("Ponytail desktop skill bundle is unavailable")
    else:
        ponytail_skill = max(
            (CODEX_HOME / "plugins" / "cache" / "ponytail" / "ponytail").glob("*/skills/ponytail/SKILL.md")
        ).read_text(encoding="utf-8")
        for required_text in (
            "Maximize the requested result and quality",
            "Reuse only after checking semantic, provenance",
            "Code without proportionate verification is unfinished",
        ):
            if required_text not in ponytail_skill:
                errors.append(f"Ponytail outcome guard is missing: {required_text}")

    hooks = json.loads((CODEX_HOME / "hooks.json").read_text(encoding="utf-8"))
    missing_hooks = REQUIRED_HOOKS - set(hooks.get("hooks", {}))
    if missing_hooks:
        errors.append(f"missing desktop lifecycle hooks: {sorted(missing_hooks)}")
    for event, action in CONTEXT_HOOK_ACTIONS.items():
        expected = f"context-mode hook codex {action}"
        commands = hook_commands(hooks, event)
        correctly_wired = any(command.endswith(expected) for command in commands)
        if event == "SessionStart":
            correctly_wired = any(
                command == f"/usr/bin/python3 {HOOK_GUARD} sessionstart"
                for command in commands
            )
        if not correctly_wired:
            errors.append(f"Context-Mode desktop hook is miswired: {event}")
        for command in commands:
            if not command_exists(command):
                errors.append(f"desktop hook command is unavailable: {event}: {command}")
    guard_probe = subprocess.run(
        ["/usr/bin/python3", str(HOOK_GUARD), "--self-test"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    if guard_probe.returncode or not guard_probe.stdout.startswith("PASS "):
        errors.append("Context-Mode SessionStart injection guard failed")
    fresh_guard_probe = subprocess.run(
        ["/usr/bin/python3", str(HOOK_GUARD), "sessionstart"],
        input=json.dumps({"source": "clear"}),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    try:
        fresh_context = json.loads(fresh_guard_probe.stdout)[
            "hookSpecificOutput"
        ]["additionalContext"]
    except (json.JSONDecodeError, KeyError, TypeError):
        fresh_context = ""
    if (
        fresh_guard_probe.returncode
        or not fresh_context
        or len(fresh_guard_probe.stdout.encode()) > 2_048
        or len(fresh_context.encode()) > 1_024
    ):
        errors.append("Context-Mode fresh-task injection exceeds compact byte budget")

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
        runtime_session_start = {
            hook.get("command")
            for hook in runtime_hooks
            if hook.get("eventName") == "sessionStart" and hook.get("enabled")
        }
        if f"/usr/bin/python3 {HOOK_GUARD} sessionstart" not in runtime_session_start:
            errors.append("Codex runtime is not using the guarded SessionStart hook")
        if not any(
            "ponytail-activate.js" in str(command)
            for command in runtime_session_start
        ):
            errors.append("Codex runtime Ponytail SessionStart hook is missing")

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
            continue
        instruction_text = path.read_text(encoding="utf-8")
        if len(instruction_text.splitlines()) > max_lines:
            errors.append(f"instruction file exceeds {max_lines} lines: {path}")
        if path in {CODEX_HOME / "AGENTS.md", ROOT / "AGENTS.md"} and "20-line" not in instruction_text:
            errors.append(f"instruction file lacks discovery output budget: {path}")
        foreign_hosts = {"Claude", "Gemini", "Cursor", "Copilot", "OpenCode", "Qwen", "Kimi"}
        if foreign_hosts.intersection(instruction_text.split()):
            errors.append(f"Codex instruction file references another agent host: {path}")

    return {
        "enabled_plugins": sorted(
            name for name, settings in plugins.items() if settings.get("enabled") is True
        ),
        "enabled_mcp_servers": sorted(
            name for name, settings in servers.items() if settings.get("enabled") is not False
        ),
        "external_context_read_rule": external_read_rule,
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

    real_graphs: dict[str, dict[str, object]] = {}
    for root_name, project_root in expected_roots.items():
        database = project_root / ".codegraph" / "codegraph.db"
        if not database.is_file():
            errors.append(f"real-root CodeGraph index is unavailable: {project_root}")
            continue
        try:
            real_graphs[root_name] = real_root_graph(project_root)
        except (RuntimeError, sqlite3.DatabaseError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
            continue
        summary = real_graphs[root_name]
        if summary["missing_on_disk"]:
            errors.append(f"real-root CodeGraph has missing files: {project_root}")
        if summary["parser_errors"]:
            errors.append(f"real-root CodeGraph has parser errors: {project_root}")
        if not summary["source_readable"]:
            errors.append(f"real-root CodeGraph cannot serve exact source: {project_root}")
        real_paths = {f"{root_name}/{path}" for path in summary["paths"]}
        unified_paths = {path for path in graph if path.startswith(f"{root_name}/")}
        if real_paths != unified_paths:
            errors.append(f"unified and real-root CodeGraph indexes differ: {root_name}")

    desktop = audit_desktop_config(errors)
    report = {
        "runtime": {
            "app_bundle": str(DESKTOP_APP),
            "app_executable": str(DESKTOP_EXECUTABLE),
            "codex_binary": str(DESKTOP_CODEX),
            "project_root": str(ROOT),
        },
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
        "real_root_graphs": {
            name: {key: value for key, value in summary.items() if key != "paths"}
            for name, summary in real_graphs.items()
        },
        "inventory": inventory() if WORKSPACE.exists() else {},
        "desktop": desktop,
        "errors": errors,
    }
    if "--verbose" in sys.argv[1:]:
        output = report
    else:
        runtime_hooks = desktop["runtime_hooks"]
        output = {
            "runtime": report["runtime"],
            "unified_roots": report["unified_roots"],
            "graph": {
                **report["owned_graph_coverage"],
                "files_by_root": report["graph_files_by_root"],
                "real_root_indexes": {
                    name: {
                        "files": summary["files"],
                        "areas": len(summary["areas"]),
                        "source_readable": summary["source_readable"],
                    }
                    for name, summary in real_graphs.items()
                },
                "parser_errors": report["graph_parser_errors"],
            },
            "desktop": {
                "enabled_plugins": desktop["enabled_plugins"],
                "enabled_mcp_servers": desktop["enabled_mcp_servers"],
                "runtime_hooks": len(runtime_hooks),
                "trusted_runtime_hooks": sum(
                    hook["enabled"] and hook["trust"] == "trusted"
                    for hook in runtime_hooks
                ),
                "external_context_read_rule": desktop["external_context_read_rule"],
            },
            "errors": errors,
        }
    verbose = "--verbose" in sys.argv[1:]
    print(
        json.dumps(
            output,
            indent=2 if verbose else None,
            separators=None if verbose else (",", ":"),
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
