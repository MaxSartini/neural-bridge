"""Enforce the Neural Bridge Graphify and RTK workflow through Codex hooks."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path("/Users/maxsartini/Neural Bridge")
ARTIFACTS = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
ROOTS = (REPOSITORY, ARTIFACTS)
GRAPHIFY_TOOL = "mcp__graphify__get_exact_neural_bridge_node"
SHELL_TOOL_NAMES = frozenset(
    {"bash", "command_execution", "exec_command", "shell", "shell_command"}
)
DIRECT_RETRIEVAL_TOOLS = frozenset(
    {
        "cat",
        "fd",
        "find",
        "grep",
        "head",
        "less",
        "ls",
        "more",
        "rg",
        "sed",
        "tail",
        "tree",
    }
)
RETRIEVAL_TOOL_PATTERN = re.compile(
    r"(?:context.?mode|codegraph|filesystem|caveman|ponytail)", re.IGNORECASE
)
RETRIEVAL_VERB_PATTERN = re.compile(r"(?:^|_)(?:fetch|find|get|list|read|search)(?:_|$)")
SESSION_CONTEXT = (
    "Neural Bridge workflow enforcement is active. Treat /Users/maxsartini/Neural Bridge "
    "and /Volumes/onn. Drive/Neural Bridge Artifacts as one logical Graphify codebase. At "
    "task start, call "
    "mcp__graphify__get_exact_neural_bridge_node for "
    "/Users/maxsartini/Neural Bridge/internal/handoff/CURRENT_STATE.md as the scientific handoff; "
    "do not execute its next scientific action unless the active task authorizes it. Throughout "
    "the task, use that same Graphify tool for every repository or artifact retrieval. Supply the "
    "unique identity available--a symbol or qualified symbol, path suffix, unique filename, "
    "absolute path, or node ID--plus provenance context when names repeat, without requiring a "
    "memorized absolute path, and consume its one "
    "bounded inline content result or direct heavy-payload consumer path; do not use "
    "filesystem search, candidate dumps, pointer chasing, Context Mode, or substitute retrieval "
    "systems. Every shell command and every top-level command segment must begin with rtk."
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError:
        return False
    return True


def _is_scoped_cwd(cwd: object) -> bool:
    if not isinstance(cwd, str):
        return False
    path = Path(cwd)
    return any(_is_within(path, root) for root in ROOTS)


def _shell_segments(command: str) -> list[str] | None:
    """Split top-level shell command segments, rejecting nested command execution."""
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\" and quote != "'":
            current.append(character)
            escaped = True
        elif quote is not None:
            current.append(character)
            if character == quote:
                quote = None
        elif character in {"'", '"'}:
            current.append(character)
            quote = character
        elif character == "`" or command.startswith("$(", index):
            return None
        elif character in {"\n", ";", "|", "&"}:
            if character == "&" and current and current[-1] == ">":
                current.append(character)
            else:
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                if index + 1 < len(command) and command[index + 1] == character:
                    index += 1
        else:
            current.append(character)
        index += 1
    if quote is not None or escaped:
        return None
    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments


def _segment_tokens(segment: str) -> list[str] | None:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return None


def _direct_scope_read(tokens: list[str], cwd: Path) -> bool:
    if len(tokens) < 2:
        return False
    executable = Path(tokens[1]).name
    if executable not in DIRECT_RETRIEVAL_TOOLS:
        return False
    absolute_arguments = [Path(token) for token in tokens[2:] if token.startswith("/")]
    if absolute_arguments and all(
        not any(_is_within(argument, root) for root in ROOTS) for argument in absolute_arguments
    ):
        return False
    return any(_is_within(cwd, root) for root in ROOTS)


def shell_denial_reason(command: object, cwd: str) -> str | None:
    """Return the reason a scoped Bash call must be denied, if any."""
    if not isinstance(command, str) or not command.strip():
        return "Neural Bridge shell commands must be non-empty and begin with rtk."
    segments = _shell_segments(command)
    if segments is None:
        return "Neural Bridge shell commands cannot use nested or malformed shell execution."
    for segment in segments:
        tokens = _segment_tokens(segment)
        if not tokens or tokens[0] != "rtk":
            return "Every Neural Bridge shell command segment must begin with rtk."
        if _direct_scope_read(tokens, Path(cwd)):
            return (
                "Use mcp__graphify__get_exact_neural_bridge_node for Neural Bridge retrieval; "
                "direct filesystem search/read commands are blocked."
            )
    return None


def _references_scope(value: object) -> bool:
    serialized = json.dumps(value, sort_keys=True, default=str).lower()
    return any(str(root).lower() in serialized for root in ROOTS)


def _deny(reason: str) -> dict[str, object]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def evaluate_hook(payload: dict[str, Any]) -> dict[str, object] | None:
    """Evaluate one supported Codex hook payload."""
    if not _is_scoped_cwd(payload.get("cwd")):
        return None
    event = payload.get("hook_event_name")
    if event == "SessionStart":
        return {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": SESSION_CONTEXT,
            }
        }
    if event != "PreToolUse":
        return None

    tool_name = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    normalized_name = tool_name.lower()
    if tool_name == GRAPHIFY_TOOL:
        return None
    if normalized_name in SHELL_TOOL_NAMES or "command" in tool_input or "cmd" in tool_input:
        command = tool_input.get("command", tool_input.get("cmd"))
        reason = shell_denial_reason(command, str(payload["cwd"]))
        return _deny(reason) if reason else None

    if RETRIEVAL_TOOL_PATTERN.search(normalized_name):
        return _deny(
            "Graphify is the only retrieval layer for Neural Bridge repository and artifact data."
        )
    if (
        tool_name in {"Glob", "Grep", "Read", "Search"}
        or (
            normalized_name.startswith("mcp__")
            and RETRIEVAL_VERB_PATTERN.search(normalized_name)
            and _references_scope(tool_input)
        )
    ):
        return _deny(
            "Use mcp__graphify__get_exact_neural_bridge_node for this exact Neural Bridge lookup."
        )
    return None


def main() -> int:
    """Read one Codex hook payload from stdin and emit its supported decision."""
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise TypeError("Codex hook payload must be a JSON object")
    output = evaluate_hook(payload)
    if output is not None:
        json.dump(output, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
