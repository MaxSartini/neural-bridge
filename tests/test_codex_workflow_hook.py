from __future__ import annotations

from typing import Any, cast

import pytest

from neural_bridge.codex_workflow_hook import (
    GRAPHIFY_TOOL_PREFIX,
    evaluate_hook,
    shell_denial_reason,
)

REPOSITORY = "/Users/maxsartini/Neural Bridge"


def test_session_start_injects_continuous_workflow() -> None:
    result = evaluate_hook(
        {"hook_event_name": "SessionStart", "cwd": REPOSITORY, "source": "startup"}
    )

    assert result is not None
    output = cast(dict[str, Any], result["hookSpecificOutput"])
    context = cast(str, output["additionalContext"])
    assert "/Volumes/onn. Drive/Neural Bridge Artifacts" in context
    assert "one logical Graphify codebase" in context
    assert "ordinary-language job" in context
    assert "do not guess paths" in context
    assert "Do not execute a scientific action" in context
    assert "Graphify" in context
    assert "rtk" in context


def test_shell_policy_accepts_only_rtk_for_every_segment() -> None:
    assert shell_denial_reason("rtk git status", REPOSITORY) is None
    assert shell_denial_reason("rtk git status && rtk uv run pytest", REPOSITORY) is None
    assert shell_denial_reason("git status", REPOSITORY) is not None
    assert shell_denial_reason("rtk git status; pytest", REPOSITORY) is not None
    assert shell_denial_reason("rtk echo $(pwd)", REPOSITORY) is not None


def test_shell_policy_blocks_direct_repository_retrieval() -> None:
    assert shell_denial_reason("rtk rg Graphify AGENTS.md", REPOSITORY) is not None
    assert shell_denial_reason("rtk cat AGENTS.md", REPOSITORY) is not None
    assert (
        shell_denial_reason(
            "rtk cat '/Users/maxsartini/Neural Bridge/internal/handoff/CURRENT_STATE.md'",
            REPOSITORY,
        )
        is None
    )
    assert (
        shell_denial_reason("rtk cat /Users/maxsartini/.codex/config.toml", REPOSITORY) is None
    )


@pytest.mark.parametrize("tool_name", ["Bash", "exec_command", "shell", "shell_command"])
def test_pre_tool_use_enforces_current_shell_tool_names(tool_name: str) -> None:
    result = evaluate_hook(
        {
            "hook_event_name": "PreToolUse",
            "cwd": REPOSITORY,
            "tool_name": tool_name,
            "tool_input": {"cmd": "pwd"},
        }
    )

    assert result is not None
    output = cast(dict[str, Any], result["hookSpecificOutput"])
    assert output["permissionDecision"] == "deny"


def test_pre_tool_use_allows_graphify_and_denies_substitutes() -> None:
    common = {"hook_event_name": "PreToolUse", "cwd": REPOSITORY}
    assert (
        evaluate_hook(
            {
                **common,
                "tool_name": f"{GRAPHIFY_TOOL_PREFIX}query_graph",
                "tool_input": {"question": "What implements the active Neural Bridge task?"},
            }
        )
        is None
    )
    result = evaluate_hook(
        {
            **common,
            "tool_name": "mcp__filesystem__read_file",
            "tool_input": {"path": f"{REPOSITORY}/AGENTS.md"},
        }
    )
    assert result is not None
    output = cast(dict[str, Any], result["hookSpecificOutput"])
    assert output["permissionDecision"] == "deny"


def test_pre_tool_use_allows_apply_patch_and_targeted_read() -> None:
    common = {"hook_event_name": "PreToolUse", "cwd": REPOSITORY}
    assert (
        evaluate_hook(
            {
                **common,
                "tool_name": "apply_patch",
                "tool_input": {"command": "*** Begin Patch\n*** End Patch"},
            }
        )
        is None
    )
    assert (
        evaluate_hook(
            {
                **common,
                "tool_name": "Read",
                "tool_input": {"path": f"{REPOSITORY}/src/neural_bridge/artifact_graph.py"},
            }
        )
        is None
    )


def test_hook_is_inert_outside_neural_bridge() -> None:
    result = evaluate_hook(
        {
            "hook_event_name": "PreToolUse",
            "cwd": "/Users/maxsartini/other",
            "tool_name": "Bash",
            "tool_input": {"command": "pwd"},
        }
    )

    assert result is None
