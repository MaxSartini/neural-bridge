# Compact Agent Workflow

Updated: 2026-07-20

This is a machine-local efficiency contract for Codex desktop work on Neural Bridge. It is not a project dependency and never requires a contributor to install Codex CLI, Rust Token Killer, Context-Mode, or CodeGraph.

## Discovery

- Start a clean task with one Context-Mode search for `current result constraints exact next action`, scoped to `neural-bridge-canonical-handoff`, limit `3`.
- Read [`handoff/CURRENT_STATE.md`](handoff/CURRENT_STATE.md) only when that result is absent or stale.
- Use the canonical repository CodeGraph for source flow, blast radius, exact search, and source lookup. The ignored `artifacts` symlink is outside the graph; retired roots are queried only for explicit forensic migration work.
- Start artifact lookup with one Context-Mode search scoped to `neural-bridge-canonical-artifacts`; open the exact registry entry or mapped artifact only when needed.
- Use repository search only for literals, configuration, non-code material, or after graph lookup fails.
- Keep discovery output under 20 lines or 3 KB. Aggregate large results off-window and return at most three exact snippets.

## Execution

- Prefix supported agent-run shell commands with `rtk`; use `rtk proxy` only when exact raw output is required.
- Current files and executable evidence outrank chat history and stale reports.
- Preserve unrelated worktree changes and use surgical patches.
- Run the smallest relevant checks first, then the full supported verification surface for substantive code changes.
- Project-facing commands remain standard `uv`, Python, and pytest commands.

## Scientific work

- Follow the provenance, split, fold, seed, target, control, and locked-pool rules in `AGENTS.md` and the canonical handoff.
- Never promote discovery or smoke evidence into a concluded claim.
- Keep VEATIC, AGAIN, and VEATIC 2.1 fitted artifacts separate unless a prospectively declared experiment explicitly permits transfer.

## Handoff

When project state changes:

1. Validate the canonical evidence.
2. Update [`handoff/CURRENT_STATE.md`](handoff/CURRENT_STATE.md) with the result, constraints, exact next action, and detailed-history links.
3. Index only that handoff as `neural-bridge-canonical-handoff` and this file as `neural-bridge-canonical-workflow`.
4. Verify that the current-state query returns the updated handoff in its top three results.

Do not index the full repository into Context-Mode by default. Its compact artifact registry is indexed as `neural-bridge-canonical-artifacts`; heavy bytes remain outside Git behind the ignored `artifacts` symlink.
