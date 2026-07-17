# Tokless Agent Workflow

Updated: 2026-07-17

## Active stack

The Codex desktop app for macOS uses one Tokless-managed efficiency stack across Neural Bridge. Command-line tools below are agent-run implementation details and diagnostics; the user is never expected to launch Codex CLI.

- Tokless `0.2.6`: installer and update coordinator.
- Karpathy Guidelines: always-on principles plus the installed `karpathy-guidelines` skill.
- Caveman `1.9.1`: installed Codex skills plus terse, technically complete response guidance.
- Ponytail `4.8.4`: enabled Codex plugin, lifecycle hooks, skills, and minimal-build discipline.
- Rust Token Killer (RTK) `0.43.0`: shell-output compression. Prefix supported shell commands with `rtk`; use `rtk proxy` when exact raw output is required.
- CodeGraph `1.4.1`: local source graph and first-line code discovery.
- Context-Mode `1.0.169`: off-window processing for large files and outputs, searchable content, session capture, and compaction restoration.

The upstream repositories are retained under `$HOME/.codex/vendor/tokless-stack/` for audit, version comparison, and repair.

## Model and reasoning coverage

The workflow is global in the desktop app, not model-specific. It applies to the locally available GPT-5.6 Sol, Terra, and Luna families at every enabled reasoning level: `low`, `medium`, `high`, `xhigh`, `max`, and `ultra`.

Project authority and scientific provenance rules in `AGENTS.md` outrank generic Karpathy, Ponytail, Caveman, graph, and context-saving guidance at every model and effort level.

CodeGraph is the always-available contact list for the codebase: look up the target, callers, flow, and blast radius before editing; sync and re-query the affected path after substantive edits. RTK wraps supported shell work throughout discovery, implementation, verification, and handoff so command output is compressed before entering model context.

Discovery has a hard default response budget: 20 lines or 3 KB. Do not run broad content searches and then ask Context-Mode to return their matching sections; indexing oversized `rg` output can still produce oversized search chunks. Count or group candidates first, process raw matches inside the sandbox, and print only the verdict plus at most three exact source snippets. Use `ctx_search(limit: 3)` for indexed recall and keep CodeGraph `maxFiles` at six or fewer unless wider source is explicitly required.

## New-chat handoff

There are two continuity paths:

- Resuming a Codex thread restores Context-Mode's compact session snapshot automatically.
- A genuinely new task reads only its applicable `AGENTS.md`, then makes one `ctx_search` for `current result constraints exact next action` with `source: "neural-bridge-handoff"` and `limit: 3`. Read the handoff file only if that indexed result is missing or stale; do not read this workflow unless maintaining it.

Before ending work, update the canonical handoff when project state changed, then use the Codex Context-Mode tools:

```text
ctx_index(path: "docs/handoff/CURRENT_STATE.md", source: "neural-bridge-handoff")
ctx_index(path: "docs/tokless_agent_workflow.md", source: "neural-bridge-workflow")
ctx_search(queries: ["current result constraints exact next action"], source: "neural-bridge-handoff", limit: 3)
```

This deliberately replaces broad MemPalace recall. A clean chat gets the exact current state and targeted searchable knowledge, not an uncontrolled dump of prior conversation.

## Unified Neural Bridge graph

The logical workspace at `$HOME/.codex/workspaces/neural-bridge-unified` links:

- `internal` to the Git checkout on the internal SSD.
- `external` to the Neural Bridge artifact and H100 work volume.

Its single CodeGraph database indexes every supported source file visible across both roots. At installation it contained 493 files, 8,260 nodes, and 21,633 edges; exact counts change as source changes. Queries must target this unified workspace rather than creating independent per-checkout graphs.

`.gitignore` remains the GitHub publication boundary. `.codexignore` remains the broad raw-context boundary. Neither is a substitute for the local graph: CodeGraph may know source exists without uploading it or dumping it into a model context. Binary datasets, model weights, tensors, video, and generated results are not code nodes; inspect them only through targeted Context-Mode or existing audit scripts.

## Format coverage

The desktop workflow covers both complete Neural Bridge roots and routes formats by purpose instead of forcing every byte into the source graph:

- Every eligible, non-ignored source/config file in the internal repository and across the full external Neural Bridge root must appear in the unified CodeGraph. The audit excludes only dependency/build internals such as virtual environments, `site-packages`, `node_modules`, and generated caches; those are not Neural Bridge-owned source.
- Shell, Markdown, reStructuredText, plain text, HTML, CSS, templates, and logs use Context-Mode plus RTK-compressed search/read/log operations.
- JSON/JSONL uses `jq` and RTK's JSON filter; YAML uses `yq`; TOML/INI/XML and configuration literals use targeted Context-Mode or RTK searches.
- CSV, TSV, Parquet, and SQLite use DuckDB or SQLite so agents query and aggregate off-window instead of printing whole tables into chat.
- NumPy, MATLAB, pickle, safetensors, checkpoints, and model weights use Python metadata loaders or existing provenance-aware audit scripts. Raw arrays and weights never enter chat by default.
- Video, audio, images, PDFs, archives, and unknown binaries use `ffprobe`, `file`, PDF/image tooling, or archive manifests for metadata-first inspection.

The machine-local `npm run audit:codex-desktop` gate verifies desktop configuration, retired-workflow absence, hooks, skills, format tools, both unified roots, graph parser errors, and whole-root owned-source coverage. Its default output is compact; use `npm run audit:codex-desktop -- --verbose` only when full inventory is explicitly needed.

## Replaced components

- The `codebase-memory-mcp` Codex registration is removed. Its binary and old indexes remain only as rollback data.
- The MemPalace plugin is disabled. Context-Mode lifecycle hooks plus `docs/handoff/CURRENT_STATE.md` now provide session and durable handoff state.
- Duplicate per-checkout CodeGraph indexes are not used; the unified workspace is canonical.

## Update and verification

Tokless `0.2.6` deliberately installs only Context-Mode's `PreToolUse` hook and treats the complete upstream lifecycle hook set as a doctor failure. Neural Bridge needs the complete set. After any Tokless update, repair and verify in this order:

```bash
rtk proxy tokless update --agents codex --yes
rtk proxy context-mode upgrade
rtk summary codegraph sync "$HOME/.codex/workspaces/neural-bridge-unified"
rtk summary context-mode doctor
rtk summary codegraph status "$HOME/.codex/workspaces/neural-bridge-unified"
rtk gain
rtk npm run audit:codex-desktop
```

Expected configuration:

- `features.hooks = true` and `features.plugin_hooks = true`.
- `mcp_servers.context-mode` and `mcp_servers.codegraph` enabled; Context-Mode MCP pins `CONTEXT_MODE_PLATFORM=codex` so tools and hooks share `~/.codex/context-mode` storage.
- `~/.codex/settings.json` grants Context-Mode `Read($NEURAL_BRIDGE_EXTERNAL_ROOT/**)` (stored with the resolved machine-local root) so every external Neural Bridge file is available without opening unrelated paths.
- CodeGraph MCP pinned with `--path "$HOME/.codex/workspaces/neural-bridge-unified"`; its `internal` and `external` symlinks resolve to both complete Neural Bridge roots.
- Context-Mode `PreToolUse`, `PostToolUse`, `SessionStart`, `PreCompact`, `UserPromptSubmit`, and `Stop` hooks present.
- Ponytail installed and enabled; Caveman and Karpathy skill directories present.
- MemPalace disabled and no `codebase-memory-mcp` MCP registration.

The pre-migration Codex files are backed up under `$HOME/.codex/backups/`. After installation or configuration changes, fully quit and reopen the Codex desktop app for macOS so its MCP servers, skills, plugins, and hooks reload. No Codex CLI session is part of the user workflow.
