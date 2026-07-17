# Neural Bridge Agent Contract

The supported host is the Codex Mac desktop app. Agents may use bundled command-line diagnostics internally, but must not require the user to launch Codex CLI or perform routine terminal setup.

## Authority and context

- Current repository files, executable artifacts, and the current user prompt are authoritative. New valid executable evidence outranks stale wording; never promote a failed gate without new passing evidence.
- At the start of a new chat, read `docs/handoff/CURRENT_STATE.md` and `docs/tokless_agent_workflow.md`, then use Context-Mode search only when a targeted prior detail is needed. Resumed chats restore their compact session snapshot automatically; clean chats start from these durable files rather than replaying the old transcript.
- For benchmark status, continuation, or experiment decisions, read `docs/handoff/CURRENT_STATE.md` and only the canonical artifacts it links. Do not preload historical reports for routine work.
- Prior chat state, old plans, editor state, Context-Mode session memory, and generated summaries are navigation aids only. When history is needed, recall the smallest relevant set, normally 3–5 results, then verify against current files.
- Keep context compact: use targeted snippets and summaries; do not dump full reports, logs, matrices, or metadata unless asked.

## Project integrity

- Preserve fold, seed, target, checkpoint, PCA, label, quality-mask, and score-cache provenance. Inner discovery, smoke runs, dry runs, and post-hoc diagnostics cannot promote canonical claims or authorize outer-test confirmation.
- Keep VEATIC and AGAIN fitted artifacts, labels, PCA transforms, heads, and videos separate unless a preregistered cross-domain experiment explicitly permits mixing. Never tune on the locked AGAIN 299-video pool.
- VEATIC scientific runs use all eligible dense 2 Hz rows, apply the locked black/static quality mask consistently, pool event PR-AUC over valid rows, retain negatives from zero-event videos, and never invent per-video zero scores.
- `AR-only` is a standalone trained baseline. `Frozen AR` is the exact fold/seed-specific AR score reused unchanged by real and matched-control residual lanes. Predicted cortical/fMRI features are video-generated upstream model outputs, not benchmark-viewer neural recordings.
- Maintain academic and commercial defensibility: report numerical losses and failed gates plainly, distinguish exploratory from confirmatory evidence, and never imply exact forecasting, mind reading, medical inference, universal emotion prediction, or production validity without executable proof.
- Neural Bridge-specific methods, heads, caches, evidence, and product surfaces are founder work. V-JEPA, TRIBE, VEATIC, and AGAIN are third-party dependencies or research inputs governed by their own terms.
- Do not modify protected dense caches, canonical output roots, or evidence snapshots identified in `docs/handoff/CURRENT_STATE.md` unless explicitly authorized. Do not force-add ignored outputs.

## Tools and implementation

- Treat CodeGraph as the code contact list, not an optional diagnostic. Before code work, query the relevant symbol, flow, callers, and blast radius; after substantive edits, sync the graph and re-check the affected path before handoff. For Neural Bridge, always target `$HOME/.codex/workspaces/neural-bridge-unified`, which spans the internal checkout and external H100/work volume. Use `rg`/`rg --files` for literals, configs, scripts, non-code files, or when CodeGraph is insufficient.
- Use `$neural-bridge-app` as the single router for Mac app product, design, Swift/SwiftUI, HIG, accessibility, Figma, testing, performance, and shipping work; load only its task-relevant expert references.
- Route supported shell discovery, implementation checks, tests, builds, diffs, status, and handoff commands through `rtk`. Use `rtk proxy` only when exact raw output is required. Use `apply_patch` for edits, preserve unrelated dirty-worktree changes, and avoid destructive Git commands.
- Prefer existing scripts, cached PCA/features, and sealed compatible artifacts when provenance proves reuse is safe. Never reuse merely because names or dimensions match.
- For substantive code changes, run the smallest relevant tests first, then `npm run verify` before handoff. Run `codegraph sync "$HOME/.codex/workspaces/neural-bridge-unified"` after substantive code changes.

## Handoff

- Update canonical evidence first, validate it, then update `docs/handoff/CURRENT_STATE.md` with only the current result, active constraints, exact next action, and links to detailed history.
- Let Context-Mode capture session and compaction state automatically. Before handoff, index `docs/handoff/CURRENT_STATE.md` and `docs/tokless_agent_workflow.md` into the project knowledge base, then verify one top-3 search result. Never index the full project by default.
- Commit validated repository changes on the active branch and finish with a clean worktree. Documentation-only work does not require codebase reindexing.
