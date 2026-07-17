# Neural Bridge Agent Contract

## Authority and context

- Current repository files, executable artifacts, and the current user prompt are authoritative. New valid executable evidence outranks stale wording; never promote a failed gate without new passing evidence.
- For benchmark status, continuation, or experiment decisions, read `docs/handoff/CURRENT_STATE.md` and only the canonical artifacts it links. Do not preload historical reports for routine work.
- Prior chat state, old plans, editor state, MemPalace, and generated summaries are navigation aids only. When history is needed, recall the smallest relevant set, normally 3–5 results, then verify against current files.
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

- Use `codebase-memory-mcp` graph search and call tracing first for code discovery; use `rg`/`rg --files` for literals, configs, scripts, and non-code files or when graph results are insufficient.
- Prefix shell work with `rtk` where supported. Use `apply_patch` for edits, preserve unrelated dirty-worktree changes, and avoid destructive Git commands.
- Prefer existing scripts, cached PCA/features, and sealed compatible artifacts when provenance proves reuse is safe. Never reuse merely because names or dimensions match.
- For substantive code changes, run the smallest relevant tests first, then `npm run verify` before handoff. Refresh internal and external codebase-memory indexes only after substantive code changes.

## Handoff

- Update canonical evidence first, validate it, then update `docs/handoff/CURRENT_STATE.md` with only the current result, active constraints, exact next action, and links to detailed history.
- For routine handoff, mine only `docs/handoff/` into the `neural_bridge` MemPalace wing and verify one top-3 recall. Never remine the full project unless explicitly authorized.
- Commit validated repository changes on the active branch and finish with a clean worktree. Documentation-only work does not require codebase reindexing.
