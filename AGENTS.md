# Neural Bridge Agent Contract

## Authority

- Start clean tasks with exactly one `ctx_search`: query `current result constraints exact next action`, `source: "neural-bridge-handoff"`, `limit: 3`. Read `docs/handoff/CURRENT_STATE.md` only when that result is missing or stale. Read `docs/tokless_agent_workflow.md` only when maintaining this workflow. For benchmark decisions, read only canonical artifacts linked by current state.
- Current files and new executable evidence outrank chat, memory, and stale reports. Recall only 3–5 targeted prior results, then verify.
- Keep context compact. Never dump full logs, reports, matrices, metadata, or historical transcripts unless asked.
- Discovery output has a 20-line/3-KB cap unless explicitly requested. Search canonical references first; broad matches must be counted/aggregated off-window, then return only verdict and at most 3 exact source snippets.

## Scientific integrity

- Preserve fold, seed, target, checkpoint, PCA, label, quality-mask, and score-cache provenance. Discovery/smoke evidence cannot promote canonical claims or open outer tests.
- Keep VEATIC and AGAIN fitted data, PCA, labels, AR scores, heads, checkpoints, and videos separate unless a preregistered cross-domain experiment permits mixing. Never tune on locked AGAIN 299-video pool.
- VEATIC scientific runs use all eligible dense 2 Hz rows; apply locked black/static mask consistently; pool event PR-AUC over valid rows; retain valid negatives from zero-event videos; never invent per-video zero scores.
- `AR-only` is separately trained. `Frozen AR` is exact fold/seed AR reused unchanged by real and matched-control residual lanes. Cortical/fMRI features are video-generated predictions, not viewer neural recordings.
- Report losses and failed gates plainly. Distinguish exploratory from confirmatory evidence. Make no forecasting, mind-reading, medical, universal-emotion, or production-validity claim without proof.
- Do not alter protected caches, canonical output roots, or evidence snapshots named in current state without explicit authorization. Do not force-add ignored outputs.

## Execution

- Supported host: Codex desktop app for macOS. Route supported shell commands through Rust Token Killer (`rtk`); use `rtk proxy` only for exact raw output.
- For code work, use `$HOME/.codex/workspaces/neural-bridge-unified` with CodeGraph `explore` for cross-repo flow/blast radius. For exact source, use `search`, then `node` on current repo root or the real root resolved from unified `external`. Use `rg` only for literals/config/non-code after graph failure.
- Use `$neural-bridge-app` only for Mac app product/design/Swift work; load only relevant references.
- Prefer sealed compatible caches/features/PCA only when provenance proves safe reuse.
- Smallest relevant tests first; then `npm run verify` for substantive code changes. Keep unified, internal, and external CodeGraph indexes synced; never return raw progress streams.

## Handoff

- Update canonical evidence, validate it, then update `docs/handoff/CURRENT_STATE.md` with current result, constraints, exact next action, and detailed-history links.
- Index only current state and tokless workflow into Context-Mode; verify one top-3 result. Never index full project by default.
- Commit validated repository changes on active branch and finish clean. Documentation-only changes need no graph sync.
