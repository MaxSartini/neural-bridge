# Neural Bridge Agent Guide

## Source Rules

- Use current repo/workspace files and current user prompts as benchmark truth.
- Do not use prior chat memory, Claude/VS Code state, old plans, compacted context, or Spark-era outputs as authority.
- `codebase-memory-mcp` is for code navigation only, not benchmark authority.
- Keep context use compact. Do not dump full reports, logs, or generated metadata unless asked.

## Current Claim

Defensible claim: cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Do not claim:

- exact continuous future arousal forecasting is solved
- strict forward-time temporal generalization is proven
- Phase 5b/5c/Spark/max-capacity/deep/chimera outputs are canonical
- `holy_shit_pass` is a valid gate

Canonical adversarial review:
`docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`

Review bottom line: Phase 5 is not fraud and not grossly leaky; label permutation supports no gross leakage; grouped-video signal is real and fold-robust; honest matched-control grouped edge is about `+0.02` PR-AUC; blocked-temporal matched controls beat real.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Phase 5 main: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Phase 5 sanity: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Phase 5 runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`

Do not touch dense cache files, Phase 4 outputs, Phase 5 output roots, or evidence bundle contents unless explicitly asked.

## Current Numbers

- AR-only grouped spike PR-AUC: about `0.14725`
- Phase 3 AR+raw grouped spike PR-AUC: about `0.17030`
- Phase 4 grouped spike PR-AUC: about `0.17165`
- Phase 5 grouped spike PR-AUC: about `0.21913`
- Matched-control grouped cortical edge: about `+0.02` PR-AUC

Primary lane: `arousal_spike_rows_2_6_train_q90` using `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Task

Next work is original Phase 5 adversarial repair, starting with the primary lane above.

Repair checklist:

- restore best checkpoint before test scoring
- add matched controls for every promoted loss, especially `regression_plus_binary`
- require real > best matched blocked control
- use blocked inner validation for blocked outer protocol
- add blocked split audit
- add video-mean/static PCA diagnostic
- add within-video ranking metrics
- add top-percent product metrics
- add paired fold delta / CI versus matched controls
- remove or retire `holy_shit_pass`
- report real-minus-matched-control as headline effect

## Code Discovery

Prefer MCP graph tools for code-level lookup: `search_graph`, `trace_path`, `get_code_snippet`. Use `rg` for literal text, docs, configs, and generated artifacts.
