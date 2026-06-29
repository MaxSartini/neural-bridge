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

Canonical deterministic repair report:
`reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`

Current bottom line: grouped-video ranking signal survives eval-mode checkpoint rescoring; blocked-temporal matched controls and AR-only beat real, so strict forward-time temporal generalization is not proven.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Primary repair checkpoint root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`
- Eval-mode rescore root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Do not force-add ignored output roots.

## Current Numbers

- grouped real `regression_plus_binary` PR-AUC: `0.2300639382`
- grouped best matched control `ar_plus_shuffled_pca` PR-AUC: `0.2042740689`
- grouped real-minus-control delta: `+0.0257898694`
- grouped AR-only PR-AUC: `0.2246816187`
- grouped fold-seed delta: positive in `15/15`
- blocked real PR-AUC: `0.2218656156`
- blocked best matched control `ar_plus_random_pca` PR-AUC: `0.2311845051`
- blocked real-minus-control delta: `-0.0093188895`
- blocked AR-only PR-AUC: `0.2654721820`

Primary lane: `arousal_spike_rows_2_6_train_q90` using `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Task

Next work is blocked-temporal mechanism diagnosis/repair before any secondary heads. Focus on AR-only dominance, why real PCA helps grouped but hurts blocked, blocked AR+random/shuffled PCA diagnostics, temporal-negative controls, and gate/fusion instrumentation. Do not rerun the full 702 matrix unless a targeted diagnostic requires it.

## Code Discovery

Prefer MCP graph tools for code-level lookup: `search_graph`, `trace_path`, `get_code_snippet`. Use `rg` for literal text, docs, configs, and generated artifacts.
