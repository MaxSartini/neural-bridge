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

Canonical frozen-AR residual report:
`reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`

Current bottom line: grouped-video ranking signal survives eval-mode checkpoint rescoring, and the frozen-AR residual experiment strengthens the cross-video ranking claim by showing real cortical residual improves grouped beyond frozen AR and matched residual controls. Blocked strict forward-time temporal generalization is still not proven. Frozen-AR residual reduced blocked harm: old fused real was far below blocked AR, while frozen residual is within do-no-harm tolerance.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Primary repair checkpoint root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`
- Eval-mode rescore root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence_bundle_phase5_frozen_ar_residual_/`

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
- frozen-AR residual grouped frozen AR PR-AUC: `0.2246816187`
- frozen-AR residual grouped best real residual PR-AUC: `0.2383409298`
- frozen-AR residual grouped matched control PR-AUC: `0.2248361805`
- frozen-AR residual grouped delta vs frozen AR: `+0.0136593110`
- frozen-AR residual grouped delta vs matched control: `+0.0135047493`
- frozen-AR residual blocked frozen AR PR-AUC: `0.2654721820`
- frozen-AR residual blocked best real residual PR-AUC: `0.2635930904`
- frozen-AR residual blocked delta vs frozen AR: `-0.0018790916`
- frozen-AR residual blocked delta vs matched control: `-0.0017473477`
- frozen-AR residual do_no_harm_blocked_pass: yes
- frozen-AR residual full_forward_time_pass: no

Primary lane: `arousal_spike_rows_2_6_train_q90` using `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Next Task

Next work is targeted blocked residual improvement, not broad secondary heads. Candidate repairs include stronger residual alpha regularization, blocked-only inner-val delta selection, rank/lift auxiliary loss, monotonic/do-no-harm residual gating, or training residual branches only where AR confidence is low. Do not claim strict temporal generalization unless frozen residual beats AR and matched controls under blocked temporal validation. Do not rerun the full 702 matrix unless a targeted diagnostic requires it.

## Code Discovery

Prefer MCP graph tools for code-level lookup: `search_graph`, `trace_path`, `get_code_snippet`. Use `rg` for literal text, docs, configs, and generated artifacts.
