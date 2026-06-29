# Neural Bridge

Neural Bridge is a local research stack for turning video-derived cortical representations into ranked human-response intelligence under strict controls.

## Current Canonical State

The repository is reset to the point immediately after original Phase 5 and Claude/adversarial review. Phase 5a/5b/5c, max-capacity, deep, chimera, strict-longtrain, Spark-generated exploratory outputs, and `holy_shit_pass`-style gates are non-canonical and should not be used as evidence.

Canonical benchmark state through original Phase 5:

- AR-only grouped spike PR-AUC: about `0.14725`
- Phase 3 AR+raw grouped spike PR-AUC: about `0.17030`
- Phase 4 best grouped spike PR-AUC: about `0.17165`
- Original Phase 5 best grouped spike PR-AUC: about `0.21913`

Phase 4 completed a dense AGAIN 2Hz fold-safe PCA bridge. Its best primary spike lane used target `arousal_spike_rows_2_6_train_q90`, feature `temporal_mean_2s_then_pca256`, and lane `AR_plus_PCA_plus_temporal_diagnostics`.

Original Phase 5 completed a learned-head pass at:

- Main root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Label-permutation sanity root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`

The winning original Phase 5 lane was `gated_ar_pca_mlp` with `regression_plus_binary`, using AR + `temporal_mean_2s_then_pca256` + temporal diagnostics for `arousal_spike_rows_2_6_train_q90`. The continuous training source was `future_arousal_max_delta_rows_2_6`.

## Defensible Claim

The current proven wedge is cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Neural Bridge should not currently claim that exact continuous future arousal forecasting is solved. Strict forward-time temporal generalization remains under repair because blocked-temporal matched controls were not properly required to beat real in the old promotion logic.

Commercially, Neural Bridge is a translation layer that turns noisy video/cortical representations into ranked human-response intelligence. The bridge/benchmark/control protocol is the moat, not V-JEPA/TRIBE themselves.

## Adversarial Review Caveats

Claude/adversarial review found that the original Phase 5 result is not fake and not gross leakage, but the headline was overpromoted. The old Phase 5 promotion gate was too generous and grouped-biased.

Continuous arousal movement scoring remains promising, but should be evaluated with ranking/lift metrics rather than only MAE/MSE.

## Next Planned Work

The next task is original Phase 5 adversarial repair, not Phase 5b/5c expansion. Repair should first re-run or rescore the original winning lane:

`gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

Required repair items include restoring the best checkpoint before test scoring, using matched controls, fixing blocked support logic, using blocked inner validation for blocked outer protocol, adding blocked split audit, adding video-mean/static PCA diagnostics, and adding within-video plus top-percent metrics.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 external root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Original Phase 5 main root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Original Phase 5 sanity root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Original Phase 5 runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless a task explicitly requires documentation-only references.

## Source Of Truth

Use local files and current prompts as benchmark truth. Do not use prior Codex chat memory, Claude state, VS Code state, previous agent plans, old compacted context, or post-original-Phase-5 Spark outputs as authority.

Use `codebase-memory-mcp` for code navigation and source lookup only, not as benchmark authority.

See [AGENTS.md](AGENTS.md) and [docs/current_project_state.md](docs/current_project_state.md) before making project-state claims.
