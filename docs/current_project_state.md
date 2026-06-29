# Current Project State

Last updated: 2026-06-29

## Source Of Truth

This document describes the canonical local state immediately after original Phase 5 and Claude/adversarial review. Use only current local repository/workspace files and current user prompts as benchmark truth. Do not use prior Codex chat memory, Claude/Anthropic state, VS Code chat state, previous agent plans, old compacted context, or post-original-Phase-5 Spark outputs as authority.

`codebase-memory-mcp` is allowed for code navigation, imports, symbol lookup, and file location. It is not benchmark authority.

## Canonical Artifact State

Preserve these artifacts:

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 external root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Original Phase 5 main root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Original Phase 5 sanity root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Original Phase 5 runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Claude/adversarial review artifact, if present.

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly only referencing them from docs.

## Phase 4 Summary

Phase 4 completed the dense AGAIN 2Hz fold-safe PCA bridge. The best primary spike lane was:

- Target: `arousal_spike_rows_2_6_train_q90`
- Feature: `temporal_mean_2s_then_pca256`
- Lane: `AR_plus_PCA_plus_temporal_diagnostics`
- Grouped-video PR-AUC: about `0.17165`
- AR-only grouped-video PR-AUC: about `0.14725`
- Phase 3 AR+raw grouped-video PR-AUC: about `0.17030`

Phase 4 is the canonical PCA bridge reference for the original Phase 5 learned-head result.

## Original Phase 5 Summary

Original Phase 5 completed at:

- Main root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Label-permutation sanity root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`

Primary setup:

- Target: `arousal_spike_rows_2_6_train_q90`
- Continuous training source: `future_arousal_max_delta_rows_2_6`
- Feature: `temporal_mean_2s_then_pca256`
- Input: AR + `temporal_mean_2s_then_pca256` + temporal diagnostics
- Best learned head: `gated_ar_pca_mlp`
- Best loss: `regression_plus_binary`
- Best grouped-video PR-AUC: about `0.21913`

This was a large improvement over AR-only `0.14725` and Phase 4 `0.17165`. The original Phase 5 label-permutation sanity showed collapse near chance/prevalence and supported no gross leakage.

## Claude/Adversarial Critique

The original Phase 5 result is not fake and not gross leakage. The old headline was overpromoted.

The defensible current claim is cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Strict forward-time temporal generalization remains under repair because blocked-temporal matched controls were not properly required to beat real. Exact continuous future arousal forecasting is not proven as the main claim.

Continuous arousal movement scoring remains promising and should be evaluated with ranking/lift metrics, not only MAE/MSE. The old Phase 5 promotion gate was too generous and grouped-biased.

## Quarantine And Non-Canonical Outputs

The post-original-Phase-5 Spark-era quarantine was reviewed and deleted after confirming canonical Phase 4, dense cache, original Phase 5 roots, original runner, and evidence bundle still existed.

Ignore Phase 5a, Phase 5b, Phase 5c, max-capacity, deep, strict-longtrain, strict-recheck, chimera, Spark-generated exploratory outputs, and `holy_shit_pass`-style gates. They are non-canonical and should not be cited, resurrected, or used to steer future work.

## Exact Next Repair Tasks

The next task is original Phase 5 adversarial repair, not Phase 5b/5c expansion. Start with the original winning lane:

`gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

Required repair items:

- Restore the best checkpoint before test scoring.
- Fix control labels and make controls explicitly matched.
- Ensure `regression_plus_binary` has matched `regression_plus_binary` controls.
- Correct blocked support logic: real must beat the best matched blocked control, not merely AR.
- Use blocked inner validation for blocked outer protocol.
- Add blocked split audit.
- Add video-mean/static PCA diagnostic.
- Add within-video and top-percent metrics.
- Re-run or rescore the original winning lane first before expanding scope.

## What Not To Do Next

- Do not train or benchmark unless the task explicitly asks for the adversarial repair run.
- Do not write Phase 5b/5c/max-capacity/deep/chimera runners.
- Do not cite Spark weird outputs as evidence.
- Do not claim continuous arousal forecasting is solved.
- Do not claim strict blocked-temporal matched controls are solved.
- Do not use `holy_shit_pass` as a valid gate.
- Do not touch dense cache files, Phase 4 outputs, original Phase 5 roots, or evidence bundle contents.

## Claim Framing

Current proven wedge: cross-video future arousal spike / emotional moment ranking.

Not yet proven: exact continuous future arousal forecasting or strict full forward-time temporal mechanism.

Commercial framing: Neural Bridge is a translation layer that turns noisy video/cortical representations into ranked human-response intelligence. The bridge/benchmark/control protocol is the moat, not V-JEPA/TRIBE themselves.
