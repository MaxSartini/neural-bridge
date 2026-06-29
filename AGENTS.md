# Neural Bridge Agent Guide

This repo is Neural Bridge.

## Source Of Truth

Use only this repository, the configured local workspace, and current user prompts as benchmark truth. Do not use prior Codex chat memory, Claude/Anthropic state, VS Code chat state, old compacted context, previous agent plans, or post-original-Phase-5 Spark outputs as authority.

`codebase-memory-mcp` may be used for code navigation, imports, symbol lookup, and file location. Do not use MCP memory as previous-task or benchmark authority.

## Current Canonical State

The repo is reset to the point immediately after original Phase 5 and Claude/adversarial review.

Phase 4 completed the dense AGAIN 2Hz fold-safe PCA bridge. The best Phase 4 primary spike lane was:

- Target: `arousal_spike_rows_2_6_train_q90`
- Feature: `temporal_mean_2s_then_pca256`
- Lane: `AR_plus_PCA_plus_temporal_diagnostics`
- Grouped-video PR-AUC: about `0.17165`
- AR-only grouped-video PR-AUC: about `0.14725`
- Phase 3 AR+raw grouped-video PR-AUC: about `0.17030`

Original Phase 5 completed the learned-head pass over the canonical Phase 4 feature. The canonical roots are:

- Main run: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Label-permutation sanity run: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`

Original Phase 5 primary setup:

- Target: `arousal_spike_rows_2_6_train_q90`
- Continuous training source: `future_arousal_max_delta_rows_2_6`
- Feature: `temporal_mean_2s_then_pca256`
- Input: AR + `temporal_mean_2s_then_pca256` + temporal diagnostics
- Best learned head: `gated_ar_pca_mlp`
- Best loss: `regression_plus_binary`
- Best grouped-video PR-AUC: about `0.21913`

This was a large grouped-video improvement over AR-only `0.14725` and Phase 4 `0.17165`. The original Phase 5 label-permutation sanity collapsed near chance/prevalence and supports no gross leakage.

## Canonical Adversarial Review

Canonical review artifact: `docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`.

The original Phase 5 result is not fraud and not grossly leaky. Label permutation sanity supports no gross leakage, and the grouped-video cross-video spike/event ranking signal is real and fold-robust.

The old headline was overpromoted. The honest matched-control grouped cortical edge is closer to about `+0.02` PR-AUC, not the larger deltas versus AR-only or Phase 4. The defensible current claim is cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Do not claim exact continuous future arousal forecasting is solved. Do not claim strict full forward-time temporal mechanism is solved. Strict forward-time temporal generalization remains under repair because blocked-temporal matched controls were not properly required to beat real.

Blocked-temporal matched controls beat real, so strict forward-time temporal generalization is not yet proven. The old Phase 5 promotion gate was structurally flawed because `blocked_temporal_support` checked real > AR, not real > best matched blocked control.

Continuous arousal movement scoring remains promising and should be evaluated with ranking/lift metrics, not only MAE/MSE. The old Phase 5 promotion gate was too generous and grouped-biased.

## Non-Canonical Work

Phase 5a, Phase 5b, Phase 5c, max-capacity, deep, strict-longtrain, chimera, Spark-generated exploratory outputs, and any `holy_shit_pass`-style gates are non-canonical. Do not use them as evidence, do not cite them in docs, and do not resurrect their runners or reports.

The next task is original Phase 5 adversarial repair, not Phase 5b/5c expansion.

## Next Repair Tasks

- Restore the best checkpoint before test scoring.
- Add matched controls for every promoted loss, especially `regression_plus_binary`.
- Correct the blocked gate so real must beat the best matched blocked control, not merely AR.
- Use blocked inner validation for blocked outer protocol.
- Add blocked split audit.
- Add video-mean/static PCA diagnostic.
- Add within-video ranking metrics.
- Add top-percent product-facing metrics.
- Add paired fold delta / CI versus matched controls.
- Remove or retire `holy_shit_pass`.
- Report real-minus-matched-control as the headline effect.
- Re-run or rescore the original winning lane first: `gated_ar_pca_mlp` / `regression_plus_binary` / `temporal_mean_2s_then_pca256` / AR + temporal diagnostics.

## Canonical Artifacts To Preserve

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 external root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Original Phase 5 main root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423/`
- Original Phase 5 sanity root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_185338/`
- Original Phase 5 runner: `backend/scripts/run_again_dense_2hz_phase5_learned_heads.py`
- Evidence bundle: `evidence_bundle_phase0_to_phase5_20260625/`
- Canonical adversarial review: `docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly only referencing them from docs.

## Current Claim Framing

Current proven wedge: cross-video future arousal spike / emotional moment ranking from video-derived cortical bridge features.

Not yet proven: exact continuous future arousal forecasting or strict full forward-time temporal mechanism.

Commercial framing: Neural Bridge is a translation layer that turns noisy video/cortical representations into ranked human-response intelligence. The bridge/benchmark/control protocol is the moat, not V-JEPA/TRIBE themselves.

## Start Here

Read these files before making project-state claims:

1. `README.md`
2. `docs/current_project_state.md`
3. `docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`
4. `docs/veatic_v2_evidence_summary.md`
5. `REQUIREMENTS.md`
6. `ROADMAP.md`
7. `docs/external_assets_manifest.md`
8. `docs/veatic_v2_evidence_freeze.md`

## Default Commands

```bash
npm run audit:repo
npm run evidence:verify
python3 backend/scripts/run_veatic_strict_benchmark.py --modality-audit-only
python3 backend/scripts/run_veatic_strict_benchmark.py --dry-run --primary-only
python3 -m pytest -q tests/test_veatic_raw_representation_contract.py tests/test_veatic_strict_benchmark_contract.py tests/test_grouped_video_split.py
python3 -m pytest -q tests/test_mlx_vjepa21_cortical.py tests/test_veatic_tribe_cache_runtime.py tests/test_veatic_frozen_tensor_adapter.py tests/test_veatic_frozen_tensor_trained_heads.py
```

Use `npm run verify` before pushing when dependencies are available.

## Code Discovery

Prefer the codebase-memory MCP graph for code structure. This repo requires MCP-first discovery on turns that need code-level lookup:

1. `search_graph`
2. `trace_path`
3. `get_code_snippet`
4. literal search only when MCP returns insufficient signal

Default project selection:

- Local source project: `Users-maxsartini-Neural-Bridge` (`/Users/maxsartini/Neural Bridge`)
- External artifact project: `Volumes-onn.-Drive-Neural-Bridge` (`/Volumes/onn. Drive/Neural Bridge`)

Use local source first for source edits and code decisions. Use the external artifact project only for evidence bundles, large outputs, or artifacts, and include explicit file scope when possible.

Use `rg` for literal strings, docs, configs, and generated reports when appropriate. Keep discovery output compact.

## Guardrails

- Keep train-only thresholds, train-only transforms, grouped-video validation, blocked validation, and matched controls intact.
- Do not tune decision thresholds on filtered test subsets.
- Do not report positive-only event/pre-event masks as PR-AUC discrimination tests.
- Do not describe a cache as multimodal unless modality flags or audit output show text, audio, and video present.
- Do not commit heavyweight data, model weights, raw media, local caches, external tensor payloads, or machine-specific paths.
- Keep machine-specific paths in local `.env`; `.env.example` must stay portable.
- Treat `benchmarks/` and `outputs/` as retained evidence artifacts. Use current docs before generated metadata.
