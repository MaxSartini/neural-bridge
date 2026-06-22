# Neural Bridge Agent Guide

This repo is Neural Bridge. Treat older cloned-project context as archaeology, not active project state.

## Current Truth

- The proven baseline is VEATIC-124 v2 using video-dominant TRIBE-predicted cortical response features.
- The headline claim is arousal event/spike ranking under strict controls, not exact continuous arousal-value forecasting.
- The current cache is not a proven full text+audio+video multimodal cache.
- The strict modality audit reports `122/124` video-only cache entries and `2/124` text+audio+video entries.
- Only VEATIC videos `83` and `84` contain audio streams, so do not propose re-encoding all 124 videos for multimodal coverage.
- The guarded `83,84` multimodal pilot reaches audio extraction, word extraction, Text/Sentence creation, and text feature preparation.
- That pilot is blocked until `meta-llama/Llama-3.2-3B` is available locally or authorized through Hugging Face.
- The raw representation audit is complete. Keep frozen `cortical_pca64_delta` as the v2 baseline.
- The trained-head benchmark layer is implemented. It uses frozen `pca_sequence_128_causal_past_2s_mean` tensors, fresh same-row AR, fresh controls, grouped gates, and MPS-only training.
- `AR_plus_PCA128` and `residualized_AR_plus_PCA128` passed grouped spike incremental gates in the completed trained-head run; `PCA128_only` did not stably beat AR.
- `roi_parcel_features` is an important side candidate; `topk_vertices_512` is supervised/cautionary.
- Model-ready tensors are frozen under `NEURAL_BRIDGE_EXTERNAL_ROOT/tensors/veatic_124_raw_representation_v1/` with lightweight summaries in `outputs/veatic_124_raw_representation_tensor_export_v1/`.
- V-JEPA 2.1 MLX support is implemented and selected from converted weights with `tensor_layout=vjepa2_1_mlx_port`.
- AGAIN boundary/scout/full-AR/sparse-teacher tooling is implemented. The 50-video / 480-window sparse teacher pilot failed hybrid sparse PCA128 promotion gates. A cache-only smaller-width follow-up looked promising at 500-window scale. The later 2000-budget confirmatory run completed 1,948 windows on the same 50-video selector subset and did not confirm promotion: locked PCA32 beat AR but lost to raw sparse and matched-random controls; train-selected small PCA failed nuisance and matched-random controls. Do not scale AGAIN sparse teacher from this result.

## Start Here

Read these files before making claims about project state:

1. `README.md`
2. `docs/current_project_state.md`
3. `docs/veatic_v2_evidence_summary.md`
4. `REQUIREMENTS.md`
5. `ROADMAP.md`
6. `docs/external_assets_manifest.md`
7. `docs/veatic_v2_evidence_freeze.md`
8. `docs/superseded_artifacts.md`

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

## Guardrails

- Keep train-only thresholds, train-only transforms, blocked validation, grouped-video validation, and shuffled/random/time controls intact.
- Do not tune decision thresholds on filtered test subsets.
- Do not report positive-only event/pre-event masks as PR-AUC discrimination tests.
- Do not describe a cache as multimodal unless modality flags or audit output show text, audio, and video present.
- Do not commit heavyweight data, model weights, raw media, local caches, or machine-specific paths.
- Do not commit external tensor payloads (`.npy`); only lightweight tensor summaries/manifests and row samples belong in git.
- Do not rerun the raw representation audit or tensor export when existing verified outputs can be reused.
- Do not rebuild the frozen-tensor trained-head runner from stale CSVs; use the implemented runner and frozen tensor contract.
- Do not scale AGAIN sparse teacher beyond the bounded pilot from the current evidence; the 2000-budget confirmatory run failed matched-random and nuisance-control promotion gates.
- Keep machine-specific paths in local `.env`; `.env.example` must stay portable.
- Treat `benchmarks/` and `outputs/` as retained evidence artifacts. Do not use older generated metadata inside them to override the current docs.
- Treat `benchmarks/veatic/veatic_v2_evidence_manifest.json` and `evidence_snapshots/veatic_124_v2_20260616` under the external root as the frozen v2 evidence contract.

## External Assets

Heavy assets live outside git under `NEURAL_BRIDGE_EXTERNAL_ROOT`.

Expected current families:

- `benchmarks/veatic/tribe_cache`
- `evidence_snapshots/veatic_124_v2_20260616`
- `outputs/veatic_124_raw_representation_audit_primary_20260620_152411`
- `tensors/veatic_124_raw_representation_v1`
- `datasets/veatic`
- `models/tribe-mlx`
- `models/upstream-encoders/facebook-w2v-bert-2.0`
- `models/upstream-encoders/meta-llama-Llama-3.2-3B` when authorized and populated
- `models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256`
- `models/vjepa21_mlx/vitg`
- `benchmarks/again`
- `data/external/AGAIN/cleaned` or the local equivalent under `NEURAL_BRIDGE_EXTERNAL_ROOT`
- `models/transcription/mlx-community-whisper-small-mlx`

## Code Discovery

Prefer the codebase-memory MCP graph for code structure:

1. `search_graph`
2. `trace_path`
3. `get_code_snippet`
4. `query_graph`
5. `get_architecture`

Use `rg` for literal strings, docs, configs, generated reports, and stale-name audits.
