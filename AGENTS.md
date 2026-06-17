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

## Start Here

Read these files before making claims about project state:

1. `README.md`
2. `docs/current_project_state.md`
3. `docs/veatic_v2_evidence_summary.md`
4. `REQUIREMENTS.md`
5. `ROADMAP.md`
6. `docs/external_assets_manifest.md`

## Default Commands

```bash
npm run audit:repo
python3 backend/scripts/run_veatic_strict_benchmark.py --modality-audit-only
python3 backend/scripts/run_veatic_strict_benchmark.py --dry-run --primary-only
python3 -m pytest -q tests/test_veatic_strict_benchmark_contract.py tests/test_grouped_video_split.py
```

Use `npm run verify` before pushing when dependencies are available.

## Guardrails

- Keep train-only thresholds, train-only transforms, blocked validation, grouped-video validation, and shuffled/random/time controls intact.
- Do not tune decision thresholds on filtered test subsets.
- Do not report positive-only event/pre-event masks as PR-AUC discrimination tests.
- Do not describe a cache as multimodal unless modality flags or audit output show text, audio, and video present.
- Do not commit heavyweight data, model weights, raw media, local caches, or machine-specific paths.
- Keep machine-specific paths in local `.env`; `.env.example` must stay portable.
- Treat `benchmarks/`, `outputs/`, and `reports/` as retained evidence artifacts. Do not use older generated metadata inside them to override the current docs.

## External Assets

Heavy assets live outside git under `NEURAL_BRIDGE_EXTERNAL_ROOT`.

Expected current families:

- `benchmarks/veatic/tribe_cache`
- `datasets/veatic`
- `models/tribe-mlx`
- `models/upstream-encoders/facebook-w2v-bert-2.0`
- `models/upstream-encoders/meta-llama-Llama-3.2-3B` when authorized and populated
- `models/upstream-encoders-mlx/facebook-vjepa2-vitg-fpc64-256`
- `models/transcription/mlx-community-whisper-small-mlx`

## Code Discovery

Prefer the codebase-memory MCP graph for code structure:

1. `search_graph`
2. `trace_path`
3. `get_code_snippet`
4. `query_graph`
5. `get_architecture`

Use `rg` for literal strings, docs, configs, generated reports, and stale-name audits.
