# Neural Bridge

Neural Bridge is a local neuroscience-to-behavior research system. Its current proven core is VEATIC-124 v2: video-dominant TRIBE-predicted cortical response features improve arousal event/spike ranking under leakage-controlled evaluation.

## Current Status

The VEATIC-124 v2 evidence bundle validates these hypotheses:

1. Real cortical/TRIBE features contain stimulus-specific signal for future arousal spike ranking.
2. PCA cortical feature modes beat autoregressive, shuffled, random, timestamp, and video/time controls on the strongest event/spike rows.
3. Balanced event-vs-stable evaluation exposes signal that full-frame continuous MAE hides.
4. Short causal temporal context improves selected future spike-ranking rows over current-only evaluation.
5. A single 0s feature snapshot can underfeed the bridge head for spike/event tasks.
6. The alignment audit resolved the benchmark policy: keep current 0s alignment as the primary non-leaky benchmark and report offset grids as diagnostics, not as a test-derived correction.

Post-v2 raw-representation work has frozen model-ready tensors without re-encoding videos. The frozen baseline remains `cortical_pca64_delta`; the primary learned-head input is `pca_sequence_128_causal_past_2s_mean`; `roi_parcel_features` is an important unsupervised side candidate; and `topk_vertices_512` is exported as a supervised/cautionary comparison.

The first frozen-tensor trained-head runner is implemented and MPS-backed. It recomputes same-row AR and controls fresh, compares PCA128 lanes against the frozen PCA64-delta baseline, and records gate checks for `AR_plus_PCA128`, `residualized_AR_plus_PCA128`, and controls. This is a post-v2 model-head benchmark layer; it does not replace the frozen VEATIC-124 v2 evidence bundle.

V-JEPA 2.1 MLX support and AGAIN sparse-teacher tooling are also implemented. The current AGAIN 50-video sparse teacher work is bounded scaling evidence, not final AGAIN proof. Hybrid sparse PCA128 was negative in the first pilot; a 500-window small-PCA follow-up looked promising, but the later 2000-budget confirmatory run completed 1,948 windows and did not confirm promotion against raw sparse, nuisance, and matched-random controls.

The current claim is intentionally precise: Neural Bridge has evidence for arousal event/spike ranking and temporal-context sufficiency from a mostly visual/video TRIBE cache, not exact continuous arousal-value prediction, a finished downstream product model, or a proven full text+audio+video multimodal cache.

## Key VEATIC-124 v2 Results

- Manifest: 124 videos, 10,357 1 Hz target rows.
- Cache: complete cortical/TRIBE cache under the configured external assets root, at `benchmarks/veatic/tribe_cache`.
- Modality coverage: the current v2 cache is video-dominant. The strict audit reports `122/124` entries with video present and text/audio missing, and `2/124` entries with text+audio+video present.
- Multimodal follow-up: a guarded uncached pilot on videos `83` and `84` now reaches audio extraction, word extraction, Text/Sentence creation, and text feature preparation. It is blocked at the gated `meta-llama/Llama-3.2-3B` text encoder because the local SSD directory is only a placeholder.
- Strongest blocked spike row: `cortical_pca64_delta`, `arousal__future_spike_1_3s`, threshold `0.05`.
- PR-AUC for that row: real `0.2536`, AR `0.1969`, shuffled `0.1840`, random `0.1944`.
- Official split spike rows pass controls across current feature families.
- Grouped-video aggregate spike F1 improves over AR for PCA modes: `cortical_pca_64` `+0.0256`, `cortical_pca64_delta` `+0.0177`.
- Balanced event-vs-stable spike row at threshold `0.05`: `cortical_pca64_delta` PR-AUC `0.3394`, `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random.
- Temporal context v2: 4/4 focused feature-target rows improve over current-only by more than `0.005` PR-AUC; best focused windows are `causal_past_2s`.
- Alignment repair: no safe global lag correction was selected; the final non-leaky policy is `keep_current_0s_as_primary_plus_report_offset_diagnostics`.
- Raw representation audit: `pca_sequence_128_causal_past_2s_mean` beat same-run grouped `cortical_pca64_delta` on primary spike targets by about `+0.039` to `+0.041` PR-AUC, while raw uncompressed ridge was valid but not the best next build target.
- Tensor export v1: `84` representation/split/target contracts and `420` `.npy` tensors were written under the external tensor root with verification `pass`; PCA fit-cache payloads were reused and none were rebuilt.
- Frozen-tensor trained heads: `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py` runs simple MPS-backed heads over the frozen tensor contract. The delivered full run used fresh same-row AR and controls; `AR_plus_PCA128` and `residualized_AR_plus_PCA128` beat `AR_only` and canonical shuffled/random controls across grouped spike gates, while `PCA128_only` did not stably beat AR.
- V-JEPA 2.1 / AGAIN scaling: `backend/app/services/mlx_vjepa21_cortical.py` and the AGAIN sparse-teacher scripts are implemented. The 50-video sparse teacher reports record the negative PCA128 pilot, the promising 500-window small-PCA reanalysis, and the 2000-budget confirmatory run where small-PCA lanes failed matched-random/nuisance promotion gates.

Source summaries:

- [AGENTS.md](AGENTS.md)
- [docs/veatic_v2_evidence_summary.md](docs/veatic_v2_evidence_summary.md)
- [docs/veatic_v2_evidence_freeze.md](docs/veatic_v2_evidence_freeze.md)
- [benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md](benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md)
- [benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md](benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md)
- [benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md](benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md)
- [outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md](outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md)
- [outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md](outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md)
- [benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md](benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md)
- [docs/veatic_raw_representation_audit.md](docs/veatic_raw_representation_audit.md)
- [outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md](outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md)

## What Remains Unproven

- Exact continuous future arousal-value forecasting. Continuous MAE remains diagnostic because zero-change baselines still win most continuous checks.
- Strong universal early-warning claims. The alignment pass supports 0s primary scoring plus transparent offset diagnostics, not a global lag correction.
- The first trained-head layer exists and shows incremental VEATIC spike-ranking signal for AR-plus-PCA128 lanes, but this is still a benchmark layer, not a finished product model or recursive architecture.
- AGAIN generalization remains unproven. The current sparse teacher work is bounded to 50 selected videos; the 2000-budget confirmatory run did not justify broader scaling.
- The exported `topk_vertices_512` tensors are supervised feature-selection artifacts and should remain cautionary unless confirmed in a locked rerun.

## System Shape

```text
backend/app/
  Flask neuro-viewer API plus TRIBE/MLX service adapters, including V-JEPA 2.1 MLX video support.

backend/scripts/
  VEATIC/TRIBE extraction, frozen-tensor trained-head benchmarks, AGAIN sparse-teacher tooling, and the consolidated strict VEATIC benchmark suite.

frontend/
  Vue/Vite cortical cache and stimulus viewer.

docs/, benchmarks/, outputs/
  Current evidence summaries, benchmark reports, and small tracked result artifacts.
```

Active cortical extraction path:

```text
backend/scripts/run_veatic_tribe_cache.py
  -> app.services.tribe_adapter.TribeAdapter
  -> tribe_mlx runtime
  -> MLX/MPS TRIBE path
  -> per-video tribe_raw_output.npz
```

## External Assets

Heavy assets stay outside git. Configure the external assets root in `.env` for each workstation:

```bash
NEURAL_BRIDGE_EXTERNAL_ROOT=/path/to/neural-bridge-assets
```

This external root contains model weights, Hugging Face caches, raw/processed datasets, TRIBE caches, benchmark caches, and large generated outputs.

The frozen raw-representation tensor payloads live outside git at:

```text
${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1/
```

Tracked lightweight summaries for that export live at `outputs/veatic_124_raw_representation_tensor_export_v1/`.

See [docs/external_assets_manifest.md](docs/external_assets_manifest.md).

## Local Setup

Create local configuration:

```bash
cp .env.example .env
```

Keep machine-specific paths in `.env`; it is ignored by git. The template is the tracked contract for fresh machines and Codex sessions.

Backend:

```bash
cd <repo-root>/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Frontend:

```bash
cd <repo-root>/frontend
npm install
npm run dev
```

Repo-root helpers:

```bash
cd <repo-root>
npm install
npm run audit:repo
npm run backend   # terminal 1
npm run frontend  # terminal 2
```

Full benchmark workflows require the configured external assets root to be mounted or otherwise available.

## Strict Benchmark

Use the unified VEATIC-124 suite as the default benchmark entrypoint:

```bash
python3 backend/scripts/run_veatic_strict_benchmark.py --primary-only
```

It consolidates the formerly separate event/spike, event-conditioned, alignment, and temporal-context checks into one coordinated run. The suite includes AR, shuffled cortical rows, split-local shuffles, Gaussian feature controls, label shuffles, feature shuffles, timestamp-only, video/time-only, majority, fixed-split holdouts, grouped-video holdouts, zero-change diagnostics, and a single-backend policy.

Check cache modality coverage before describing a run as multimodal:

```bash
python3 backend/scripts/run_veatic_strict_benchmark.py --modality-audit-only
```

Verify the frozen v2 evidence bundle without re-encoding videos:

```bash
npm run evidence:verify
```

Do not re-encode all 124 VEATIC videos for multimodal coverage. Only videos `83` and `84` contain audio streams, so the current actionable path is to finish the gated/local text encoder dependency for the two-video pilot, then compare that pilot against the video-dominant cache.

The current v2 evidence should be described as visual/video cortical TRIBE unless a new cache passes full text+audio+video coverage checks.

## Post-v2 Benchmarks

Run the frozen tensor trained-head benchmark only when intentionally refreshing post-v2 head results:

```bash
python3 backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py
```

This runner requires MPS, reuses frozen external tensors, recomputes same-row AR and controls, and refuses CPU sklearn fallback. It is the implemented learned-head benchmark layer for `pca_sequence_128_causal_past_2s_mean`; do not rebuild it from old report CSVs.

AGAIN sparse-teacher tooling is implemented under `backend/scripts/again_*` and `tools/run_again_*`. The tracked reports in `reports/again_*20260622_005732.md` summarize the original bounded PCA128 pilot; `reports/again_sparse_tribe_teacher_500_*_20260622_pca_width_reanalysis_v2.md` is the fresh cache-only smaller-width follow-up. Treat both as bounded 50-video evidence, not as full-AGAIN proof.

## Guardrails

- Keep train-only thresholding, train-only PCA, grouped-video validation, and blocked validation intact.
- Do not promote positive-only event/pre-event masks as PR-AUC discrimination tests; use recall/top-k or balanced event-vs-stable rows.
- Do not tune thresholds on filtered test subsets.
- Do not treat continuous MAE diagnostics as the headline result.
- Do not add CUDA/Triton/FlashAttention stacks to the Mac/MPS runtime without an isolated adapter plan.
- Do not commit heavyweight caches or model weights.

## Roadmap

The active post-v2 roadmap is in [ROADMAP.md](ROADMAP.md). The short version is:

1. Freeze and package the v2 evidence bundle.
2. Surface the already-strict v2 benchmark rules as a named contract and verifier.
3. Use the frozen raw-representation tensor contract for model heads.
4. Maintain and extend the implemented MPS trained-head benchmark on `pca_sequence_128_causal_past_2s_mean`, with `cortical_pca64_delta` retained as the baseline comparator.
5. Productize the evidence verifier, trained-head summaries, and dashboard workflow.

## Root Files

- `.env.example` is the tracked configuration template.
- `.env` is local-only and ignored by git.
- `AGENTS.md` is the current Codex/fresh-session operating contract.
- `benchmarks/veatic/veatic_v2_evidence_manifest.json` is the checksum manifest for the frozen v2 evidence bundle.
- `package.json` only provides root helper commands; frontend dependencies and their lockfile live under `frontend/`.

## License

AGPL-3.0. Check upstream licenses for bundled or referenced research projects before redistribution.
