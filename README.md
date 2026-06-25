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

V-JEPA 2.1 MLX support and AGAIN sparse-teacher tooling are also implemented. The current AGAIN 50-video sparse teacher work is bounded scaling evidence, not final AGAIN proof. Hybrid sparse PCA128 was negative in the first pilot; a 500-window small-PCA follow-up looked promising, but the later 2000-budget confirmatory work remains non-promotable. V-JEPA 2.1 is the encoding engine for TRIBE v2, and the tested compressed lanes are AR + train-only PCA of frozen TRIBE v2 cortical predictions. AR + locked PCA32 beats matched random under delta-over-AR, but it remains slightly below AR + raw sparse and fails a same-width shuffled PCA32 nuisance control.

The project has now moved beyond sparse AGAIN pilots for the main scaling artifact. A paid H100 run completed a dense 995-video AGAIN V-JEPA 2.1 ViT-G encode at `2Hz` rows, `2Hz` sampling, `256px`, and float16 precision, then ran a cache-only TRIBE v2 postpass over those V-JEPA caches. The postpass completed `995/995` videos with `0` failed videos and `243,575` row-level 2Hz cortical predictions. The Drive-hosted TRIBE output bundle is `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`; the local pull target is `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`. This is a data-generation milestone, not a benchmark result: no PCA, bridge training, delta/spike evaluation, or promotion claim was made during the H100 postpass.

AGAIN scout iteration is model-stamped. ViT-B and ViT-L scout outputs write canonical `scout_novelty_z` plus `scout_model_name`; old `vjepa_b_*` names are compatibility aliases, not the contract for new runs. ViT-L scout (`vjepa21_vitl_dgrauet_mlx_scout`) remains useful for future selector experiments, but it is no longer the immediate path to a first full AGAIN cortical cache because the dense H100 V-JEPA/TRIBE artifact now exists.

The current AGAIN supervised manifest is still a boundary-aligned `1Hz` view built from native decimal-timestamp annotations. The dense H100 cache is `2Hz`, so future 2Hz supervised claims require explicit label alignment or a boundary-aligned 2Hz label manifest. Until that exists, the 2Hz cache should be treated as a richer feature substrate, not automatically as twice as many supervised target rows.

Current local AGAIN media is not multimodal even though TRIBE itself can accept video, audio, and text. A 2026-06-22 `ffprobe` sweep over the local internal and external AGAIN roots found `0/1,095` video containers with embedded audio streams and `0` probe errors. Treat AGAIN runs from these cleaned `.webm` mirrors as video plus annotations/telemetry unless a separate/original audio-bearing source is identified and audited.

For MLX memory, use the implemented runtime shim rather than unverified environment variables. `MLX_MAX_MAPPED_MEM_MB=<MB>` is supported by Neural Bridge scripts and is translated to `mx.set_wired_limit(...)`; upstream MLX in this environment does not parse that variable directly. The local 32 GiB M2 Max reports about `24.96 GiB` recommended working set by default, so `MLX_MAX_MAPPED_MEM_MB=24576` applies without sysctl. Higher values such as `26624` require first raising macOS' system wired limit with `sudo sysctl iogpu.wired_limit_mb=26624`.

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
- V-JEPA 2.1 / AGAIN scaling: `backend/app/services/mlx_vjepa21_cortical.py` and the AGAIN sparse-teacher scripts are implemented. The 50-video sparse teacher reports record the negative PCA128 pilot, the promising 500-window small-PCA reanalysis, and the 2000-budget confirmatory work where AR + locked PCA32 shows matched-random delta-over-AR signal but still fails raw sparse and same-width shuffled-PCA32 promotion gates.
- Dense AGAIN H100 cache: 995 videos were encoded with official V-JEPA 2.1 ViT-G at `2Hz` rows / `2Hz` sampling / `256px` / float16, then passed through cache-only TRIBE v2. TRIBE postpass output is row-level `cortical_prediction [rows, 20484]` plus timestamps, grouped V-JEPA adapter features, quality signals, row alignment, compact temporal diagnostics, manifests, and baseline-readiness metadata. The completed postpass manifest reports `995` successes, `0` failures, and `243,575` rows.
- ViT-L scout scaling: `backend/scripts/again_real_scout_selector_validation.py` supports `vjepa21_vitl_dgrauet_mlx_scout`, canonical `scout_novelty_z`, internal video/cache overrides, compiled MLX forward, and the `MLX_MAX_MAPPED_MEM_MB` compatibility shim.
- AGAIN audio audit: `reports/again_video_audio_stream_inventory_20260622.md` records the local embedded-stream sweep that found no audio streams in the cleaned AGAIN video mirrors.

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
- AGAIN benchmark generalization remains unproven. The sparse teacher work is bounded to 50 selected videos and did not justify broader sparse scaling. The dense H100 995-video cache and TRIBE postpass are now available as upstream data assets for later local PCA, bridge training, controls, and benchmarking; they are not themselves a scored AGAIN result.
- The exported `topk_vertices_512` tensors are supervised feature-selection artifacts and should remain cautionary unless confirmed in a locked rerun.

## System Shape

```text
backend/app/
  Flask neuro-viewer API plus TRIBE/MLX service adapters, including V-JEPA 2.1 MLX video support.

backend/scripts/
  VEATIC/TRIBE extraction, frozen-tensor trained-head benchmarks, AGAIN sparse-teacher tooling, H100 cache-only TRIBE postpass tooling, and the consolidated strict VEATIC benchmark suite.

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

For AGAIN specifically, do not assume the cleaned local `.webm` mirrors contain sound. The current audit found no embedded audio streams across the internal scratch and external SSD AGAIN video roots, so Wav2Vec-BERT cannot add signal to those exact files without a different audio-bearing source.

## Post-v2 Benchmarks

Run the frozen tensor trained-head benchmark only when intentionally refreshing post-v2 head results:

```bash
python3 backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py
```

This runner requires MPS, reuses frozen external tensors, recomputes same-row AR and controls, and refuses CPU sklearn fallback. It is the implemented learned-head benchmark layer for `pca_sequence_128_causal_past_2s_mean`; do not rebuild it from old report CSVs.

AGAIN sparse-teacher tooling remains in the codebase for archaeology and future bounded experiments, but the old sparse report files have been removed from git because they are no longer the active path. For current full-dataset AGAIN work, start from the dense H100 bundle described in `docs/again_dense_h100_cache.md`.

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
