# Neural Bridge Roadmap

This is the post-VEATIC-124 v2 roadmap. VEATIC proved the core video-dominant cortical/TRIBE hypothesis for arousal event/spike ranking. The roadmap now focuses on preserving the strict benchmark suite that produced that evidence, maintaining the frozen tensor and trained-head benchmark layers, and using the completed dense AGAIN V-JEPA 2.1/TRIBE cache as the next full-dataset substrate.

## Proven Baseline

Completed:

- VEATIC-124 manifest and video-dominant cortical/TRIBE cache are complete.
- Arousal future-spike/event ranking has validated signal from mostly visual/video cortical/TRIBE features.
- PCA feature modes beat AR, shuffled, random, timestamp, and video/time controls on the strongest spike/event rows.
- Official split spike rows pass controls across current feature families.
- Grouped-video spike F1 improves over AR for PCA modes.
- Balanced event-vs-stable sampling confirms event-conditioned discrimination.
- Temporal context v2 shows short causal windows can improve selected spike-ranking rows.
- Alignment repair selected the final benchmark policy: keep current 0s alignment primary and report offset grids as diagnostics.
- Small v2 evidence reports are tracked in this repo.
- The protected v2 evidence snapshot and checksum verifier are in place.
- Root config/package files describe the current VEATIC/TRIBE workspace.
- The strict suite now audits modality coverage so video-only and full multimodal caches cannot be confused.
- AGAIN local cleaned media has also been audited for embedded audio. The 2026-06-22 sweep found `0/1,095` local AGAIN video containers with audio streams, so current AGAIN work is video plus annotations/telemetry unless a separate audio-bearing source is found.
- The raw cortical representation audit is complete and keeps `cortical_pca64_delta` as the frozen baseline while promoting `pca_sequence_128_causal_past_2s_mean` as the primary trained-head input.
- The v1 model-ready tensor contract is frozen externally with lightweight tracked summaries and verification.
- The frozen-tensor trained-head benchmark is implemented in `backend/scripts/veatic_frozen_tensor_trained_heads.py` with `run_veatic_frozen_tensor_trained_heads_benchmark.py`; it uses MPS, fresh same-row AR, fresh controls, grouped gates, and no prior result-row reuse.
- V-JEPA 2.1 MLX support is implemented in `backend/app/services/mlx_vjepa21_cortical.py` and integrated through `TribeAdapter` when converted weights declare `tensor_layout=vjepa2_1_mlx_port`.
- Dense AGAIN V-JEPA 2.1/TRIBE data generation is complete. A paid H100 run encoded all `995` AGAIN videos with official V-JEPA 2.1 ViT-G at `2Hz` rows / `2Hz` sampling / `256px` / float16, then ran cache-only TRIBE v2. The TRIBE postpass completed `995/995` videos with `0` failures and `243,575` row-level cortical predictions. This does not by itself prove AGAIN benchmark gains.

This is the current scientific foundation, not a hypothesis waiting for another dataset to validate it.

## 1. Evidence Freezing And Reproducibility

Goal: make the proven v2 baseline impossible to lose or confuse with generated run artifacts.

- [x] Create a protected external snapshot for the VEATIC-124 v2 manifest, cache metadata, benchmark JSON/CSV outputs, and tracked reports.
- [x] Add checksums and a manifest that identifies the authoritative files.
- [x] Add one verification command that rechecks the v2 evidence bundle without re-encoding videos.
- [x] Keep small summary reports in git; keep heavy caches and raw outputs external.

## 2. Benchmark Contract

Goal: surface and preserve the exact rules already enforced by the v2 benchmark suite.

- [x] Audit spike rows that prefer non-zero offsets.
- [x] Select 0s alignment as the primary non-leaky benchmark baseline.
- [x] Keep offset grids as diagnostics rather than final-score corrections.
- [x] Confirm no future-looking feature leakage in causal/delta/window features.
- [x] Enforce train-only thresholds, train-only PCA/transforms, grouped-video folds, blocked validation, and shuffled/random/time controls in the benchmark scripts.
- [x] Enforce balanced event-vs-stable scoring for event-conditioned PR-AUC claims.
- [x] Consolidate the already-implemented rules into `backend/scripts/run_veatic_strict_benchmark.py` with a named v2 contract manifest.
- [x] Add a dry-run/status mode that prints the strict benchmark contract without loading cache files.
- [x] Add modality coverage reporting for text/audio/video cache provenance.
- [x] Run and freeze the consolidated strict suite outputs as the new canonical v2 artifact set.
- [ ] Promote full text+audio+video TRIBE only after the guarded `83,84` pilot has gated/local `meta-llama/Llama-3.2-3B` text encoder access and beats or complements the current video-dominant baseline.

## 3. v2 Training Tensor Contract

Goal: freeze the exact model-input interface that heads consume, separate from the already-strict scoring suite.

- [x] Audit raw cortical representation families without re-encoding videos.
- [x] Retain `cortical_pca64_delta` as the frozen v2 baseline comparator.
- [x] Freeze model-ready tensors for `pca_sequence_128_causal_past_2s_mean`, `roi_parcel_features`, `topk_vertices_512`, and `cortical_pca64_delta_frozen_baseline`.
- [x] Record target definitions, masks, split fields, temporal windows, selected vertices, ROI mapping metadata, leakage contracts, and checksums.
- [x] Store immutable training tensors externally under `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`.
- [x] Track lightweight tensor summaries under `outputs/veatic_124_raw_representation_tensor_export_v1`.
- [x] Add representation contract tests that fail on leakage-prone transforms.
- [x] Build simple trained-head runners on `pca_sequence_128_causal_past_2s_mean` before adding recursive or larger architectures.

## 4. Frozen-Tensor Model Heads

Goal: maintain and extend the implemented model-head layer without weakening controls.

- [x] Start with simple, auditable heads over `pca_sequence_128_causal_past_2s_mean`.
- [x] Compare against the v2 PCA/ridge baseline, fresh same-row AR, shuffled/random controls, and PCA64-delta incremental lanes.
- [x] Keep grouped-video and blocked validation as required gates.
- [x] Keep `roi_parcel_features` as an unsupervised side candidate and `topk_vertices_512` as a supervised/cautionary comparison.
- [ ] Only promote recursive heads after simple heads define the floor.
- [ ] Keep CUDA-only HRM-style dependencies out of the main Mac/MPS environment unless isolated.

## 5. V-JEPA 2.1 And AGAIN Scaling

Goal: use the full dense H100 artifact as the next benchmark substrate while preventing raw cache generation from being mistaken for proven AGAIN generalization.

- [x] Add V-JEPA 2.1 ViT-g MLX model loading, preprocessing, RoPE attention, and TRIBE-compatible selected hidden-state output.
- [x] Add ffmpeg/VideoToolbox frame sampling, per-window checkpointing, worker claims, resume guards, and protected VEATIC-cache write refusal.
- [x] Add AGAIN cleaned-dataset audits, boundary-aligned manifests, and full-AGAIN AR context support.
- [x] Run dense full-995 H100 V-JEPA 2.1 ViT-G encoding at `2Hz` rows, `2Hz` sampling, `256px`, float16.
- [x] Run cache-only TRIBE v2 postpass over the dense H100 V-JEPA caches, producing row-level cortical predictions, grouped adapter features, compact temporal diagnostics, quality signals, row alignment, and global manifests.
- [x] Finish local download of `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz` and run a local completeness/schema audit.
- [x] Build a boundary-aligned true-2Hz AGAIN label manifest from native decimal annotations and dense cache `time_seconds`.
- [x] Run the first dense 995-video 2Hz baseline layer: AR-only, raw cortical block summaries, temporal diagnostics, AR+raw, AR+temporal, AR+raw+temporal, shuffled cortical/temporal, random, timestamp/video-time, and quality/motion/luma controls with grouped-video and blocked temporal validation.
- [ ] Build dense 995-video AGAIN PCA/bridge benchmark scripts over the TRIBE output bundle, with train-only PCA widths and learned bridge heads.
- [ ] If an original/audio-bearing AGAIN source is located, run a fresh embedded-audio inventory before adding Wav2Vec-BERT features.

## 6. Productized Evidence Workflow

Goal: make a fresh session or teammate able to inspect, verify, and extend the current result.

- [x] Add a status check for external assets, VEATIC cache, model paths, and tracked evidence artifacts.
- [x] Add compact CLI/report summaries for frozen tensor trained-head runs.
- [x] Add a compact dashboard for the v2 baseline and post-v2 head summaries, including the raw/global early losses, PCA64-delta v2 win, raw-audit compression result, and AR+PCA128 trained-head gate.
- [x] Add a one-command evidence verifier for the tracked reports and external artifact snapshot.
- [x] Add a verified lightweight tensor-export summary for the external v1 tensor contract.
- [ ] Keep machine-specific paths only in local `.env` files.
- [ ] Keep generated heavy outputs out of git.

## 7. Repository Hygiene

Goal: keep fresh sessions focused on the current Neural Bridge system.

- [x] Remove redundant local/static assets and unused atlas copies.
- [x] Replace behavioural ROI-calibration scaffolding with a plain cortical atlas mapper for the viewer.
- [x] Clean root `.env.example`, `.gitignore`, package scripts, and dependency locks.
- [x] Add a lightweight repo-audit command that checks for accidentally staged heavy artifacts.

## De-Scoped

- Retired secondary model expansion as a roadmap item. The current priority is the proven cortical/TRIBE signal.
- Video `83` as an active roadmap concern. Its resampling policy is documented and does not block the v2 claim.
- Legacy app-era workflows as roadmap items.
- Finance or quant-desk prediction.
- Generic chatbot benchmarks as proof of Neural Bridge.
- Exact continuous arousal-value forecasting as the current headline.
- Test-selected lag correction as a headline result.
- CUDA-only training stacks in the main local environment.
- Treating H100 cache generation or TRIBE postpass completion as a benchmark win before the grouped/control AGAIN evaluation is run.
