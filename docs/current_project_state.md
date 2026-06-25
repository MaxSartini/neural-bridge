# Current Project State - 2026-06-25

This is the short operating snapshot for the cleaned Neural Bridge repo after the VEATIC-124 v2 evidence pass, raw representation audit, v1 model-ready tensor export, frozen-tensor trained-head benchmark, V-JEPA 2.1 / AGAIN sparse-teacher implementation, and the dense 995-video H100 AGAIN cache/postpass run.

## Repo

- Active repo: this Git checkout.
- External asset root: configured locally through `.env` as `NEURAL_BRIDGE_EXTERNAL_ROOT`.

The repo should stay lightweight. Heavy research assets belong on the external drive, not in git.

## Active Scientific Direction

Neural Bridge is testing where predicted neural response trajectories improve human-response forecasts under controlled baselines.

Current evidence is strongest for VEATIC-124 video-dominant cortical/TRIBE arousal event and spike ranking. v2 has validated specific hypotheses around event/spike ranking and causal temporal context. It still should not claim exact continuous arousal-value forecasting or full text+audio+video multimodal TRIBE evidence.

The model-head input is frozen as tensors rather than only described in reports. Keep `cortical_pca64_delta` as the frozen v2 baseline. The implemented trained-head layer uses `pca_sequence_128_causal_past_2s_mean` first, includes fresh same-row AR and controls, keeps `roi_parcel_features` as an important side candidate, and treats `topk_vertices_512` as supervised/cautionary.

The AGAIN/V-JEPA 2.1 path is implemented as scaling infrastructure. It is not a new proven baseline. The current tracked sparse teacher work covers a 50-video selector subset. Hybrid sparse PCA128 failed the first promotion gates. A 500-window cache-only smaller-width reanalysis looked promising, but the 2000-budget confirmatory run remains non-promotable. V-JEPA 2.1 is the encoding engine for TRIBE v2; the tested compressed lanes are AR + train-only PCA of frozen TRIBE v2 cortical predictions, not PCA-only and not PCA of raw V-JEPA tokens. AR + locked PCA32 beat AR and matched random under delta-over-AR, but it remains slightly below AR + raw sparse current/causal mean and fails the same-width shuffled PCA32 nuisance control. A follow-up true fixed-random same-budget rerun matched hybrid at 849 unique windows per arm and no longer beat hybrid, so the earlier undersized fixed-random caveat is closed without changing the no-scale decision.

Current AGAIN scout follow-up: ViT-L scout support is implemented through `vjepa21_vitl_dgrauet_mlx_scout`. New scout feature rows use canonical `scout_novelty_z` plus `scout_model_name`; legacy `vjepa_b_novelty_z`/`vjepa_l_novelty_z` aliases are compatibility fields only. The sparse ViT-G/TRIBE stage consumes the canonical scout field with B/L fallback, so old column names must not be read as forcing a weaker B scout. Practical scout runtime findings on the local M2 Max: ViT-L at `384px`, `16` frames, `1Hz` was about `100s/video`; ViT-L at `256px`, `16` frames, `1Hz` is about `35s/video`. Batch/concurrency probes did not help. Compiled MLX scout forward is enabled for future runs but has not materially changed full-video throughput. Active scout and future ViT-G/TRIBE encoding should run from internal scratch, then mirror completed caches and reports back to the external SSD.

The current AGAIN benchmark manifest is a deliberately built boundary-aligned `1Hz` view (`again_boundary_aligned_1hz_manifest.csv`) from native decimal-timestamp annotations. Therefore a `2Hz` ViT-G/TRIBE pass over the current manifest is only a denser feature-aggregation ablation around the same 1Hz labeled centers, not a source of additional supervised rows. True 2Hz supervised claims require a new boundary-aligned 2Hz manifest and fresh selector rows from the native annotations.

Dense AGAIN cache status: the expensive data-generation run has now been completed on H100 rather than locally. The run used official V-JEPA 2.1 ViT-G, full dense `995` videos, `2Hz` rows, `2Hz` sampling, `256px`, and float16 precision. It did not use sparse windows, scout filtering, PCA, bridge training, or benchmarking. The follow-up cache-only TRIBE v2 postpass consumed precomputed V-JEPA caches only, preserved the `[rows,20,1,1408] -> [rows,2,1408] -> [1,2,1408,rows]` adapter contract, and wrote row-level `cortical_prediction [rows,20484]` outputs. Final postpass status: `995` videos succeeded, `0` failed, `243,575` rows. The Drive folder is `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`; the local download target is `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`.

This dense H100 artifact changes the next safe move. The sparse 50-video AGAIN evidence still should not be scaled as-is, but the full 995-video dense V-JEPA/TRIBE output bundle is now the correct substrate for later local PCA, temporal diagnostics, AR + cortical bridge training, shuffled/random/time controls, quality-filtered checks, and full grouped-video benchmarks. Do not describe the H100 postpass itself as a positive benchmark result; it generated the cache needed to run those benchmarks.

AGAIN multimodal status: TRIBE/V-JEPA infrastructure can support multimodal inputs, but the local cleaned AGAIN video mirrors currently available to this repo are video-only containers. A 2026-06-22 `ffprobe` sweep over the internal scratch and external SSD AGAIN roots checked `1,095` `.webm`/video containers and found `0` embedded audio streams with `0` probe errors. `facebook/w2v-bert-2.0` is present and recognized as an encoder, but it has no usable AGAIN audio stream in these cleaned files. The external `meta-llama-Llama-3.2-3B` path remains a placeholder; a real MLX text candidate exists at `/Users/maxsartini/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit` and passes the repo's MLX text-model directory check.

MLX memory knobs are verified. `iogpu.wired_limit_mb` exists on this macOS install and is referenced by the installed MLX stubs for raising the system wired limit. MLX itself does not parse `MLX_MAX_MAPPED_MEM_MB`; Neural Bridge implements that name as a compatibility shim that calls `mx.set_wired_limit(bytes)` in heavy scripts. Without changing sysctl, MLX reports `max_recommended_working_set_size` around `24.96 GiB` on this 32 GiB M2 Max. `MLX_MAX_MAPPED_MEM_MB=24576` applies; `26624` requires first raising the system limit with `sudo sysctl iogpu.wired_limit_mb=26624`.

## Current Benchmark Assets

- Complete VEATIC manifest: `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- Manifest rows: 10,357 at 1 Hz
- Complete cortical cache: `<external-assets-root>/benchmarks/veatic/tribe_cache`
- Cache shape contract: per-video `tribe_raw_output.npz` with required key `predictions`
- Modality coverage: current audit shows `122/124` cache entries are video-only (`text` and `audio` missing) and `2/124` contain text+audio+video.
- Main targets: `valence`, `arousal`

Current feature families:

- `cortical_global`
- `cortical_global_delta`
- `cortical_pca_64`
- `cortical_pca64_delta`
- raw cortical trajectories used for the raw-representation audit and available for future loader work

Raw-representation/tensor-export assets now available:

- Raw representation audit output: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/outputs/veatic_124_raw_representation_audit_primary_20260620_152411`
- Tracked lightweight audit copy: `outputs/veatic_124_raw_representation_audit_primary_20260620_152411`
- Frozen tensor root: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`
- Tracked tensor summary: `outputs/veatic_124_raw_representation_tensor_export_v1`
- Tensor export coverage: `84` contracts, `420` external `.npy` files, verification `pass`
- PCA cache reuse: `14` cache entries reused, `0` rebuilt
- Video `83`: included in the all-video tensor export; exclude-video-83 tensor sensitivity was intentionally skipped

Post-v2 trained-head and scaling assets now available:

- Frozen tensor trained-head code: `backend/scripts/veatic_frozen_tensor_adapter.py`, `backend/scripts/veatic_frozen_tensor_trained_heads.py`, and `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py`
- Trained-head policy: MPS required, no CPU sklearn fallback, fresh same-row AR, fresh shuffled/random controls, no prior result-row reuse
- Trained-head result handle from the completed run: `outputs/veatic_124_frozen_tensor_trained_heads_mps_20260620_full_lightweight.zip`
- V-JEPA 2.1 MLX adapter: `backend/app/services/mlx_vjepa21_cortical.py`
- V-JEPA 2.1 selection trigger: converted MLX weights with `tensor_layout=vjepa2_1_mlx_port`
- V-JEPA 2.1 ViT-L scout: `vjepa21_vitl_dgrauet_mlx_scout`, currently best used as `1Hz`, `16` frames, `256px`, batch `1`, internal scratch active storage.
- AGAIN audio inventory: `reports/again_video_audio_stream_inventory_20260622.md`, confirming `0/1,095` local AGAIN video containers with embedded audio streams.
- AGAIN current tracked reports: `reports/again_real_scout_selector_validation_20260621_230940_n50.md`, `reports/again_full_ar_context_20260622_005713.md`, `reports/again_sparse_tribe_teacher_500_*_20260622_005732.md`, `reports/again_sparse_tribe_teacher_500_*_20260622_pca_width_reanalysis_v2.md`, corrected `reports/again_sparse_tribe_teacher_2000_*_20260622_2000_small_pca_confirmatory_v3.md`, and true same-budget fixed-random `reports/again_sparse_tribe_teacher_2000_true_fixed_random_same_budget_*_20260622_2000_true_fixed_random_same_budget_v3.md`
- H100 dense AGAIN V-JEPA/TRIBE asset: Google Drive folder `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`, local pull target `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`. Expected contents include global manifests, `row_index.csv`, `row_index.parquet`, `video_metadata.csv`, split manifests, `BASELINE_READINESS.md`, `README_OUTPUT_SCHEMA.md`, and `per_video/<video_id>/` folders with `tribe_v2_cortical_predictions.npz`, `baseline_features_rowlevel.npz`, `vjepa_temporal_diagnostics.npz`, `rows_aligned.csv`, `input_mapping.json`, `diagnostics.json`, `manifest.json`, and `status.json`.

Current v2 evidence reports now tracked in this repo:

- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md`
- `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md`
- `outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md`
- Frozen evidence manifest: `benchmarks/veatic/veatic_v2_evidence_manifest.json`
- External protected snapshot: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/evidence_snapshots/veatic_124_v2_20260616`

Current default benchmark entrypoint:

```bash
python3 backend/scripts/run_veatic_strict_benchmark.py --primary-only
```

Fresh Codex sessions should read `AGENTS.md` and run `npm run audit:repo` before changing repo state.
Run `npm run evidence:verify` before using the v2 baseline as a reference.

Use `--dry-run` to print the strict contract and control ledger without loading the external cache.
Use `--modality-audit-only` to report cache-level text/audio/video coverage.

## Validated v2 Findings

- Video-dominant cortical/TRIBE features improve arousal future-spike/event ranking under blocked validation.
- `cortical_pca64_delta` is the strongest blocked full-frame spike row at threshold `0.05`: PR-AUC `0.2536` versus AR `0.1969`, shuffled `0.1840`, and random `0.1944`.
- Official split event/spike rows pass controls across the current feature families.
- Grouped-video validation improves aggregate spike F1 over AR for PCA modes.
- Balanced event-vs-stable sampling confirms event-conditioned discrimination for the strongest spike rows.
- Temporal context v2 shows short causal windows improve selected future arousal spike ranking over current-only evaluation.
- Alignment policy is resolved: current 0s alignment remains the primary non-leaky benchmark, while offset-grid results are diagnostics.
- Raw representation audit promotes `pca_sequence_128_causal_past_2s_mean` as the best learned-head input for event/spike ranking; `roi_parcel_features` is the best compact side candidate; `topk_vertices_512` is useful but supervised/cautionary; raw uncompressed ridge is valid but not the best next build target.
- Tensor export v1 materialized model-ready train/test tensors for `pca_sequence_128_causal_past_2s_mean`, `roi_parcel_features`, `topk_vertices_512`, and `cortical_pca64_delta_frozen_baseline` across `blocked`, `official`, and `grouped_0..4` splits for the three primary targets.
- Frozen tensor trained heads are implemented. In the completed MPS run, `AR_plus_PCA128` and `residualized_AR_plus_PCA128` beat `AR_only`, canonical shuffled/random controls, and their PCA64-delta incremental counterparts across grouped spike gates; `PCA128_only` did not stably beat AR.
- AGAIN sparse teacher is implemented but not promoted. The 2000-budget confirmatory run completed 1,948 sparse V-JEPA/TRIBE windows on the same 50-video selector subset. AR + locked PCA32 shows a matched-random delta-over-AR signal, but it remains below AR + raw sparse and fails same-width shuffled-PCA32 nuisance control.
- Dense AGAIN data generation is complete but not yet benchmarked. The H100 run produced the full 995-video 2Hz V-JEPA 2.1/TRIBE v2 working bundle. This enables full-dataset downstream PCA and bridge experiments without re-encoding videos, but no scientific promotion gate has been run on the dense bundle yet.

## Benchmark Rules

- Full-frame VEATIC rows remain the main baseline.
- Event-conditioned rows are diagnostics unless balanced against stable controls.
- Positive-only pre-event and event masks should report recall/top-k style diagnostics, not PR-AUC as the main claim.
- Thresholds must be fit on train data only.
- PCA and other transforms must be fit on train data only.
- Controls include AR, shuffled cortical rows, split-local shuffles, Gaussian features, label shuffles, feature shuffles, timestamp-only, video/time-only, majority, fixed-split holdouts, grouped-video holdouts, zero-change diagnostics, and one backend policy per final run.
- CPU/MPS device consistency should be checked before mixing thresholded results.
- Do not describe a cache as multimodal unless `modality_missing_flags` or `tribe_summary.event_quality` show text, audio, and video present.

## Remaining Work

1. Finish pulling the H100 dense TRIBE postpass bundle locally and run a quick local completeness audit against the global manifest before using it.
2. Build the next AGAIN benchmark layer from the dense 995-video TRIBE bundle: AR-only, quality/motion/luma controls, timestamp/video-time controls, shuffled/random controls, PCA widths, train-only PCA transforms, grouped-video folds, and blocked temporal checks.
3. Do not scale AGAIN sparse teacher from the current 50-video sparse runs. Any future sparse attempt needs a new selector/subset design, not just more windows on this same subset.
4. Keep grouped-video validation, blocked validation, and controls as promotion gates for any learned head or sparse-teacher follow-up.
5. Finish the guarded `83,84` multimodal pilot after populating or authorizing the gated `meta-llama/Llama-3.2-3B` text encoder.
6. Productize the tensor/evidence/trained-head summaries into the benchmark dashboard.

Current multimodal pilot status:

- `facebook/w2v-bert-2.0` is present on the external SSD.
- The local LM Studio directory `/Users/maxsartini/.lmstudio/models/mlx-community/Llama-3.2-3B-Instruct-4bit` is a real MLX Llama 3.2 3B Instruct 4-bit model directory and passes the repo's MLX text-model directory check.
- The pilot reaches audio extraction, word extraction, Text/Sentence creation, and text feature preparation.
- It is blocked by gated/missing `meta-llama/Llama-3.2-3B` text encoder assets.
- A full VEATIC-124 multimodal re-encode is not warranted because only videos `83` and `84` contain audio streams.

## Next Safe Move

Use the frozen v2 evidence bundle and frozen raw-representation tensor contract as the VEATIC baseline. The first trained-head layer already exists; do not rebuild it from stale report CSVs. For AGAIN, the next safe move is to finish the local copy of the dense H100 TRIBE bundle and run benchmark/control layers over that full 995-video artifact, not more sparse windows on the same 50-video selector subset.
