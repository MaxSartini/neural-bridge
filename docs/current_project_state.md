# Current Project State - 2026-06-22

This is the short operating snapshot for the cleaned Neural Bridge repo after the VEATIC-124 v2 evidence pass, raw representation audit, v1 model-ready tensor export, frozen-tensor trained-head benchmark, and V-JEPA 2.1 / AGAIN sparse-teacher implementation.

## Repo

- Active repo: this Git checkout.
- External asset root: configured locally through `.env` as `NEURAL_BRIDGE_EXTERNAL_ROOT`.

The repo should stay lightweight. Heavy research assets belong on the external drive, not in git.

## Active Scientific Direction

Neural Bridge is testing where predicted neural response trajectories improve human-response forecasts under controlled baselines.

Current evidence is strongest for VEATIC-124 video-dominant cortical/TRIBE arousal event and spike ranking. v2 has validated specific hypotheses around event/spike ranking and causal temporal context. It still should not claim exact continuous arousal-value forecasting or full text+audio+video multimodal TRIBE evidence.

The model-head input is frozen as tensors rather than only described in reports. Keep `cortical_pca64_delta` as the frozen v2 baseline. The implemented trained-head layer uses `pca_sequence_128_causal_past_2s_mean` first, includes fresh same-row AR and controls, keeps `roi_parcel_features` as an important side candidate, and treats `topk_vertices_512` as supervised/cautionary.

The AGAIN/V-JEPA 2.1 path is implemented as scaling infrastructure. It is not a new proven baseline. The current tracked sparse teacher pilot covers 50 selected AGAIN videos and 480 completed windows; hybrid sparse PCA128 failed promotion gates against AR and coverage-matched random controls.

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
- AGAIN current tracked reports: `reports/again_real_scout_selector_validation_20260621_230940_n50.md`, `reports/again_full_ar_context_20260622_005713.md`, and `reports/again_sparse_tribe_teacher_500_*_20260622_005732.md`

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
- AGAIN sparse teacher is implemented but not promoted. The current 50-video pilot completed 480 sparse V-JEPA/TRIBE windows and failed hybrid sparse PCA128 promotion gates.

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

1. If continuing the sparse-teacher question, run a cache-only smaller PCA-width follow-up (`8/16/32/64`, train/inner-validation selected) on the existing sparse rows before spending more ViT-G/TRIBE forwards.
2. Keep grouped-video validation, blocked validation, and controls as promotion gates for any learned head or sparse-teacher follow-up.
3. Finish the guarded `83,84` multimodal pilot after populating or authorizing the gated `meta-llama/Llama-3.2-3B` text encoder.
4. Productize the tensor/evidence/trained-head summaries into the benchmark dashboard.

Current multimodal pilot status:

- `facebook/w2v-bert-2.0` is present on the external SSD.
- The pilot reaches audio extraction, word extraction, Text/Sentence creation, and text feature preparation.
- It is blocked by gated/missing `meta-llama/Llama-3.2-3B` text encoder assets.
- A full VEATIC-124 multimodal re-encode is not warranted because only videos `83` and `84` contain audio streams.

## Next Safe Move

Use the frozen v2 evidence bundle and frozen raw-representation tensor contract as the baseline. The first trained-head layer already exists; do not rebuild it from stale report CSVs. For the immediate sparse-sample concern, prefer a cache-only smaller PCA-width follow-up on the existing 50-video sparse teacher rows before considering larger or recursive architectures.
