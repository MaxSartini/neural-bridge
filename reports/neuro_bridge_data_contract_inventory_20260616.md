# Neuro Bridge Data Contract Inventory - 2026-06-16

## 1. Executive Summary

This pass inspected existing TRIBE / Neuro Bridge / VEATIC benchmark artifacts only. It read the previous recursive repo readiness report, the current VEATIC benchmark code, the TRIBE adapter save path, VEATIC manifests, benchmark JSON/CSV outputs, temporal fairness/context outputs, and representative cached TRIBE NPZ files.

What exists:

- A complete 124-video VEATIC manifest at `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl` with 10,357 1 Hz rows, valence/arousal targets, source annotation paths, media paths, and three split fields.
- A complete 124-video cache under `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`, with 124 `tribe_raw_output.npz` files.
- Current 124-video benchmark outputs for `cortical_global`, `cortical_global_delta`, `cortical_pca_64`, and `cortical_pca64_delta`.
- Current 89-video benchmark, event-conditioned, event/spike, and device-consistency outputs.
- Temporal fairness and temporal context v2 outputs under `outputs/`, including leakage audits, selected offsets, context windows, representation ablations, real-vs-control specificity, and reused-artifact hashes.

What is missing or not frozen yet:

- No protected immutable baseline snapshot directory was found for the current 124-video baseline artifacts.
- The 124-video manifest report records one cache/manifest row mismatch: video `83` has cached predictions shape `(263, 20484)` for `143` manifest rows and is included by linear resampling in the benchmark.
- No current production training tensor loader contract was found; the usable contract is visible from benchmark code and artifacts, not formalized as a separate loader spec.
- Subcortical VEATIC cache is not present in the main 124-video cache; subcortical exists only in smoke/test artifacts inspected here.

Enough information exists for the next operational step: freeze/copy the current benchmark baseline artifacts into a protected baseline snapshot before any model work. It is not enough to start recursive model code safely, because the baseline freeze and loader contract are not yet explicit.

## 2. Read-Only Safety Confirmation

- Current working directory checked: `/Users/maxsartini/Neural Bridge`.
- `git status` failed with: `fatal: not a git repository (or any of the parent directories): .git`.
- No model/training/benchmark code was edited.
- No production configs or data preparation code were edited.
- No dependencies were installed.
- No training, extraction, benchmark rerun, or model build was run.
- Cached artifacts were inspected read-only and were not moved, deleted, or modified.
- The only intended file modification from this step is this new report: `reports/neuro_bridge_data_contract_inventory_20260616.md`.

## 3. Artifact Map

| Path | Type | Size | Modified | Role | Freeze status |
| --- | ---: | ---: | --- | --- | --- |
| `reports/recursive_repos_install_and_readiness_20260616.md` | Markdown | 18 KB | 2026-06-16 | Previous readiness report | Context only |
| `backend/scripts/run_veatic_neuro_benchmark.py` | Python | 85 KB | 2026-06-16 13:18 | Current benchmark contract/code | Do not edit before freeze |
| `backend/app/services/tribe_adapter.py` | Python | 37 KB | 2026-06-13 17:59 | Writes `tribe_raw_output.npz` and summaries | Do not edit before freeze |
| `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl` | JSONL | 7.2 MB | 2026-06-16 13:01 | 124-video row manifest, targets, splits | Freeze |
| `benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json` | JSON | 87 KB | 2026-06-16 13:01 | Manifest validation/cache report | Freeze |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/*/tribe_raw_output.npz` | NPZ | 124 files, about 1.8-43 MB each | 2026-06-12 to 2026-06-16 | Main cached TRIBE cortical features | Freeze |
| `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_global_mpsgram_cpu.json` | JSON | 1.7 MB | 2026-06-16 13:02 | 124-video 6-feature baseline metrics | Freeze |
| `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_global_delta_mpsgram_cpu.json` | JSON | 1.7 MB | 2026-06-16 13:04 | 124-video global+dynamics metrics | Freeze |
| `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_pca_64_mpsgram_cpu.json` | JSON | 1.5 MB | 2026-06-16 13:25 | 124-video PCA-64 metrics | Freeze |
| `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_pca64_delta_mpsgram_cpu.json` | JSON | 1.6 MB | 2026-06-16 13:33 | 124-video PCA-64+dynamics metrics | Freeze |
| `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.event_masks.csv` | CSV | 2.5 MB | 2026-06-16 13:46 | Event-conditioned mask metrics | Freeze |
| `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.per_video.csv` | CSV | 595 KB | 2026-06-16 13:46 | Per-video event metrics | Freeze |
| `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.diagnostics.csv` | CSV | 229 KB | 2026-06-16 13:42 | Event/spike diagnostics | Freeze |
| `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.local_targets.csv` | CSV | 298 KB | 2026-06-16 13:42 | Local target retest outputs | Freeze |
| `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.per_video.csv` | CSV | 112 KB | 2026-06-16 13:42 | Event/spike per-video diagnostics | Freeze |
| `benchmarks/veatic/veatic_89_benchmark_device_consistency_20260616.*` | JSON/CSV/MD | 27 KB to 21 MB | 2026-06-16 05:21-05:26 | CPU/MPS consistency audit | Freeze as diagnostic |
| `outputs/veatic_124_temporal_fairness_20260616_1509/*` | CSV/JSON/MD | 672 B to 43 MB | 2026-06-16 15:38 | Temporal fairness outputs | Freeze |
| `outputs/veatic_124_temporal_context_v2_20260616_1557/*` | CSV/JSON/MD | 537 B to 2.0 MB | 2026-06-16 16:11 | Temporal context v2 outputs | Freeze |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_smoke/cortical_plus_subcortical/video_52/tribe_raw_output.npz` | NPZ | 2.2 MB | 2026-06-07 | Smoke cortical+subcortical feature archive | Reference only |
| `backend/uploads/tribe_text_smoke_test/tribe_raw_output.npz` | NPZ | 399 KB | 2026-06-05 | Text smoke cortical+subcortical archive | Reference only |

## 4. Current Benchmark Contract

Visible from `backend/scripts/run_veatic_neuro_benchmark.py` and current result JSONs:

- Current input: a JSONL VEATIC manifest plus a cache directory containing one `tribe_raw_output.npz` per `video_id`.
- Feature source: `cache_dir / video_id / "tribe_raw_output.npz"`.
- Required cortical feature key: `predictions`.
- Optional subcortical feature key: `subcortical_predictions`; absent in the main 124-video cache.
- Targets: manifest row `targets.valence` and `targets.arousal`.
- Derived targets: `raw`, `delta_prev_1s`, `residual_after_persistence`, future state/change at 1/2/3s, persistence/time residuals, and event targets such as future spike/drop/rise/reversal/peak/recovery.
- Splits: manifest row `splits.official_70_30`, `splits.blocked_temporal_gap`, and `splits.leave_video_out_group`.
- Current predictions/metrics: benchmark JSONs contain metric tables under `modes`, with models such as mean, time ridge, autoregressive, real neuro, shuffled neuro, random Gaussian, autoregressive+neuro, and residualized autoregressive+neuro.
- Per-video diagnostics: event/spike and event-conditioned `.per_video.csv` files.
- Event/spike masks: event-conditioned `.event_masks.csv` plus local target/diagnostic CSVs.
- Lag/window/context settings: temporal fairness/context CSVs use fields such as `offset_seconds`, `window_name`, `window_seconds`, `window_start_seconds`, `window_end_seconds`, `context_type`, and `aggregation`.
- Evaluation unit: manifest rows are 1 Hz time rows per video; grouped leave-video-out outputs aggregate over video groups/folds. Event diagnostics also report per-video summaries.
- Train/test split policy visible in files: official 70/30 within video, blocked temporal gap with 60% train, 10% gap, 30% test, and leave-video-out groups.
- Leakage-control clues:
  - PCA modes fit PCA on train rows only inside each split/fold and transform test rows with the train-fitted basis.
  - Temporal dynamics are computed within video only.
  - Blocked temporal gap excludes an intermediate gap from train/test scoring.
  - Event classifications use train-prevalence thresholding inside each split.
  - Temporal fairness/context leakage audits pass causal-window checks for final claims.

Unclear from current artifacts:

- Whether a future model loader should consume raw per-video NPZs directly or a pre-materialized row-level tensor table.
- Whether video `83` should be repaired before freezing or frozen with the current resampling policy.
- Whether subcortical features are intentionally excluded from future training or only missing from the current 124-video cache.

## 5. Tensor/Data Shape Inventory

### NPZ / NPY

Representative inspected NPZ shapes:

| Path | Keys | Shapes / dtypes | Notes |
| --- | --- | --- | --- |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/0/tribe_raw_output.npz` | `predictions`, `modality_missing_flags`, `segment_retention_features` | `predictions (58, 20484) float64`; flags `(3,) float32`; retention `(5,) float32` | Exact match to video 0 manifest rows |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/52/tribe_raw_output.npz` | same | `predictions (11, 20484) float64` | Short video, exact-style cached cortical archive |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/83/tribe_raw_output.npz` | same | `predictions (263, 20484) float64` | Manifest report says expected rows are 143; benchmark resamples |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_smoke/cortical_plus_subcortical/video_52/tribe_raw_output.npz` | `predictions`, `subcortical_predictions`, flags, retention | `predictions (11, 20484) float64`; `subcortical_predictions (11, 8808) float32` | Smoke-only subcortical reference |
| `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache_mlx/52/tribe_raw_output.npz` | `predictions`, flags, retention | `predictions (11, 20484) float64` | MLX cache parity reference |
| `backend/uploads/tribe_text_smoke_test/tribe_raw_output.npz` | `predictions`, `subcortical_predictions` | `predictions (2, 20484) float64`; `subcortical_predictions (2, 8808) float32` | Text smoke only |

Aggregate main VEATIC cache:

- Count: 124 `tribe_raw_output.npz` files under `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`.
- All 124 inspected archives have keys `predictions`, `modality_missing_flags`, and `segment_retention_features`.
- All main-cache `predictions` arrays have 20,484 feature columns.
- Main cache does not include `subcortical_predictions`.
- Unique row-count signatures: 73, reflecting variable video durations.

### JSONL / JSON

`benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`:

- Lines/rows: 10,357.
- Schema: `veatic_temporal_window_v1`.
- Important columns/keys per row: `dataset`, `stimulus_id`, `video_id`, `frame_index`, `time_start_seconds`, `time_end_seconds`, `sampling_frequency_hz`, `split`, `splits`, `targets`, `media_path`, `source_annotation`.
- Targets: `targets.valence`, `targets.arousal`, both scalar floats.
- Sampling frequency field: `1.0`.
- Source annotations point to `/Volumes/onn. Drive/Neural Bridge/datasets/veatic/rating_averaged/*_{valence,arousal}.csv`.

`benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json`:

- `valid_videos`: 124.
- `rejected_videos`: 0.
- `rows`: 10,357.
- `sample_hz`: 1.0.
- `cache_dir`: `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`.
- `complete_video_ids`: 124.
- Cache validation: no missing raw outputs, no incomplete status entries, no nonfinite prediction videos.
- Feature alignment counts: 123 exact, 1 linear-resampled by benchmark.
- Shape mismatch: video `83`, cached predictions `(263, 20484)`, manifest rows `143`.
- Manifest hash: `9b33000cf297344e6c3470d52fe1960fa66ff469429dd3f6507e1704e7fddad7`.

124-video benchmark JSONs:

- Schema: `veatic_neuro_temporal_dynamics_benchmark_v2`.
- `accepted_videos`: 124.
- `accepted_rows`: 10,357.
- `backend_policy`: PCA backend `mps_gram`; ridge backend `cpu_pinv`; seed 17.
- `target_contract.base_targets`: `valence`, `arousal`.
- `target_contract.derived_targets`: 24 derived target types.
- Split modes: `mode_a_official_veatic_70_30`, `mode_b_blocked_temporal_gap`, `mode_c_leave_video_out`.
- Official split in current 124 output: 7,277 train rows, 3,080 test rows, 124 train videos, 124 test videos.
- Blocked split in current 124 output: 6,237 train rows, 1,040 gap rows, 3,080 test rows, 124 train videos, 124 test videos.

### CSV

Selected inspected CSV schemas:

| Path | Rows | Key columns |
| --- | ---: | --- |
| `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.event_masks.csv` | 9,680 | `split`, `mask`, `target`, `task_type`, `feature_mode`, `model`, `threshold`, `event_count`, `n`, `pr_auc`, `recall`, `top_10pct_recall`, `mae` |
| `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.local_targets.csv` | 224 | `split`, `feature_mode`, `target`, `threshold`, `window_seconds`, `offset_seconds`, real/ar/random/shuffled metric groups |
| `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.diagnostics.csv` | 152 | `split`, `feature_mode`, `target`, `threshold`, `gap_rows`, `offset_seconds`, real/ar/random/shuffled/zero metric groups |
| `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.per_video.csv` | 496 | `video_id`, `split`, `target`, `threshold`, `event_count`, `n`, `real_pr_auc`, `ar_pr_auc`, control deltas |
| `outputs/veatic_124_temporal_fairness_20260616_1509/balanced_event_stable_results.csv` | 144,000 | `arm`, `feature_mode`, `fold`, `model`, `target`, `threshold`, `seed`, `n_pos`, `n_neg`, `pr_auc`, `roc_auc`, `recall` |
| `outputs/veatic_124_temporal_fairness_20260616_1509/selected_offsets_by_fold.csv` | 160 | `feature_mode`, `fold`, `held_out_video_ids`, `inner_validation_video_ids`, `selected_offset_seconds`, `selection_policy`, `selection_variant`, `target`, `threshold` |
| `outputs/veatic_124_temporal_fairness_20260616_1509/leakage_audit.csv` | 9 | `check`, `final_claim_safe`, `status` |
| `outputs/veatic_124_temporal_context_v2_20260616_1557/reused_artifacts_manifest.csv` | 8 | `path`, `bytes`, `mtime`, `sha256`, `contains`, `reuse_reason`, `leakage_safe_reason`, `exists` |
| `outputs/veatic_124_temporal_context_v2_20260616_1557/context_window_ablation_results.csv` | 740 | `arm`, `context_type`, `feature_mode`, `fold`, `held_out_video_ids`, `window_name`, `window_seconds`, `representation`, real/ar/control metrics |
| `outputs/veatic_124_temporal_context_v2_20260616_1557/representation_ablation_results.csv` | 740 | Same schema as context window ablation, organized by representation |
| `outputs/veatic_124_temporal_context_v2_20260616_1557/real_vs_control_context_specificity.csv` | 3,600 | `control`, `feature_mode`, `fold`, `representation`, `target`, `threshold`, `window_name`, `window_specific_gain` |
| `outputs/veatic_124_temporal_context_v2_20260616_1557/leakage_audit.csv` | 8 | `check`, `final_claim_safe`, `status` |

## 6. Data Contract Needed Before Future Model Work

### A. Candidate Feature Inputs

| Name | Path | Shape / dtype | Meaning | Axes | Alignment notes | Usable later |
| --- | --- | --- | --- | --- | --- | --- |
| Raw cortical TRIBE predictions | `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/<video_id>/tribe_raw_output.npz:predictions` | `(timepoints, 20484)`, float64 on disk, cast to float32 in benchmark | Per-video cortical TRIBE/V-JEPA-style response features | Time x feature; video is directory name | Row count expected to match manifest rows, except video 83 currently resampled | Yes, after baseline freeze and video-83 policy |
| Cortical global features | Derived in benchmark from raw cortical | `(manifest_rows, 6)`, float32 | mean, mean abs, std, peak abs, p95 abs, positive fraction | Row x feature | Derived per row from raw; no fitting | Yes |
| Cortical global delta | Derived in benchmark | `(manifest_rows, 36)`, float32 | Global features plus within-video dynamics | Row x feature | Within-video only | Yes |
| Cortical PCA-64 | Derived in benchmark | `(rows, 64)`, float32 | Train-fitted PCA projection of raw cortical | Row x component | PCA basis fit on train rows only per split/fold | Yes |
| Cortical PCA64 delta | Derived in benchmark | `(rows, 384)`, float32 | PCA-64 plus temporal dynamics | Row x feature | PCA train-only; dynamics within-video | Yes, but slower |
| Subcortical predictions | Smoke-only paths inspected | `(timepoints, 8808)`, float32 | Subcortical model outputs | Time x feature | Not present in main 124 cache | Not usable for 124 training unless extracted/frozen separately |
| Modality missing flags | Main cache NPZ | `(3,)`, float32 | Missing text/audio/video indicators from adapter | Feature only per video | Not row-aligned | Diagnostic only |
| Segment retention features | Main cache NPZ | `(5,)`, float32 | Retention/repair metadata from adapter | Feature only per video | Not row-aligned | Diagnostic only |

### B. Candidate Targets

| Name | Path | Shape / dtype | Meaning | Time axis | Task type | Usable later |
| --- | --- | --- | --- | --- | --- | --- |
| Valence | `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl:targets.valence` | 10,357 scalar floats | Human valence annotation at 1 Hz rows | Manifest row order, per video | Regression base; derived classification possible | Yes |
| Arousal | `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl:targets.arousal` | 10,357 scalar floats | Human arousal annotation at 1 Hz rows | Manifest row order, per video | Regression base; event/spike classification | Yes |
| Future state/change/residual targets | Derived by benchmark code | Variable row counts after horizon filtering | Future affect state/change/residuals | Within-video future horizons 1/2/3s | Regression | Yes, if derived identically |
| Event future spike/drop/rise/reversal/peak/recovery | Derived by benchmark code and materialized in event outputs | Variable rows per target/split | Binary event labels based on thresholds/horizons | Within-video only | Classification/mask | Yes, if thresholds and derivation are frozen |

### C. Candidate Masks/Splits

| Name | Path | Shape / columns | Meaning | Policy | Leakage notes |
| --- | --- | --- | --- | --- | --- |
| Official 70/30 | Manifest `splits.official_70_30` | 10,357 scalar strings | First 70% / last 30% frame protocol per video | Train/test within every video | Adjacent temporal leakage risk remains |
| Blocked temporal gap | Manifest `splits.blocked_temporal_gap` | 10,357 scalar strings | 60% train, 10% gap, 30% test | Train/gap/test within every video | Gap reduces adjacent-frame leakage |
| Leave-video-out group | Manifest `splits.leave_video_out_group` | 10,357 group labels | Whole-video grouped folds | Held-out video IDs by fold | Stronger video-generalization test |
| Event masks | `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.event_masks.csv` | 9,680 metric rows by `mask` | All frames, pre-event/event-conditioned masks | Diagnostic subsets | Positive-only masks should not replace main discrimination |
| Temporal context windows | `outputs/veatic_124_temporal_context_v2_20260616_1557/*` | Rows by `window_name`, `window_seconds`, `context_type` | Causal and diagnostic windows | Final claims use causal windows | Leakage audit passes causal checks |

### D. Candidate Baseline Outputs

| Name | Path | Shape / columns | Meaning | Metric fields | Freeze |
| --- | --- | --- | --- | --- | --- |
| 124 cortical global benchmark | `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_global_mpsgram_cpu.json` | Nested JSON | Main 6-feature baseline | MAE/RMSE/Pearson/Spearman and event metrics | Yes |
| 124 PCA benchmarks | `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_pca_64_mpsgram_cpu.json`, `...pca64_delta...json` | Nested JSON | Current stronger feature baselines | Same | Yes |
| 124 event-conditioned retest | `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.*` | CSV/JSON/MD | Event-conditioned diagnostics | PR-AUC, recall/top-k, MAE, F1 | Yes |
| 124 event/spike core retest | `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.*` | CSV/JSON/MD | Spike/core target diagnostics | PR-AUC, F1, recall/top-k, MAE | Yes |
| 89 device consistency audit | `benchmarks/veatic/veatic_89_benchmark_device_consistency_20260616.*` | CSV/JSON/MD | CPU/MPS drift and threshold consistency | prediction/metric/threshold diffs | Yes |
| Temporal fairness outputs | `outputs/veatic_124_temporal_fairness_20260616_1509/*` | CSV/JSON/MD | Offset/window/fairness controls | PR-AUC, ROC-AUC, recall, selected offsets, leakage checks | Yes |
| Temporal context v2 outputs | `outputs/veatic_124_temporal_context_v2_20260616_1557/*` | CSV/JSON/MD | Context/representation ablations | real-vs-control gains, window gains, hashes | Yes |

## 7. Existing Baseline / Frozen-Output Inventory

Current benchmark outputs that should be preserved before any experiment:

- `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- `benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json`
- `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_global_mpsgram_cpu.json`
- `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_global_delta_mpsgram_cpu.json`
- `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_pca_64_mpsgram_cpu.json`
- `benchmarks/veatic/veatic_neuro_benchmark_124video_cortical_pca64_delta_mpsgram_cpu.json`
- Matching `.summary.md` files for the above benchmarks.
- `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.*`
- `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.*`
- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `outputs/veatic_124_temporal_fairness_20260616_1509/*`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/*`
- `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/*/tribe_raw_output.npz`

Important diagnostic outputs:

- `benchmarks/veatic/veatic_89_benchmark_device_consistency_20260616.metric_diff.focus.csv`
- `benchmarks/veatic/veatic_89_benchmark_device_consistency_20260616.prediction_diff.csv`
- `benchmarks/veatic/veatic_89_benchmark_device_consistency_20260616.threshold_diff.csv`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/reused_artifacts_manifest.csv`, which records 8 reused artifacts with hashes and leakage-safe reuse reasons.

## 8. Missing Information / Blockers

Exact missing items before model work:

- Protected baseline snapshot directory and manifest of frozen files.
- Formal loader contract that maps per-video NPZ rows to manifest rows, including dtype normalization and video-83 mismatch policy.
- Decision on whether video `83` should be repaired, excluded, or frozen as current benchmark-resampled behavior.
- Formal list of target derivations to expose to future model code; current derivations live in benchmark code.
- Explicit policy on whether future recursive model work should train on raw 20,484-dimensional cortical features, 6-feature global summaries, PCA features, temporal-dynamics features, or multiple arms.
- Explicit policy on whether subcortical is out of scope for first model work or requires a new frozen 124-video subcortical cache.
- Stable baseline identity: which of global, PCA-64, PCA64-delta, event-conditioned, and temporal-context outputs is the primary baseline for future comparisons.

## 9. Next Operational Step Recommendation

Next step: freeze/copy the current benchmark baseline artifacts into a protected baseline snapshot with hashes.

Do not write model code yet. The data and benchmark outputs are sufficient to create a baseline snapshot, but the training tensor contract is not yet formalized and the current baseline artifacts are not protected from accidental overwrite.
