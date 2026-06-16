# Neuro Bridge Implemented State Inventory

Generated: 2026-06-13, Africa/Johannesburg  
Repository: `/Users/maxsartini/Neural Bridge`

This is an inventory of what is currently implemented and what has already run. It does not propose a new architecture.

## 1. TRIBE v2 Extraction Path Currently Used

The active VEATIC extraction path is:

```text
backend/scripts/run_veatic_tribe_cache.py
  -> app.services.tribe_adapter.TribeAdapter
  -> backend="tribe_mlx"
  -> MLX / Apple Silicon TRIBE path when Config.TRIBE_MLX_ENABLED=True
  -> cortical predictions written to tribe_raw_output.npz
```

Current runtime contract used by the active VEATIC 50-video extraction:

```json
{
  "run_mode": "cortical_fast_default",
  "backend": "tribe_mlx",
  "subcortical_enabled": false,
  "subcortical_policy": "disabled_default_cortical_fast_path",
  "device": "mps",
  "video_dtype": "float16",
  "video_num_frames": 64,
  "mps_memory_fraction": 0.35,
  "mps_high_watermark": "0.45",
  "mps_low_watermark": "0.25",
  "attention": "exact_mps_query_chunked_sdpa",
  "attention_query_chunk_size": "128"
}
```

Relevant implementation files:

- `/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_tribe_cache.py`
- `/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_neuro_benchmark.py`
- `/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_gated_pipeline.py`
- `/Users/maxsartini/Neural Bridge/backend/app/services/tribe_adapter.py`
- `/Users/maxsartini/Neural Bridge/backend/app/config.py`

The extractor sets external-cache paths before inference:

```text
HF_HOME=/Volumes/onn. Drive/Neural Bridge/cache/huggingface
TMPDIR=/Volumes/onn. Drive/Neural Bridge/tmp
TRIBE_CACHE_DIR=/Volumes/onn. Drive/Neural Bridge/cache/tribev2
TRIBE_VIDEO_WINDOW_CACHE_DIR=/Volumes/onn. Drive/Neural Bridge/cache/tribev2/video_windows
```

## 2. VEATIC Cache Status And Cache File Locations

VEATIC manifest/report files:

```text
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_manifest_1hz.jsonl
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_manifest_1hz.report.json
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_readiness_report.md
```

TRIBE cache root:

```text
/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache
```

Per-video cache layout:

```text
/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/<video_id>/cache_status.json
/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache/<video_id>/tribe_raw_output.npz
```

Current active 50-video extraction status at inspection:

```text
target videos: 50
complete: 39
incomplete/missing: 11
failed: 0
contract mismatches: 0
active extractor PID: 10975
```

The active extractor command:

```bash
backend/scripts/run_veatic_tribe_cache.py --run-mode cortical_fast_default --limit 50
```

Latest incomplete video at inspection:

```text
video_id: 61
duration_seconds: 61.04
state: incomplete
```

Remaining missing/incomplete target-video IDs at inspection:

```text
61, 108, 109, 45, 58, 64, 1, 97, 70, 10, 7
```

Completed target-video IDs at inspection:

```text
52, 20, 95, 12, 19, 81, 63, 6, 82, 54,
84, 21, 17, 8, 62, 2, 14, 92, 55, 28,
122, 123, 60, 53, 75, 68, 100, 101, 79, 66,
23, 36, 47, 0, 78, 4, 44, 65, 71
```

Recent completed extraction timings:

```text
video 78: 677.900 s, predictions shape [58, 20484]
video 4: 678.884 s, predictions shape [59, 20484]
video 44: 684.204 s, predictions shape [59, 20484]
video 65: 679.264 s, predictions shape [59, 20484]
video 71: 703.318 s, predictions shape [60, 20484]
```

## 3. Current Run Modes

Implemented in:

```text
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_tribe_cache.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_neuro_benchmark.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_openlav_tribe_cache.py
```

Available run modes:

```python
RUN_MODES = ("cortical_fast_default", "full_research", "subcortical_ablation")
```

### cortical_fast_default

Current default for VEATIC and OpenLAV extraction/benchmark development.

Behavior:

```text
cortical TRIBE only
subcortical disabled
uses MLX/MPS
uses 64 video frames
uses float16 V-JEPA path
```

In `run_veatic_tribe_cache.py`:

```python
Config.TRIBE_ENABLE_SUBCORTICAL = args.run_mode in {"full_research", "subcortical_ablation"}
```

Therefore `cortical_fast_default` sets:

```text
TRIBE_ENABLE_SUBCORTICAL=False
```

### full_research

Explicit research mode.

Behavior:

```text
cortical + subcortical
subcortical enabled
not default
used only when explicitly requested
```

### subcortical_ablation

Explicit ablation mode.

Behavior:

```text
subcortical path enabled
supports subcortical-only / cortical+subcortical comparisons in benchmark code
not default
```

## 4. Current Feature Columns Produced For cortical_fast_default

Implemented in:

```text
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_neuro_benchmark.py
```

Function:

```python
def cortical_global_features(cortical: np.ndarray) -> np.ndarray:
    abs_values = np.abs(cortical)
    return np.stack(
        [
            cortical.mean(axis=1),
            abs_values.mean(axis=1),
            cortical.std(axis=1),
            abs_values.max(axis=1),
            np.percentile(abs_values, 95, axis=1),
            (cortical > 0).mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)
```

Feature names:

```text
cortical_mean
cortical_mean_abs
cortical_std
cortical_peak_abs
cortical_p95_abs
cortical_positive_fraction
```

## 5. Is cortical_global Still Only 6 Features?

Yes.

The current `cortical_fast_default` benchmark report records:

```json
{
  "feature_sets": {
    "cortical_global": 6
  }
}
```

For `cortical_fast_default`, the selected feature sets are:

```python
FEATURE_SETS_BY_RUN_MODE = {
    "cortical_fast_default": ("cortical_global",),
    "full_research": ("cortical_global", "subcortical_roi", "combined"),
    "subcortical_ablation": ("cortical_global", "subcortical_roi", "combined"),
}
```

So the current cortical-only default is exactly six cortical global summary features.

## 6. Current Benchmark Targets Implemented

Implemented target base axes:

```text
valence
arousal
```

Implemented derived targets:

```text
raw
delta_prev_1s
residual_after_persistence
future_state_p1s
future_state_p2s
future_state_p3s
future_change_p1s
future_change_p2s
future_change_p3s
residual_future_p1s_persistence
residual_future_p2s_persistence
residual_future_p3s_persistence
residual_future_p1s_rolling3
residual_future_p2s_rolling3
residual_future_p3s_rolling3
residual_future_p1s_time_only
residual_future_p2s_time_only
residual_future_p3s_time_only
event_future_spike_1_3s
event_future_drop_1_3s
event_future_rise_1_3s
event_trend_reversal_1_3s
event_peak_onset_1_3s
event_recovery_onset_1_3s
```

Current event threshold:

```text
0.05
```

The current benchmark therefore includes:

- Current state prediction.
- Previous-step delta prediction.
- Persistence residual prediction.
- Future state at +1s, +2s, +3s.
- Future change at +1s, +2s, +3s.
- Residual future prediction against persistence, rolling baseline, and time-only baseline.
- Event classification for spike/drop/rise/reversal/peak/recovery.

## 7. Current Train/Test Splits Implemented

Implemented VEATIC modes:

```text
mode_a_official_veatic_70_30
mode_b_blocked_temporal_gap
mode_c_leave_video_out
```

For the latest 20-video cortical-fast run:

```text
mode_a_official_veatic_70_30:
  train_rows: 507
  test_rows: 210
  gap_rows: 0
  train_videos: 20
  test_videos: 20

mode_b_blocked_temporal_gap:
  train_rows: 434
  test_rows: 210
  gap_rows: 73
  train_videos: 20
  test_videos: 20

mode_c_leave_video_out:
  folds: 20
  one held-out video per fold
```

## 8. Current Controls Implemented

Implemented prediction/control conditions include:

```text
mean_train
time_ridge
autoregressive
persistence_previous_known
cortical_global
shuffled_cortical_global
random_gaussian_cortical_global
autoregressive_plus_cortical_global
autoregressive_plus_shuffled_cortical_global
autoregressive_plus_random_gaussian_cortical_global
residualized_autoregressive_plus_cortical_global
residualized_autoregressive_plus_shuffled_cortical_global
residualized_autoregressive_plus_random_gaussian_cortical_global
```

When `combined` exists in full/subcortical modes, the benchmark code also has a shuffled-label condition:

```text
shuffled_labels_autoregressive_plus_combined
```

For `cortical_fast_default`, `combined` is not selected as a separate feature set because the run mode uses only:

```text
cortical_global
```

Lead/lag diagnostics are implemented:

```text
lead_lag_analysis()
offsets: -3, -2, -1, 0, +1, +2, +3 seconds
targets: future_change p1/p2/p3 for valence and arousal
```

Permutation/feature-importance diagnostics are implemented for diagnostic targets:

```text
feature_importance
permutation_importance_for_split()
mode_a_official_veatic_70_30
mode_b_blocked_temporal_gap
```

The 50-video gated pipeline checks real cortical against:

```text
autoregressive
autoregressive_plus_shuffled_cortical_global
autoregressive_plus_random_gaussian_cortical_global
```

## 9. Latest 20-Video VEATIC Result Files And Key Metrics

Latest cortical-only 20-video result files:

```text
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_20video_cortical_fast_default.json
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_20video_cortical_fast_default.summary.md
```

Run metadata:

```text
schema_version: veatic_neuro_temporal_dynamics_benchmark_v2
run_mode: cortical_fast_default
subcortical_enabled: false
accepted_videos: 20
accepted_rows: 717
feature_sets: cortical_global = 6
```

### 20-Video Arousal Dynamics: Official VEATIC 70/30 Split

Future arousal change +1s:

```text
autoregressive: MAE 0.0187, Pearson 0.1275, Spearman 0.1781
autoregressive + cortical: MAE 0.0178, Pearson 0.0443, Spearman 0.1775
autoregressive + shuffled cortical: MAE 0.0189, Pearson 0.1265, Spearman 0.1679
autoregressive + random Gaussian: MAE 0.0189, Pearson 0.1212, Spearman 0.1738
```

Future arousal change +2s:

```text
autoregressive: MAE 0.0442, Pearson 0.0492, Spearman 0.1007
autoregressive + cortical: MAE 0.0381, Pearson 0.0054, Spearman 0.1242
autoregressive + shuffled cortical: MAE 0.0432, Pearson 0.1080, Spearman 0.1404
autoregressive + random Gaussian: MAE 0.0443, Pearson 0.0394, Spearman 0.1152
```

Future arousal spike 1-3s:

```text
autoregressive: F1 0.2286, accuracy 0.7158
autoregressive + cortical: F1 0.3143, accuracy 0.7474
autoregressive + shuffled cortical: F1 0.2000, accuracy 0.7053
autoregressive + random Gaussian: F1 0.2286, accuracy 0.7158
```

Future arousal drop 1-3s:

```text
autoregressive: F1 0.2222, accuracy 0.8158
autoregressive + cortical: F1 0.2667, accuracy 0.8263
autoregressive + shuffled cortical: F1 0.1333, accuracy 0.7947
autoregressive + random Gaussian: F1 0.2222, accuracy 0.8158
```

### 20-Video Arousal Dynamics: Blocked Temporal Gap Split

Future arousal change +1s:

```text
autoregressive: MAE 0.0191, Pearson 0.1368, Spearman 0.2122
autoregressive + cortical: MAE 0.0219, Pearson 0.0562, Spearman 0.1509
autoregressive + shuffled cortical: MAE 0.0277, Pearson 0.1826, Spearman 0.2110
autoregressive + random Gaussian: MAE 0.0201, Pearson 0.1165, Spearman 0.1782
```

Future arousal spike 1-3s:

```text
autoregressive: F1 0.2162, accuracy 0.6947
autoregressive + cortical: F1 0.2973, accuracy 0.7263
autoregressive + shuffled cortical: F1 0.2162, accuracy 0.6947
autoregressive + random Gaussian: F1 0.2162, accuracy 0.6947
```

Future arousal drop 1-3s:

```text
autoregressive: F1 0.1277, accuracy 0.7842
autoregressive + cortical: F1 0.1277, accuracy 0.7842
autoregressive + shuffled cortical: F1 0.1277, accuracy 0.7842
autoregressive + random Gaussian: F1 0.1277, accuracy 0.7842
```

### 20-Video Arousal Dynamics: Leave-Video-Out Split

Future arousal change +1s:

```text
autoregressive: MAE 0.0203, Pearson 0.2274, Spearman 0.2408
autoregressive + cortical: MAE 0.0207, Pearson 0.2271, Spearman 0.2386
autoregressive + shuffled cortical: MAE 0.0205, Pearson 0.2206, Spearman 0.2396
autoregressive + random Gaussian: MAE 0.0204, Pearson 0.2205, Spearman 0.2386
```

Future arousal spike 1-3s:

```text
autoregressive: F1 0.4649, accuracy 0.7481
autoregressive + cortical: F1 0.4159, accuracy 0.7423
autoregressive + shuffled cortical: F1 0.4565, accuracy 0.7440
autoregressive + random Gaussian: F1 0.4077, accuracy 0.7346
```

Interpretation of the existing 20-video run:

```text
The cortical signal improved arousal spike/drop detection in the official and blocked temporal splits.
It did not improve leave-video-out arousal spike detection.
This is useful diagnostic evidence, not proof of robust generalisation.
```

### 20-Video Lead/Lag Diagnostics

For arousal future-change targets, strongest inspected cortical feature was usually:

```text
cortical_p95_abs
```

Best offsets observed in the 20-video run:

```text
arousal future_change +1s:
  best offset: +2s
  best feature: cortical_p95_abs
  best Pearson: 0.1878

arousal future_change +2s:
  best offset: +2s
  best feature: cortical_p95_abs
  best Pearson: 0.2325

arousal future_change +3s:
  best offset: +2s
  best feature: cortical_p95_abs
  best Pearson: 0.2587
```

## 10. Has A 50-Video Run Already Been Executed?

Extraction is currently in progress. The 50-video benchmark has not completed yet.

At inspection, these files did not exist:

```text
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_50video_cortical_fast_default.json
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_gated_pipeline_50video_cortical_fast_default_gate.json
```

Current state:

```text
50-video extraction: running
50-video benchmark scoring: not yet run
50-video gate report: not yet produced
```

## 11. Exact Commands Needed To Run The Next 50-Video Cortical-Only Extraction And Benchmark

### Resume or run 50-video cortical-only extraction

```bash
cd /Users/maxsartini/Neural Bridge

backend/.venv/bin/python backend/scripts/run_veatic_tribe_cache.py \
  --run-mode cortical_fast_default \
  --limit 50
```

This command is resumable. It skips complete valid caches unless `--force` is passed.

### Run the 50-video cortical-only benchmark after extraction is complete

```bash
cd /Users/maxsartini/Neural Bridge

backend/.venv/bin/python backend/scripts/run_veatic_neuro_benchmark.py \
  --run-mode cortical_fast_default \
  --cache-dir "/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache" \
  --output benchmarks/veatic/veatic_neuro_benchmark_50video_cortical_fast_default.json
```

### Run the gated extraction + benchmark + report workflow

```bash
cd /Users/maxsartini/Neural Bridge

backend/.venv/bin/python backend/scripts/run_veatic_gated_pipeline.py \
  --target 50 \
  --run-mode cortical_fast_default \
  --cache-dir "/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache" \
  --output benchmarks/veatic/veatic_neuro_benchmark_50video_cortical_fast_default.json \
  --gate-output benchmarks/veatic/veatic_gated_pipeline_50video_cortical_fast_default_gate.json
```

### If extraction is already complete and only scoring/gating is needed

```bash
cd /Users/maxsartini/Neural Bridge

backend/.venv/bin/python backend/scripts/run_veatic_gated_pipeline.py \
  --target 50 \
  --run-mode cortical_fast_default \
  --cache-dir "/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache" \
  --output benchmarks/veatic/veatic_neuro_benchmark_50video_cortical_fast_default.json \
  --gate-output benchmarks/veatic/veatic_gated_pipeline_50video_cortical_fast_default_gate.json \
  --skip-extract
```

## 12. Existing TODOs / Scripts For Reports, Cache Audit, Feature Engineering, Lag Scan

### Existing VEATIC benchmark and gate scripts

```text
/Users/maxsartini/Neural Bridge/backend/scripts/build_veatic_manifest.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_annotation_baseline.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_tribe_cache.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_neuro_benchmark.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_gated_pipeline.py
```

### Existing OpenLAV benchmark/audit scripts

```text
/Users/maxsartini/Neural Bridge/backend/scripts/audit_openlav_dataset.py
/Users/maxsartini/Neural Bridge/backend/scripts/audit_openlav_post50.py
/Users/maxsartini/Neural Bridge/backend/scripts/openlav_preflight_audit.py
/Users/maxsartini/Neural Bridge/backend/scripts/openlav_scientific_readiness_gate.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_openlav_benchmark.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_openlav_residual_neuro_benchmark.py
/Users/maxsartini/Neural Bridge/backend/scripts/run_openlav_tribe_cache.py
/Users/maxsartini/Neural Bridge/backend/scripts/summarize_openlav_benchmark.py
```

### Existing report/memory/handoff scripts

```text
/Users/maxsartini/Neural Bridge/backend/scripts/build_neuro_translation_report.py
/Users/maxsartini/Neural Bridge/backend/scripts/create_project_handoff.py
```

Existing handoff/memory files:

```text
/Users/maxsartini/Neural Bridge/docs/handoffs/20260613_133534_veatic-50-run-and-automation-handoff.md
/Users/maxsartini/Neural Bridge/docs/PROJECT_MEMORY.md
```

### Existing docs relevant to current benchmark state

```text
/Users/maxsartini/Neural Bridge/docs/benchmarking_accuracy.md
/Users/maxsartini/Neural Bridge/docs/neuro_prior_integration.md
/Users/maxsartini/Neural Bridge/docs/project_state_2026-06-12.md
/Users/maxsartini/Neural Bridge/docs/tribe_neural_bridge_architecture_review.md
/Users/maxsartini/Neural Bridge/docs/automation_workflows.md
```

### Existing VEATIC result/report files

```text
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_5video_neuro_report.md
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_5video.json
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_20video_temporal_dynamics.json
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_20video_temporal_dynamics.summary.md
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_20video_cortical_fast_default.json
/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_neuro_benchmark_20video_cortical_fast_default.summary.md
```

## What Exists vs What Remains Unrun

Already implemented:

- MLX/MPS cortical TRIBE extraction for VEATIC.
- Explicit run modes: `cortical_fast_default`, `full_research`, `subcortical_ablation`.
- Cortical-only default feature contract.
- Six-feature `cortical_global` feature set.
- VEATIC temporal targets for raw, future, delta, residual, and event prediction.
- Official VEATIC split.
- Blocked temporal gap split.
- Leave-video-out split.
- Shuffled cortical control.
- Random Gaussian control.
- Shuffled-label condition when `combined` is selected.
- Lead/lag diagnostics.
- Feature importance and permutation diagnostics.
- 5-video and 20-video VEATIC result artifacts.
- 50-video gated pipeline script.
- Handoff/project-memory script.

Already run:

- VEATIC manifest/readiness generation.
- VEATIC 5-video benchmark.
- VEATIC 20-video full temporal-dynamics benchmark.
- VEATIC 20-video `cortical_fast_default` benchmark.
- VEATIC 50-video cortical extraction is running and partially complete.

Not yet complete at inspection:

- VEATIC 50-video extraction.
- VEATIC 50-video benchmark scoring.
- VEATIC 50-video gated report.
- Any 100-video or 188-video VEATIC run under the current cortical-fast contract.

