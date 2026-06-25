# AGAIN Dense H100 TRIBE Bundle Local Audit

Date: 2026-06-25

Scope: local copy of the dense H100 TRIBE v2 postpass bundle at:

```text
.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/
```

This audit only inspected the local postpass outputs. It did not run PCA, train a bridge model, benchmark labels, decode raw videos, or re-run V-JEPA/TRIBE.

## Result

- Local bundle size: about `39G`.
- Expected videos: `995`.
- Per-video output directories found: `995`.
- Final per-video statuses: `995` success, `0` failed.
- Global manifest lines: `995`.
- Failed-video logs: `0` lines at both top-level and `_run/`.
- Total row-level cortical predictions: `243,575`.
- Row index rows: `243,575`.
- Sum of `video_metadata.csv` row counts: `243,575`.
- Missing required per-video output files: `0`.
- Partial/temp transfer files found: `0`.

The local copy is complete enough to use as the dense AGAIN downstream substrate.

## Global Metadata

The top-level and `_run/` metadata agree:

- dataset: `AGAIN_cleaned`
- precision: `float16`
- row rate: `2.0Hz`
- cache-only: `true`
- forbid video decode: `true`
- forbid V-JEPA: `true`
- PCA run: `false`
- bridge training run: `false`
- benchmark run: `false`

Output size accounting from `summary_report.json`:

- total output folder size: `41,568,957,890` bytes
- cortical output size: `11,811,812,245` bytes
- compact temporal diagnostics size: `27,760,208,130` bytes
- average runtime per video during TRIBE postpass: `2.686s`
- average runtime per row during TRIBE postpass: `0.01097s`

## Required Files

Top-level files present:

- `global_run_metadata.json`
- `summary_report.json`
- `tribe_v2_postpass_manifest.jsonl`
- `failed_videos.jsonl`
- `video_metadata.csv`
- `row_index.parquet`
- `row_index.csv`
- `splits_by_video.json`
- `splits_duration_balanced.json`
- `splits_quality_filtered.json`
- `output_schema.json`
- `README_OUTPUT_SCHEMA.md`
- `BASELINE_READINESS.md`
- `labels_aligned_2hz.README.md`

`labels_aligned_2hz.parquet` is not present. That is expected for this stage: labels still need a separate boundary-aligned 2Hz manifest/alignment pass before supervised 2Hz claims.

Every successful per-video directory has:

- `tribe_v2_cortical_predictions.npz`
- `baseline_features_rowlevel.npz`
- `vjepa_temporal_diagnostics.npz`
- `rows_aligned.csv`
- `input_mapping.json`
- `diagnostics.json`
- `manifest.json`
- `status.json`

## Sampled Array Checks

Sampled shortest, median, and longest row-count videos all matched the expected schema.

Observed shapes:

- `cortical_prediction`: `[rows,20484]`, `float16`
- `time_seconds`: `[rows]`, `float32`
- `tribe_grouped_video_feature`: `[rows,2,1408]`, `float16`
- `sample_frame_indices`: `[rows,64]`, `int32`
- `sample_time_seconds`: `[rows,64]`, `float32`
- `temporal_std_global`: `[rows]`, `float32`
- `temporal_std_by_state`: `[rows,20]`, `float16`
- `temporal_std_by_state_token`: `[rows,20,32]`, `float16`
- `temporal_mean_by_state_feature`: `[rows,20,1408]`, `float16`
- `temporal_std_by_state_feature`: `[rows,20,1408]`, `float16`

Sampled `cortical_prediction`, adapter features, time arrays, luma arrays, and sample-time arrays had `0` NaNs. Sampled diagnostics reported `0` output NaNs and `0` output infs.

## Row And Timing Checks

The global `row_index.parquet` passed the row-level checks:

- row count: `243,575`
- temporal semantics: all `causal_trailing_clip`
- future-window violations: `0`
- clip-start-after-clip-end violations: `0`
- per-video row-count mismatches against `video_metadata.csv`: `0`
- videos with non-0.5s row steps: `0`
- `row_index.csv` rows: `243,575`
- `row_index.csv` vs `row_index.parquet` row mismatch: `0`

First timestamp distribution:

- `864` videos start at `0.0s`
- `131` videos start at `0.5s`

The `0.5s` first-row case is a timing nuance, not an output-shape failure. These rows should remain exactly as encoded. Downstream label alignment must use the saved `time_seconds` and canonical `row_index.parquet` values rather than assuming all videos start at `0.0s`.

## Quality Flags

Quality metadata is present and should be used in downstream controls/filtering:

- videos with any quality exclusion flag: `966`
- total quality-excluded rows: `4,816`
- max per-video quality exclusion fraction: `0.3959`
- videos with black-frame flags: `0`
- videos with duplicate-frame flags: `966`

Do not silently drop these rows before splitting. Filtering or weighting should happen inside each train/test protocol so quality controls remain honest.

## Cache Repair

The initial local audit found `113` stale `traceback.txt` files in successful per-video folders. Sampled traces showed an earlier validator failure:

```text
StageError: First timestamp is not approximately 0.0: 0.5
```

The final `status.json` files for those folders were `success` with `error: null`, row alignment passed, NaN/inf checks passed, and outputs written. The stale traceback files were removed from per-video folders after validation. A local non-git repair manifest was written at:

```text
.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/_run/cache_repair_20260625_stale_success_tracebacks.json
```

Post-repair checks found `0` remaining per-video `traceback.txt` files.

## Bottom Line

The local dense H100 TRIBE postpass bundle is complete and schema-valid as a cache/data substrate. It is sufficient for the next local dense AGAIN work:

- train-only PCA over cortical predictions
- AR + cortical bridge training
- quality/motion/luma baselines
- timestamp/video-time controls
- shuffled/random controls
- grouped-video folds
- blocked temporal checks

It is not itself a benchmark result. No PCA, bridge training, spike/delta benchmark, or promotion gate has been run on this local audit.
