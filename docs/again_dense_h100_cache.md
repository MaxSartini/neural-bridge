# AGAIN Dense H100 V-JEPA 2.1 / TRIBE v2 Cache

This is the current handoff for the dense AGAIN artifact generated on H100.

## Status

- Dataset: AGAIN cleaned video set, `995` videos.
- Encoder: official V-JEPA 2.1 ViT-G.
- TRIBE stage: TRIBE v2 cache-only postpass over precomputed V-JEPA features.
- Precision: float16.
- Image size: `256px`.
- Row rate: `2Hz`.
- Frame sampling: `2Hz`.
- Result: `995/995` videos completed, `0` failed, `243,575` row-level cortical rows.
- Drive folder: `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`.
- Local pull target: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`.
- Local audit: `reports/again_dense_h100_local_audit_20260625.md`.

This artifact is a data-generation milestone. It is not a benchmark result and must not be described as proving AGAIN generalization until grouped/control evaluations are run.

## What Was Run

The H100 job densely encoded the full 995-video AGAIN set. It did not use PCA, bridge training, spike labels, or benchmark targets during encoding.

The TRIBE v2 postpass was cache-only:

- It consumed V-JEPA 2.1 cache files.
- It did not decode raw videos.
- It did not load or run V-JEPA again.
- It adapted cached video features from `[rows,20,1,1408]` to `[rows,2,1408]`, then to TRIBE input `[1,2,1408,rows]`.
- It wrote row-level cortical predictions as `[rows,20484]`.

## Expected Output Layout

Top-level files:

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

Per-video directory:

```text
per_video/<video_id>/
  tribe_v2_cortical_predictions.npz
  baseline_features_rowlevel.npz
  vjepa_temporal_diagnostics.npz
  rows_aligned.csv
  input_mapping.json
  diagnostics.json
  manifest.json
  status.json
```

Important per-video arrays:

- `cortical_prediction [rows,20484]`
- `time_seconds [rows]`
- `tribe_grouped_video_feature [rows,2,1408]`
- row-level quality signals: luma, motion, black-frame fraction, duplicate-frame fraction
- compact temporal diagnostics
- sample frame indices and sample time seconds

## What It Enables

The bundle is sufficient for local downstream work without re-running V-JEPA or TRIBE:

- train-only PCA widths over TRIBE cortical predictions
- AR + cortical bridge models
- grouped-video and blocked temporal validation
- timestamp/video-time controls
- shuffled/random controls
- quality/motion/luma baselines
- black-screen and duplicate-frame filtering
- lead/lag and future-delta experiments after labels are aligned correctly

## Local Audit Notes

The 2026-06-25 local audit found the internal-SSD copy complete and schema-valid:

- `995` per-video output folders
- `995` final `status.json` successes
- `0` failed-video log lines
- `243,575` row-index rows matching the global metadata and per-video row counts
- `0` missing required per-video output files
- `0` partial/temp transfer files
- `0` remaining per-video `traceback.txt` files after stale success tracebacks were removed
- sampled arrays match the expected `[rows,20484]`, `[rows,2,1408]`, and compact temporal-diagnostic shapes

Timing nuance: `864` videos start at `0.0s`; `131` videos start at `0.5s`. The `0.5s` starts are not shape failures and should not be papered over by forcing synthetic zero rows. Downstream code must use saved `time_seconds` and the canonical `row_index.parquet` values rather than assuming every video starts at zero.

Quality nuance: quality flags are present. The audit found `4,816` quality-excluded rows across `966` videos, driven by duplicate-frame flags, and `0` black-frame-flagged videos. Keep those flags for train/test-aware filtering or weighting.

Cache repair: `113` stale `traceback.txt` files from successful per-video folders were removed after final `status.json`, global manifests, row counts, and sampled arrays passed. A local non-git repair manifest was written at `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/_run/cache_repair_20260625_stale_success_tracebacks.json`.

## Current Guardrails

- Do not treat 2Hz features as 2Hz supervised evidence until labels/rows are aligned for the target experiment.
- Do not drop rows silently because of black frames; preserve quality flags and decide filtering inside each train/test protocol.
- Do not re-encode videos unless the local/Drive artifact fails a manifest/schema audit.
- Do not copy the Drive bundle into git.
