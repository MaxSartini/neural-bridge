# AGAIN Dense H100 V-JEPA 2.1 / TRIBE v2 Cache

This is the handoff for the dense AGAIN artifact generated on H100 and the substrate for the current scaled Neural Bridge evidence.

## Status

- Dataset: AGAIN cleaned video set, `995` videos.
- Encoder: official V-JEPA 2.1 ViT-G.
- TRIBE stage: TRIBE v2 cache-only postpass over precomputed V-JEPA features.
- Precision: float16.
- Image size: `256px`.
- Row rate: `2Hz`.
- Frame sampling: `2Hz`.
- Result: `995/995` videos completed, `0` failed, `243,575` video feature rows generated from video by upstream models trained on brain cortical response data.
- Drive folder: `NeuralBridge_H100_AGAIN_tribe_v2_postpass_float16_256_2hz`.
- Local pull target: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`.
- Local audit: `reports/again_dense_h100_local_audit_20260625.md`.

This artifact began as a data-generation milestone. The later Phase 5/5.5 evaluations now make AGAIN the scaled confirmation/current main result for controlled future arousal event-ranking. Do not describe the cache alone as proof; describe the cache plus the downstream eval-mode correction, frozen-AR residual design, blocked washout-gap confirmation, and updated grouped-video compatibility as the current evidence chain.

Raw predicted cortical/fMRI features alone fail badly on AGAIN. On the original Phase 3 target `arousal_spike_rows_2_6_train_q90`, blocked `raw_cortical_only` PR-AUC was `0.124315` versus AR-only `0.203622`; direct `AR_plus_raw_cortical` was `0.167731`, below AR. The cache becomes claim-bearing only after the Neural Bridge pipeline turns those predicted cortical/fMRI response representations generated from video by upstream models trained on brain cortical response data into fold-safe, AR-controlled future event-ranking evidence.

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

- train-only PCA widths over TRIBE predicted cortical/fMRI responses
- AR + predicted cortical/fMRI response bridge models
- grouped-video and blocked temporal validation
- timestamp/video-time controls
- shuffled/random controls
- quality/motion/luma baselines
- black-screen and duplicate-frame filtering
- lead/lag and future-delta experiments after labels are aligned correctly

## Dense 2Hz Label And Baseline Status

The true 2Hz supervised manifest now exists locally at:

```text
.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/labels_aligned_2hz.parquet
```

It uses saved dense-cache `time_seconds` directly, preserves the `0.0s`/`0.5s` first-row timing nuance, and does not collapse labels to 1Hz. Coverage from the first build:

- dense rows: `243,575`
- labeled rows: `243,441`
- unlabeled rows: `134`
- videos with labels: `995/995`
- +0.5s label movement histogram is tracked in `reports/again_labels_aligned_2hz_20260625_091209.md`

The first dense 2Hz raw-cortical-vs-AR benchmark has also run. It used MLX-backed ridge fits with train-only inner alpha selection, grouped-video folds as the primary gate, blocked temporal validation as the secondary protocol, and nuisance controls. Latest tracked reports:

- `reports/again_dense_2hz_ar_baseline_20260625_093722.md`
- `reports/again_dense_2hz_raw_cortical_vs_ar_20260625_094242.md`

Headline Phase 3 lesson, still not PCA/bridge proof: raw predicted cortical/fMRI features alone is weak. For `arousal_spike_rows_2_6_train_q90`, blocked `raw_cortical_only` was `0.124315` PR-AUC versus AR-only `0.203622`, and direct `AR_plus_raw_cortical` fell to `0.167731`. Grouped `AR_plus_raw_cortical` improved over AR (`0.170299` vs `0.147251`), but that direct raw lane is not the current claim and did not solve blocked validation.

## Current Downstream Evidence Status

AGAIN has since progressed beyond cache readiness and raw-cortical Phase 3 diagnostics:

- Blocked temporal binary confirmation: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`, real PR-AUC `0.2670735630`, frozen AR `0.2602336231`, best control `0.2593369051`, strong confirmation true, failed gates `[]`.
- Updated grouped compatibility: same target/head, real PR-AUC `0.2313831909`, AR/frozen `0.2174953276`, best matched control `0.2174209937`, fold-seed positives vs best control `50/50`, updated grouped compatibility pass true.
- Phase 5 eval-mode correction: grouped real PR-AUC `0.2300639382`, best matched control `0.2042740689`, delta `+0.0257898694`, fold-seed positive `15/15`; grouped continuous future-movement ranking/lift also passed, with real Spearman `0.2232222830` vs AR-only `0.1982207591`, shuffled `0.1938183619`, and random `0.1931781163`.
- Frozen-AR residual design: grouped frozen AR `0.2246816187`, best real residual `0.2383409298`, best matched residual control `0.2248361805`, delta vs AR `+0.0136593110`, delta vs control `+0.0135047493`.
- Phase 7 grouped continuous confirmation: `420/420`, real/AR/best-control Spearman `0.2603011121` / `0.2405371348` / `0.2402523335`, real/AR/best-control top-5% lift `0.0975979581` / `0.0895663763` / `0.0897088493`, all `15/15` fold-groups positive on both metrics, failed gates `[]`.

Current claim: bounded strict forward-time binary future-event ranking is proven, and controlled grouped continuous future-movement ranking/lift is independently confirmed by Phase 7 for the selected washout target/head. The separate Phase 7 blocked continuous run was a strong `4/5` near-pass. Exact values and label-free raw-video deployment are not yet claimed. No 504 run has been performed or promoted.

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

Cache cleanup: `113` stale `traceback.txt` files from successful per-video folders were removed after final `status.json`, global manifests, row counts, and sampled arrays passed. A local non-git cleanup manifest was written at `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/_run/cache_repair_20260625_stale_success_tracebacks.json`.

## Current Guardrails

- Use `labels_aligned_2hz.parquet` for dense supervised 2Hz work. Do not fall back to the older 1Hz boundary manifest for dense 2Hz claims.
- Do not drop rows silently because of black frames; preserve quality flags and decide filtering inside each train/test protocol.
- Do not re-encode videos unless the local/Drive artifact fails a manifest/schema audit.
- Do not copy the Drive bundle into git.
