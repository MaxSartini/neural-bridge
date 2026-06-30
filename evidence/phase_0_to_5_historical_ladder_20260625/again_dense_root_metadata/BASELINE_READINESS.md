# Baseline Readiness

This output bundle is prepared for later local baselines and controls. It does
not run PCA, bridge training, benchmarking, spike/delta analysis, or label
alignment.

Saved arrays support:

- Autoregressive baseline: use future local label/annotation alignment with `row_index.csv` or `row_index.parquet` timestamps.
- Quality/motion/luma baseline: `baseline_features_rowlevel.npz` quality arrays plus `video_metadata.csv` summaries.
- Black/static-row controls: use `quality_exclusion_flag` to exclude flagged rows,
  or `quality_weight_suggested` to downweight them. This post-pass records the
  flags but does not drop rows.
- V-JEPA grouped baseline: `tribe_grouped_video_feature [rows,2,1408]`.
- TRIBE cortical baseline: `cortical_prediction [rows,20484]`.
- TRIBE plus temporal diagnostics baseline: `cortical_prediction` plus `vjepa_temporal_diagnostics.npz` compact temporal arrays.
- Shuffled/shifted controls: `row_index` timestamps, `sample_frame_indices`, `sample_time_seconds`, and per-video split manifests.

The bundle is intended to be sufficient for local PCA, bridge training,
benchmarking, delta testing, spike detection, controls, and reporting without
the full upstream V-JEPA cache. The upstream cache remains the authoritative
archive for full hidden states and full temporal tensors.
