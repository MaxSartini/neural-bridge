# TRIBE v2 Post-pass Output Schema

This bundle is a cache-only TRIBE v2 working-output dataset. It was produced
from existing V-JEPA 2.1 cache folders without decoding raw video or running
V-JEPA.

No PCA, bridge training, benchmarking, delta prediction, spike detection, or
exploratory modelling is performed by this post-pass.

Per-video outputs live under `per_video/<video_id>/`.

`tribe_v2_cortical_predictions.npz` contains:
- `cortical_prediction`: `[rows, 20484]` float16 unpooled 2Hz cortical predictions.
- `time_seconds`: `[rows]` float32 row timestamps.
- `tribe_grouped_video_feature`: `[rows, 2, 1408]` float16 adapted features fed to TRIBE.
- compact temporal std diagnostics, row-level quality signals, and sample frame/time arrays.
- quality flags: `quality_black_frame_flag`, `quality_duplicate_frame_flag`,
  `quality_exclusion_flag`, and `quality_weight_suggested`. These preserve
  black/static-row handling for later benchmarking without dropping rows here.

`baseline_features_rowlevel.npz` contains the compact non-cortical row-level
features needed for later local controls and nuisance baselines without loading
the full upstream V-JEPA cache.

`vjepa_temporal_diagnostics.npz` contains compact reductions of the large V-JEPA
temporal tensors. It intentionally does not copy full `features`,
`all_layer_features`, `temporal_mean`, or `temporal_std` from the upstream cache.

Global files include `video_metadata.csv`, `row_index.csv`, optional
`row_index.parquet` when pyarrow is installed, split manifests, and
`BASELINE_READINESS.md`.
