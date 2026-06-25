# VEATIC-124 Raw Representation Tensor Export v1

## Executive Verdict
Verification status: **pass**.

`pca_sequence_128_causal_past_2s_mean` is ready for learned-head training and is the recommended next model input. `roi_parcel_features` is ready as an unsupervised atlas-compressed side input. `topk_vertices_512` was exported as a supervised/cautionary tensor. `cortical_pca64_delta_frozen_baseline` was preserved as the frozen v2 reference.

## Export Scope
Exported 84 tensor contracts across four representations, seven splits, and three primary targets.
External tensor payloads: 420 `.npy` files.

## Source Cache and No-Reencode Confirmation
The export read existing TRIBE raw cortical predictions and existing audit fit caches. No videos were re-encoded and no model scoring was run.

## Exported Representations
- `pca_sequence_128_causal_past_2s_mean`
- `roi_parcel_features`
- `topk_vertices_512`
- `cortical_pca64_delta_frozen_baseline`

## Required Splits and Targets
Splits: `blocked`, `official`, `grouped_0`, `grouped_1`, `grouped_2`, `grouped_3`, `grouped_4`.

Targets: `arousal__future_spike_1_3s@0.05`, `arousal__future_spike_1_3s@0.075`, `arousal__future_change_p3s_movement@0.05`.

## Tensor Shape Summary
- `pca_sequence_128_causal_past_2s_mean` / `blocked` / `arousal__future_spike_1_3s@0.05`: train [5989, 128], test [2956, 128]
- `roi_parcel_features` / `blocked` / `arousal__future_spike_1_3s@0.05`: train [6237, 75], test [2956, 75]
- `topk_vertices_512` / `blocked` / `arousal__future_spike_1_3s@0.05`: train [6237, 512], test [2956, 512]
- `cortical_pca64_delta_frozen_baseline` / `blocked` / `arousal__future_spike_1_3s@0.05`: train [6237, 384], test [2956, 384]
- `pca_sequence_128_causal_past_2s_mean` / `blocked` / `arousal__future_spike_1_3s@0.075`: train [5989, 128], test [2956, 128]
- `roi_parcel_features` / `blocked` / `arousal__future_spike_1_3s@0.075`: train [6237, 75], test [2956, 75]
- `topk_vertices_512` / `blocked` / `arousal__future_spike_1_3s@0.075`: train [6237, 512], test [2956, 512]
- `cortical_pca64_delta_frozen_baseline` / `blocked` / `arousal__future_spike_1_3s@0.075`: train [6237, 384], test [2956, 384]
- `pca_sequence_128_causal_past_2s_mean` / `blocked` / `arousal__future_change_p3s_movement@0.05`: train [5989, 128], test [2708, 128]
- `roi_parcel_features` / `blocked` / `arousal__future_change_p3s_movement@0.05`: train [6237, 75], test [2708, 75]
- `topk_vertices_512` / `blocked` / `arousal__future_change_p3s_movement@0.05`: train [6237, 512], test [2708, 512]
- `cortical_pca64_delta_frozen_baseline` / `blocked` / `arousal__future_change_p3s_movement@0.05`: train [6237, 384], test [2708, 384]

## Best Next Learned-Head Input
`pca_sequence_128_causal_past_2s_mean`, because it is train-only PCA128 plus a causal past 2s mean window and does not use labels for feature construction.

## Frozen Baseline
`cortical_pca64_delta_frozen_baseline` preserves the existing `cortical_pca64_delta` feature definition for reference comparisons.

## Supervised/Cautionary Tensors
`topk_vertices_512` uses train-only supervised feature selection and includes selected-vertex metadata plus label-shuffle warning metadata. Treat it as cautionary unless confirmed under locked reruns.

## Video 83 Policy
Video `83` is included in the all-video tensor contracts. Exclude-video-83 sensitivity tensor export was skipped because this request asked for the required all-video split/target tensor contracts only.

## PCA Cache Reuse/Rebuild Summary
PCA cache reused count: 14. PCA cache rebuilt count: 0. Missing cache count: 0.

## Leakage and Verification Summary
Leakage contracts are written per tensor folder. Grouped folds were verified disjoint, PCA fit scope is train rows only, and causal sequence windows use no future rows.

## Missing or Skipped Artifacts
- exclude-video-83 sensitivity tensors were skipped; all required all-video contracts include video 83.
- No large .npy tensors are included in the review zip.

## Heavy External Outputs
Heavy `.npy` tensors and full row metadata live under `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`.

## Lightweight Tracked Outputs
Commit-safe summaries and metadata live under `<repo-root>/outputs/veatic_124_raw_representation_tensor_export_v1`. The review zip excludes `.npy` tensor payloads.

## Recommended Next Benchmark
Train learned heads first on `pca_sequence_128_causal_past_2s_mean`, compare against `cortical_pca64_delta_frozen_baseline`, and keep `roi_parcel_features` as a side candidate plus `topk_vertices_512` as a supervised cautionary comparison.
