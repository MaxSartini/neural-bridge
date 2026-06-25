# AGAIN Dense 2Hz Phase 5 Feature Inputs

- Output root: `outputs/again_dense_2hz_phase5_learned_heads_20260625_182423`
- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full`
- Backend priority: `mlx` with CUDA barred.
- Primary target: `arousal_spike_rows_2_6_train_q90`
- Continuous source: `future_arousal_max_delta_rows_2_6`
- Feature inputs: `temporal_mean_2s_then_pca256`
- PCA reuse: fold-safe Phase 4 score artifacts only; no global PCA refit.
- Timing: true 2Hz row timing from `labels_aligned_2hz.parquet` is preserved.
- Scaling: train-only input normalization inside each fold/head.
