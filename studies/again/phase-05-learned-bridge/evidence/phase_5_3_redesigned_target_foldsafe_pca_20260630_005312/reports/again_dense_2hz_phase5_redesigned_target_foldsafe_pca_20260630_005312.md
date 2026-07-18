# Phase 5 Redesigned Target Fold-Safe PCA 005312

Output root: `outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312`

This is a PCA-projection-only preparation run for the redesigned blocked target follow-up. It does not train residual models, does not run grouped, does not rerun V-JEPA/TRIBE, does not use global PCA, and does not modify the dense cache.

## Why This Was Needed

The redesigned blocked split differs from the original Phase 5 split. Reusing the original Phase 4 PCA artifacts would put redesigned test rows into a PCA basis fit that saw those rows as train rows. This run creates target/protocol/fold-specific train-only PCA projections for the two approved redesigned targets.

## Prepared Targets

| Target | Train rows | Test rows | PCA width | Explained variance | Scores artifact |
|---|---:|---:|---:|---:|---|
| future_arousal_max_delta_rows_4_10_train_q90 | 159923 | 69101 | 256 | 0.9990781373 | `outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312/features/future_arousal_max_delta_rows_4_10_train_q90__blocked_temporal_70_30__fold1__temporal_mean_2s__scores_w256.npy` |
| residual_future_max_delta_rows_4_10 | 156873 | 68008 | 256 | 0.9990749483 | `outputs/again_dense_2hz_phase5_redesigned_target_foldsafe_pca_20260630_005312/features/residual_future_max_delta_rows_4_10__blocked_temporal_70_30__fold1__temporal_mean_2s__scores_w256.npy` |

## Leakage Audit

- leakage audit pass: `True`
- no test rows used in PCA/scaler fit: `true`
- redesigned test rows excluded from PCA fit: `true`
- original PCA artifact reused: `false`
- no global PCA: `true`
- no test fit: `true`
- V-JEPA/TRIBE rerun: `false`
- dense cache modified: `false`
- target-window overlap unchanged: `false`
- future leakage introduced: `false`

Dense root read-only source: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz`

## Safe To Run Redesigned Target Training

`True`

Use this output root as the fold-safe PCA source for the redesigned blocked run. Do not commit the heavy `.npy`/`.npz` PCA artifacts; only reports/manifests/audits should be considered for a later commit.