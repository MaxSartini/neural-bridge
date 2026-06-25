# AGAIN Dense 2Hz Phase 4 PCA Feature Build

- input cache root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz`
- label manifest path: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/labels_aligned_2hz.parquet`
- dense rows: `243575`
- labeled rows: `243441`
- PCA widths: `[64, 128, 192, 256]`
- feature families: `['current', 'delta', 'pca_then_temporal', 'temporal_then_pca']`
- PCA policy: train-only inside target/protocol/fold row set; max width fit is sliced for narrower widths.
- fold-specific PCA fits: `216`
- feature artifact root: `/Volumes/onn. Drive/Neural Bridge/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/features`

## Explained Variance

- width `64`: mean explained variance ratio sum `0.908219` across `54` fits
- width `128`: mean explained variance ratio sum `0.971036` across `54` fits
- width `192`: mean explained variance ratio sum `0.991914` across `54` fits
- width `256`: mean explained variance ratio sum `0.997428` across `54` fits

## Guardrails

- no V-JEPA/TRIBE re-encoding
- no global PCA for promoted claims
- row identity is keyed by `video_id,row_index,time_seconds`
- delta PCA explicitly drops first rows per video
