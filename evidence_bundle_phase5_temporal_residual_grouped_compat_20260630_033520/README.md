# Phase 5 Temporal Residual Grouped Compatibility Evidence

This is a lightweight tracked evidence snapshot for the grouped-video compatibility run:

- Output root: `outputs/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520`
- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Protocol: `grouped_video`
- Folds: full 5 grouped-video folds
- Seeds: 10 seeds, `20260625` through `20260634`
- Architecture: `short_temporal_conv_residual`
- Residual/control scored rows: 350
- AR baselines newly trained: 50 fold/seed baselines
- AR baselines reused: 0
- Each fold/seed uses its own frozen AR score.
- All controls within each fold/seed use identical frozen AR scores.
- No shared 3-seed AR cache was reused across the 10-seed grouped compatibility run.

Key result:

- Real PR-AUC: `0.2313831909`
- AR/frozen baseline PR-AUC: `0.2174953276`
- Best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- Delta vs AR/frozen baseline: `+0.0138878634`
- Delta vs best control: `+0.0139621972`
- Fold-seed positives vs best control: `50/50`
- Grouped compatibility pass: `false`
- Failed gate: `label_permutation_not_near_chance`
- Recommendation: `grouped_compatibility_failed_do_not_run_504`

This bundle contains small MD/CSV/JSON artifacts only. It excludes heavy PCA score arrays, checkpoints, row-index CSVs, and the full ignored output root.
