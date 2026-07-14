# AGAIN Phase 7 blocked continuous checkpoint-ensemble confirmation

- Output: `outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_blocked_confirm_20260714_175653`
- Target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`
- Matrix: `140/140` rows
- Real / AR / best-control Spearman: `0.1176781535` / `0.1103312855` / `0.1072552766`
- Real minus AR / best-control Spearman: `+0.0073468679` / `+0.0104228768`
- Real / AR / best-control top-5% lift: `0.0840262922` / `0.0759273576` / `0.0757026078`
- Real minus AR / best-control top-5% lift: `+0.0080989346` / `+0.0083236843`
- Blocked continuous ranking/lift confirmation pass: `False`
- Failed gates: `['spearman_positive_vs_ar_5_of_5']`

This confirmation is scoped to blocked continuous future-movement ranking/lift. Exact continuous values remain unproven regardless of this verdict.
