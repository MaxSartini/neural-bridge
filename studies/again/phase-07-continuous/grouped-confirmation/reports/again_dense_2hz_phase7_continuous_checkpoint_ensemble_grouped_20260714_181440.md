# AGAIN Phase 7 grouped continuous checkpoint-ensemble validation

- Output: `outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440`
- Evaluation matrix: `420/420` cells (`315` member + `105` ensemble); these are not independent observations
- Target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`
- Real / AR / best-control Spearman: `0.2603011121` / `0.2405371348` / `0.2402523335`
- Real minus AR / best-control Spearman: `+0.0197639773` / `+0.0200487786`
- Real / AR / best-control top-5% lift: `0.0975979581` / `0.0895663763` / `0.0897088493`
- Real minus AR / best-control top-5% lift: `+0.0080315818` / `+0.0078891089`
- Grouped continuous ranking/lift pass: `True`
- Failed gates: `[]`

This grouped held-out-video result is independently claim-bearing for future-movement ranking and top-tail lift. The blocked-temporal protocol remains separately reported because it answers a different generalization question; exact-value trajectory prediction is outside this grouped endpoint.
