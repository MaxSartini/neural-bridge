# AGAIN Phase 7 grouped continuous checkpoint-ensemble validation

- Output: `outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440`
- Matrix: `420/420` (`315` member + `105` ensemble)
- Target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`
- Real / AR / best-control Spearman: `0.2603011121` / `0.2405371348` / `0.2402523335`
- Real minus AR / best-control Spearman: `+0.0197639773` / `+0.0200487786`
- Real / AR / best-control top-5% lift: `0.0975979581` / `0.0895663763` / `0.0897088493`
- Real minus AR / best-control top-5% lift: `+0.0080315818` / `+0.0078891089`
- Grouped continuous ranking/lift pass: `True`
- Failed gates: `[]`

This user-authorized grouped test does not relabel the prior blocked `4/5` verdict and cannot prove exact continuous values.

## Interpretation

The `+8.22%` Spearman and `+8.97%` top-5% relative lifts are measured over a target-specific trained AR model, not a naïve constant or last-value baseline. That AR model uses current/lagged arousal and recent deltas, is fit per fold/seed, and is frozen before the real and control residuals are trained. The same frozen AR predictions sit beneath every lane.

Mean ensemble Spearman by lane:

- real residual: `0.2603011121`
- frozen AR only: `0.2405371348`
- diagnostics-only residual: `0.2402752332`
- train-only video-mean residual: `0.2402523335`
- random-PCA residual: `0.2399851718`
- shuffled-PCA residual: `0.2398151737`
- label-permutation residual: `0.2360263012`

The controls cluster around or below the AR floor; only the correctly aligned real bridge separates materially. This makes the result an incremental neuro-response signal result, not merely an extra-capacity result.

Historical context: on the earlier same-target raw-representation ablation, `raw_cortical_only` was `38.95%` below trained AR and direct `AR_plus_raw_cortical` remained `17.63%` below AR. The early ablation used a different target/metric from Phase 7, so no cross-task percentage is claimed. The defensible development conclusion is that Neural Bridge converted a representation with negative incremental value into a consistent positive correction over a strong learned persistence floor.

Compared with the original validated grouped continuous Phase 5 eval-mode bridge, Phase 7 improves future-movement Spearman from `0.2232222830` to `0.2603011121` (`+16.61%`), top-5% lift from `0.0789694843` to `0.0975979581` (`+23.59%`), and top-1% lift from `0.1359465244` to `0.1556892559` (`+14.52%`). The top-5% real-minus-AR margin grew from `+0.0040375083` to `+0.0080315818`, a `+98.92%` increase. This is a whole-system generation comparison; multiple design and training improvements changed together.

Supporting event metric: Phase 7's continuous prediction ranks the corresponding binary event at PR-AUC `0.2231895329` versus frozen AR `0.2088047413` and strongest control `0.2096090680`, or `+6.89%` / `+6.48%`, with `15/15` fold-groups positive versus both. This is not a primary promotion gate, but confirms that the continuous bridge retains useful spike/event ordering.
