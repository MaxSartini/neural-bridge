# Phase 7 Continuous Checkpoint-Ensemble Grouped Evidence

Status: promoted bounded grouped-video continuous future-movement ranking/lift evidence; all preregistered gates passed.

- target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`
- matrix: `420/420` (`315` member + `105` ensemble)
- fresh seeds: `20260708`–`20260716`; fixed three-seed groups; five grouped-video folds
- real / target-specific AR / best-control Spearman: `0.2603011121` / `0.2405371348` / `0.2402523335`
- Spearman deltas versus AR / best control: `+0.0197639773` / `+0.0200487786`
- real / AR / best-control top-5% lift: `0.0975979581` / `0.0895663763` / `0.0897088493`
- top-5% deltas versus AR / best control: `+0.0080315818` / `+0.0078891089`
- wins versus AR / best control: Spearman `15/15` / `15/15`; top-5% `15/15` / `15/15`
- positive fold means: `5/5`
- ensemble uplift over member mean: `+0.0077966938` Spearman / `+0.0025021192` top-5%
- audit: grouped PCA leakage, causal context, frozen-AR identity, checkpoint restoration, exact scope, and MLX passed
- failed gates: `[]`

Value interpretation: the `8.22%` Spearman and `8.97%` top-5% relative lifts are the hard residual gain over a trained target-specific AR persistence model. They are not total-system value. Earlier same-target ablation showed raw cortical-only `38.95%` below trained AR and direct AR-plus-raw `17.63%` below AR; Phase 7's real aligned bridge instead beats AR and controls in every fold-group. The earlier PR-AUC task and Phase 7 continuous task are not combined into one invalid cross-task percentage.

Generation improvement: versus the original validated grouped continuous bridge, Phase 7 is `+16.61%` on Spearman, `+23.59%` on top-5% lift, and `+14.52%` on top-1% lift. Its top-5% margin over AR is `+98.92%` larger. This compares complete system generations, not one isolated component.

Spike/event context: within the original grouped target, the bridge progressed from raw cortical `0.136579` to frozen-AR residual `0.2383409298` (`+74.51%`). Phase 7's continuous output has supporting event PR-AUC `0.2231895329` vs AR `0.2088047413` and strongest control `0.2096090680`, positive `15/15`; this is secondary evidence, not the primary Phase 7 gate.

Tracked report: `reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`

Heavy output root: `outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440/`

SHA-256 anchors:

- `metrics/result.json`: `9aa4323aa4d80f3f08620750b5a5e30b21e0bacdd3cc4df08a7134acb8fc878e`
- `metrics/fold_group_deltas.csv`: `36b17a03c6612ebc5cb0d8089e3de369b540a97ae9bbfcef590662948a690415`
- `diagnostics/audit.json`: `adb97fe3855ccd2a2097ed57cf1c725931f454e0e53e50ccbdc0ca49eb40ed29`
- `manifests/run_manifest.json`: `707a412e261043d430e9bcf721b9b751e578abdc12ad91e64dd49952ca4ea86a`
- tracked report: `971b16c068fa3aa934fb11ba625098b43d664246a939f05d51c6897bdfba1bd6`

Boundary: this promotes grouped held-out-video continuous future-movement ranking/lift only for the selected target/head and fixed checkpoint-ensemble protocol. The separate blocked confirmation remains a literal `4/5` near-pass, and `exact_continuous_value_forecasting_proven` remains `false`.
