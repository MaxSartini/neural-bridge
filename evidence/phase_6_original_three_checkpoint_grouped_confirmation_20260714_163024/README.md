# Phase 6 Original Three-Checkpoint Grouped Confirmation Evidence

Status: promoted bounded grouped-video evidence; all preregistered gates passed.

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- matrix: `420/420` (`315` member + `105` ensemble)
- fresh seeds: `20260675`–`20260683`; fixed three-seed groups; five grouped-video folds
- real / AR / best-control PR-AUC: `0.2343675680` / `0.2180497906` / `0.2179716645`
- best aggregate matched control: `train_only_video_mean_residual`
- deltas versus AR / best control / real-member mean: `+0.0163177774` / `+0.0163959035` / `+0.0082200727`
- wins versus AR / per-fold-group best control / member mean: `15/15` / `15/15` / `15/15`
- positive fold means versus AR / best control: `5/5` / `5/5`
- audit: grouped PCA leakage, causal context, frozen-AR identity, checkpoint restore, exact scope, and MLX passed
- failed gates: `[]`

Tracked report: `reports/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024.md`

Heavy output root: `outputs/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024/`

SHA-256 anchors:

- `metrics/result.json`: `6a2ec7245fefda8e4fcc0fdbfb8919702233409c200205979e37ebd878991c09`
- `metrics/fold_group_deltas.csv`: `0bb8a5dd469890eaac700c0c87c94494c6630bf3b44a43aff2b0544ac07451aa`
- `diagnostics/audit.json`: `0fa9f918fb150ad45d4988a26cd7b1632651cba7019285487a7019f26e22b501`
- `manifests/run_manifest.json`: `a307304db7fdade046c9e1f25d428c7a74e9c101a64874b625a4a28726c6fac5`
- tracked report: `5156ca07c0c1fa11fc4b5802f3d2478244e966b0fe7435b8ad6257f1905bf837`

Boundary: this promotes only the original three-checkpoint ensemble for the
selected binary target/head under grouped held-out-video evaluation. It does
not establish exact continuous arousal forecasting or broad universal
prediction.
