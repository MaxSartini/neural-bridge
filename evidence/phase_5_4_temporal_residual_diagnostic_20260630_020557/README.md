# Phase 5 Temporal Residual Blocked Evidence Bundle

This is a lightweight tracked inspection bundle for the bounded temporal/event-context residual diagnostic.

Full output root:

`outputs/again_dense_2hz_phase5_temporal_residual_blocked_20260630_020557/`

Scope:

- blocked_temporal_70_30 only
- redesigned targets only
- 2 targets x 3 seeds x 4 architectures x 7 controls = 168 rows
- fold-safe redesigned PCA256 artifacts only
- frozen AR baseline floor
- no grouped run
- no 504
- no V-JEPA/TRIBE/PCA rerun
- no PCA refit
- no AR retraining

Current status:

- This bounded diagnostic identified `short_temporal_conv_residual` as the binary washout-gap candidate.
- A later matched seed-specific 10-seed blocked confirmation passed for the binary target `future_arousal_max_delta_rows_4_10_train_q90`.
- Continuous exact arousal movement/ranking remained mixed and is still open.

Key result:

- Binary washout-gap target passed with `short_temporal_conv_residual`.
- Continuous AR-residualized target did not pass.
- Failed gates: `continuous_min_delta_threshold`, `continuous_seed_consistency`.
- Diagnostic-era recommendation was to avoid grouped or 504 until a binary-only confirmation was reviewed.
- This diagnostic alone did not prove strict temporal generalization; the later binary-only 10-seed confirmation is the current blocked proof for the redesigned target/head.

This bundle intentionally excludes checkpoints, frozen AR row-score caches, dense cache files, PCA arrays, and the ignored full output root.
