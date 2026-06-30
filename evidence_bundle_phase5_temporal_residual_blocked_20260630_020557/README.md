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

Key result:

- Binary washout-gap target passed with `short_temporal_conv_residual`.
- Continuous AR-residualized target did not pass.
- Failed gates: `continuous_min_delta_threshold`, `continuous_seed_consistency`.
- Recommendation: `temporal_residual_blocked_failed_do_not_run_grouped_or_504`.
- Strict forward-time temporal generalization remains unproven.

This bundle intentionally excludes checkpoints, frozen AR row-score caches, dense cache files, PCA arrays, and the ignored full output root.
