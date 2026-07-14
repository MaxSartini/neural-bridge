# Phase 5 Blocked AR Decomposition Evidence 20260630_001424

Small no-training evidence bundle for the blocked AR decomposition audit.

- No training was run.
- No grouped, new variants, PCA, V-JEPA, or TRIBE work was run.
- Full output roots remain under `outputs/` and are not copied here.
- Frozen AR was audited using existing blocked split labels and frozen-AR score caches.
- Leakage classification: clean/legal, with legal but overpowering temporal autocorrelation.

Current status:

- This audit motivated the redesigned washout-gap target family.
- The later binary target `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual` passed blocked confirmation and repaired grouped compatibility.
- The AR leakage conclusion remains clean/legal; current caveats are about continuous exact forecasting and broader all-target/all-dataset generalization.
