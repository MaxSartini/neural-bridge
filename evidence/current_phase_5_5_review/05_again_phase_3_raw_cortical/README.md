# AGAIN Phase 3: Raw Cortical Baseline

Purpose: test raw dense cortical bridge features against AR before fold-safe PCA bridge construction.

Bottom line: raw cortical-derived features alone fail badly on the original dense spike target. For `arousal_spike_rows_2_6_train_q90`, blocked `raw_cortical_only` PR-AUC was `0.124315` versus AR-only `0.203622`, and direct `AR_plus_raw_cortical` was only `0.167731`, below AR. Grouped `raw_cortical_only` was `0.136579` versus AR-only `0.147251`; direct `AR_plus_raw_cortical` reached `0.170299`, but that was not enough to establish the current claim and did not solve blocked validation.

Contents:
- raw cortical vs AR report
- fold metrics, summary metrics, promotion gates, run manifest, and summary JSON
- raw cortical block and temporal diagnostic metadata where relevant

This phase is the negative-control reason Neural Bridge matters. The current result is not raw cortical alone; it is the fold-safe, AR-controlled Neural Bridge pipeline built on cortical-response-derived representations.
