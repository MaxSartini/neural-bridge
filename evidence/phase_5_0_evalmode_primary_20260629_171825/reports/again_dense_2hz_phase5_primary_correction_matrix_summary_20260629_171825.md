# AGAIN Dense 2Hz Phase 5 Primary Correction Matrix Summary

Historical output root (original run slug): `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825`

## Verdict

The original Phase 5 checkpoint restoration bug is fixed for this correction matrix: the best inner-validation checkpoint is saved and restored before test scoring. The checkpoint restore audit passed for all 702 completed train jobs.

The corrected grouped-video result strengthens the defensible claim: cross-video future arousal spike / emotional moment ranking is real and robust against matched shuffled/random PCA controls. Grouped support is strengthened by a `+0.0261` PR-AUC real-minus-matched-control delta, and the grouped delta is positive in `15/15` fold-seed comparisons.

The strict forward-time temporal claim remains unproven. The blocked-temporal matched control beats the real AR+PCA+diagnostic lane, so the corrected blocked gate fails.

## Corrected Matched-Control Gate

- Grouped-video real lane: `real_ar_pca_diag`, `regression_plus_binary`, mean PR-AUC `0.2282800471`.
- Grouped-video best matched control: `ar_plus_shuffled_pca`, `regression_plus_binary`, mean PR-AUC `0.2021634578`.
- Grouped matched-control delta: `+0.0261165893` PR-AUC, reported as `+0.0261` PR-AUC.
- Grouped fold-seed matched-control deltas: positive in `15/15` comparisons.
- Blocked-temporal real lane: `real_ar_pca_diag`, `regression_plus_binary`, mean PR-AUC `0.2199977456`.
- Blocked-temporal best matched control: `ar_plus_random_pca`, `regression_plus_binary`, mean PR-AUC `0.2288685899`.
- Blocked matched-control delta: `-0.0088708442` PR-AUC, reported as `-0.00887` PR-AUC.

## Sanity Controls

Label permutation stayed near chance, supporting no gross label leakage. For `regression_plus_binary`, grouped-video label permutation PR-AUC was `0.1056653365`; blocked-temporal label permutation PR-AUC was `0.1103050201`.

The video-mean PCA oracle diagnostic did not explain the grouped-video signal. For `regression_plus_binary`, grouped-video video-mean PCA oracle PR-AUC was `0.1044820659`; blocked-temporal video-mean PCA oracle PR-AUC was `0.1926814853`.

## Ranking Metrics

For the real `regression_plus_binary` lane, within-video and continuous ranking/lift metrics are supportive of ranking future arousal spikes / emotional moments:

- Grouped-video mean within-video PR-AUC: `0.2602418556`.
- Grouped-video mean within-video future-movement Spearman: `0.2425295289`.
- Grouped-video mean within-video top-1% lift: `3.4364789478`.
- Grouped-video mean within-video top-5% lift: `2.8695370255`.
- Grouped-video mean within-video top-10% lift: `2.4997106332`.
- Blocked-temporal mean within-video PR-AUC: `0.2978716136`.
- Blocked-temporal mean within-video future-movement Spearman: `0.1487958181`.
- Blocked-temporal mean within-video top-1% lift: `1.7392340268`.
- Blocked-temporal mean within-video top-5% lift: `1.6372717243`.
- Blocked-temporal mean within-video top-10% lift: `1.5853808942`.
- Grouped-video fold-seed continuous future-movement Spearman remained positive, approximately `0.2014` to `0.2442`.
- Grouped-video fold-seed top-1% lift was approximately `3.8271` to `5.4622`.
- Blocked-temporal continuous future-movement Spearman remained positive, approximately `0.1693` to `0.1821`, but the blocked matched-control PR-AUC gate still failed.

## Training Audit

- Completed rows: `702`.
- Failed rows: `0`.
- Checkpoints restored before scoring: `702`.
- Mean epochs run: `66.5199430199`.
- Median epochs run: `44`.
- Early stopping reasons: `659` patience-exhausted, `43` max-epochs-reached.
- Overfit flags: `480`.

No secondary targets, secondary features, or secondary model classes were started. The run used only target `arousal_spike_rows_2_6_train_q90`, feature `temporal_mean_2s_then_pca256`, and model `gated_ar_pca_mlp`.

## Next Diagnostic Direction

The next diagnostic should focus on why AR plus matched random/shuffled PCA can beat real PCA under blocked-temporal validation. Recommended next work:

- tighten blocked-temporal matched-control diagnostics around AR-only, AR+random PCA, and AR+shuffled PCA;
- add temporal-negative controls that preserve video identity but disrupt future alignment;
- separate within-video ranking from cross-video identity/static-content effects;
- report real-minus-matched-control as the headline effect for every promoted loss;
- keep strict forward-time temporal generalization unpromoted until real beats the best matched blocked control.
