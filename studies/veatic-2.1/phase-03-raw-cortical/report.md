# VEATIC 2.1 Phase 03 Raw Predicted-Cortical Benchmark

Status: **PASS**

Phase 03 tested all 20,484 dimensions of the final TRIBE
`cortical_prediction` directly, before PCA, representation-width selection, or learned bridge
development. Every outer cell reused the exact Phase 02 target, row ownership, fold, seed,
q90 threshold, and frozen AR predictions. Training-owned standardization and a fixed
full-width diagonal-centroid classifier were fit with MLX on `gpu:0` in one worker. No
held-out row selected a feature, width, hyperparameter, or model.

The registered matrix contained 17 matched lanes per cell: frozen AR; real cortical-only and
direct AR-plus-real; and only/fusion variants of within-video shuffled cortical,
shape-matched random, train-only video mean, diagnostics-only, time/video-time-only,
quality/motion/luma-only, and label-permutation controls. Real cortical-only is also the
current-row/no-temporal-context ablation. No-video/architecture ablation was inapplicable
because Phase 03 has no video embedding or architecture branch.

Grouped-video median PR-AUC was `0.315086` for frozen AR,
`0.120929` for real cortical-only, and
`0.317626` for direct AR-plus-real. Blocked-temporal PR-AUC was
`0.276250`, `0.088738`, and `0.263731`, respectively.
Every lane has the complete spike metric stack, defined-only per-video PR-AUC, positive
counts, and exact held-out predictions. Primary real/fusion deltas against frozen AR and the
training-owned strongest matched control have paired whole-video bootstrap intervals.

Direct raw fusion claim gate: **FAIL**. Direct
fusion is a baseline and is not promoted by default. A scientific result failure does not
invalidate the control-complete Phase 03 execution; it motivates the already ordered Phase 04
question of whether a fold-owned PCA representation generalizes better.

No hidden-state file was opened or hashed. No grouped upstream feature, AGAIN runtime input,
PCA, washout target, representation width, or learned bridge entered Phase 03.

Code SHA-256: `51110daaa37578ae4f73d7b7cff3146d8a943e3aeeda3ca580283870f11c0fa1`
Prediction manifest SHA-256: `186acd0eb6017c7764fa1fc5215567e34ef5c55cfbee0b5bc94a0da6fc8b9d91`
Model manifest SHA-256: `720e5756f8883c90be628180ca33efd767c03bf2d1f4702951d53e13c9612f23`
Primary deltas SHA-256: `21808b3712c79a297a22c31b64a8da58b57b80222ddc9f93541a4a42c0734ac1`
Summary SHA-256: `5f453b5bdc9333d11ab18324c6de8cfc1675f5c47e8113525d8e0bb23efc1b15`
