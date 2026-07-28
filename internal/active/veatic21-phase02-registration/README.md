# VEATIC 2.1 Phase 02 frozen AR registration

This is the repository-side freeze for Phase 02 before any outer model score. It passed the
registration verifier over the sealed Phase 01 substrate and is paired with the external
split/support evidence at:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-02-target-specific-ar/registration`

The freeze covers all 21 active no-washout targets. It defines four independent 10-fold
grouped-video partitions with four nested inner folds, plus two expanding blocked-forward
folds over four native-time blocks. Every evaluation cell clears the registered minimum of
1,000 rows; the observed minima are 1,219 grouped test rows, 2,672 blocked test rows, and
2,630 blocked inner-validation rows.

The registered search includes all 21 causal history depths, six causal feature forms,
analytic controls, continuous ridge, L2 logistic, elastic-net logistic, MLP, and GRU families,
with data-scaled regularization, capacity, optimizer, learning-rate, batch, update-budget,
calibration, boundary-expansion, undertraining-recovery, and five fresh-seed rules.

No AR model, outer label/score, cortical value, PCA, learned bridge/head, or prospective
washout candidate was opened while producing this freeze. The exact next action is execution
of this registration, without changing its candidate families or using outer results to tune.
