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
washout candidate was opened while producing the original registration freeze. Subsequent
inner-only Stage A execution, rescue, aggregation, and exact Stage B registration are tracked
by the additional immutable files in this directory and the live handoff; this paragraph is
not a claim that the original registration remains the current execution boundary.

`stage-b-execution-registration.json` is the prospective execution-identity and real-data
systems-backtest freeze for the verified `40,824`-work-unit, `2,351,229`-cell Stage B
registry. It fixes the previously underspecified seed, GRU ordering, optimizer regularization,
checkpoint, plateau, recovery, artifact, topology, and hardware-gate semantics before any
Stage B fit. The complete registered backtest and independent verification passed over all
`25` topologies, `75` fresh-process repetitions, and `2,550` representative real-data cell
artifacts. `selected-stage-b-executor.json` freezes one CPU preparation worker and four MLX
stream lanes by the prospective within-three-percent fewer-resource rule. Its committed hash
is the only authority for the complete Stage B main executor; backtest cells remain disposable
systems evidence and never enter the scientific ledger.
