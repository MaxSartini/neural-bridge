# AGAIN Phase 5.5: Grouped Compatibility

Purpose: check whether the confirmed blocked binary target/head is compatible with held-out-video grouped generalization.

Contents:
- 350-row grouped compatibility evidence
- updated frozen-AR-residual-aware verdict
- metrics, fold/seed deltas, controls, gates, failure reasons, leakage/context audit, label permutation audit, train-only video mean audit, and fold-safe grouped PCA metadata

Canonical numbers:
- real PR-AUC: 0.2313831909
- AR/frozen PR-AUC: 0.2174953276
- best matched control PR-AUC: 0.2174209937
- delta vs AR/frozen: +0.0138878634
- delta vs best matched control: +0.0139621972
- fold-seed positives vs best control: 50/50
- label permutation PR-AUC: 0.2153099775
- updated grouped compatibility pass: true

This is grouped-video compatibility, not a 504 run.
