# AGAIN Phase 4: Fold-Safe PCA Bridge

Phase 4 tested whether train-fold-fitted compression and causal temporal aggregation could turn dense cortical predictions into a useful bridge.

On the original grouped spike target, the promoted `AR_plus_PCA_plus_temporal_diagnostics` lane using `temporal_mean_2s_then_pca256` reached PR-AUC `0.1716`, above frozen AR `0.1473` and direct AR-plus-raw `0.1703`. Grouped held-out-video evidence governed promotion; blocked-temporal evidence remained diagnostic.

`evidence/` retains the run manifest, promotion decisions, summary/control/grouped/blocked metrics, leakage and integrity audits, and reports. A duplicated metric view and the 33 MB fold-detail table are omitted; they remain in the registered external benchmark core with the PCA features and components.

Fold-safe PCA and split contracts now live in `src/neural_bridge/again/`. The three phase-owned snapshots and their old shared-stack imports are intentionally not duplicated; compact evidence and registered external fitted artifacts remain authoritative.

PCA width `256` and the two-second temporal mean are historical Phase 4 outcomes, not inherited truths for VEATIC 2.1.
