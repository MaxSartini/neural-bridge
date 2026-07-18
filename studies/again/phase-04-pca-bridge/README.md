# AGAIN Phase 4: Fold-Safe PCA Bridge

## Outcome

Phase 4 established the first leakage-safe representation bridge: train-fold-fitted compression followed by short causal temporal aggregation. It improved grouped event ranking, but its narrow margin over direct fusion showed that fixed PCA was a stepping stone rather than the final solution.

## Research question

Can fold-owned dimensionality reduction expose useful structure in 20,484 predicted cortical vertices without fitting a projection on held-out videos, and does short temporal context add value beyond current-row summaries?

## Design

- PCA, scaling, and any target-dependent choices fitted inside the training fold.
- Promoted representation: `temporal_mean_2s_then_pca256`.
- Promoted lane: `AR_plus_PCA_plus_temporal_diagnostics`.
- Grouped held-out-video evidence governed promotion; blocked-temporal evidence remained a separate diagnostic.
- Shuffled, random, time, quality, motion, and luminance controls remained in the benchmark.

## Decisive evidence

| Grouped spike system | PR-AUC | Change vs frozen AR |
| --- | ---: | ---: |
| Frozen AR | `0.147251` | baseline |
| Direct AR + raw cortical | `0.170299` | `+15.65%` |
| Fold-safe PCA temporal bridge | **`0.171648`** | **`+16.57%`** |

The bridge improved by only `0.001349` over direct fusion—about **`+0.79%`**. That modest increment was scientifically useful: it proved that fold-safe compression and temporal aggregation could work, while rejecting the idea that a fixed PCA recipe alone had solved the representation problem.

## Audit trail

[`evidence/`](evidence/) retains the run manifest, promotion decisions, grouped and blocked metrics, control summaries, leakage audits, integrity audits, and reports. A duplicated metric view and the `33 MB` fold-detail table remain in the registered external benchmark core with the fitted PCA features and components.

Current fold-safe PCA and split contracts live in [`src/neural_bridge/again/`](../../../src/neural_bridge/again/). Historical phase snapshots and their obsolete shared-stack imports are not duplicated as a second API.

## Transition and boundary

Phase 5 replaced fixed compression/readout with learned temporal residual heads. PCA width `256` and the two-second mean remain historical Phase 4 outcomes—not inherited truths for VEATIC 2.1 or any future target.

[Continue to Phase 5 — learned bridge](../phase-05-learned-bridge/README.md) · [Return to the journey](../../README.md)
