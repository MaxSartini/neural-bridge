# VEATIC 2.1 Spike Protocol

Updated: 2026-07-22

## Confirmed substrate

- 124 VEATIC videos produce 20,657 dense 2 Hz rows. The quality mask retains 19,734 rows after excluding black/end-screen and high-duplicate rows.
- The per-video temporal split owns 13,753 first-70% discovery rows and keeps 5,981 last-30% rows sealed.
- Calibration contains 90 arousal-spike hypotheses: all 15 start/stop pairs across VEATIC's 0.5 s, 1 s, 3 s, 5 s, and 7.5 s movement anchors crossed with six fold-supported event quantiles. They are discovery candidates, not winners.
- Five fold-owned, label-blind cortical PCA bases are complete. Each 512-component basis can supply its verified variance widths plus 64, 128, 256, and 512 without refitting.

## Ordered gates

1. Screen all spike targets against freshly fitted fold-owned AR and the three cached representation views.
2. Benchmark the full fold-owned PCA width set and any VEATIC-trained supervised representation candidates supported by gate 1.
3. Compare VEATIC-trained model and training recipes on identical folds and seeds.
4. Confirm shortlisted recipes across the fixed fold and seed panels.
5. Run matched controls, leakage checks, and no-harm tests against frozen AR.
6. Select and freeze one winner from inner-validation evidence.
7. Open the sealed tail once for confirmation, then refit the frozen recipe on all usable rows for production.

## Invariants

- Fit targets, AR, scalers, projections, models, gates, and thresholds only inside their owning VEATIC training fold. Use labels for supervised discovery and selection; keep PCA label-blind.
- AGAIN supplies hypotheses and compatible rigor only. Reuse no AGAIN fitted object or numeric choice.
- Checkpoints compete only on merit: every completed checkpoint from epoch 1 is eligible, training cannot stop before epoch 50, there is no fixed epoch ceiling, and an exact tie selects the earlier checkpoint.
- A residual lane may fall back to unchanged AR only as a whole fold/seed decision selected on inner validation. Never switch from observed row error.
- Promotion requires meaningful improvement over fresh AR and matched controls, stability across folds/seeds, leakage checks, and the registered no-harm gate.

## Current evidence

- Preregistration: `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/event-spike-v1.json`
- Calibration: `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/event-spike-v1-calibration.json`
- PCA manifest: `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge/cortical-pca-v1/manifest.json`

The exact current command and acceptance check live only in `/Users/maxsartini/Neural Bridge/internal/handoff/CURRENT_STATE.md`.
