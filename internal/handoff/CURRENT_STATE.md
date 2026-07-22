# Canonical Current State

Updated: 2026-07-22

## Confirmed result

- VEATIC 2.1 has 124 videos and 20,657 dense 2 Hz rows. The quality mask retains 19,734 rows; every video contributes to 13,753 discovery rows while 5,981 last-30% rows remain sealed.
- Train-only calibration produced 90 arousal-spike hypotheses: all 15 start/stop pairs across VEATIC's 0.5 s, 1 s, 3 s, 5 s, and 7.5 s movement anchors crossed with six fold-supported event quantiles. No target, representation, PCA width, model, or training recipe has won.
- Five fold-owned, label-blind cortical PCA bases are verified. Candidate widths are fold 0 `[8,20,59,64,128,184,256,512]`, fold 1 `[8,20,59,64,128,189,256,512]`, fold 2 `[8,20,59,64,128,186,256,512]`, fold 3 `[8,20,57,64,128,179,256,512]`, and fold 4 `[8,20,59,64,128,186,256,512]`.
- The last confirmed point is the coherent split, spike calibration, and reusable PCA bases. The unresolved gate is spike discovery with fresh AR and representation evidence.

## Fixed constraints

- Work on arousal spike only. Fit every target, AR, scaler, projection, model, gate, threshold, and checkpoint inside its owning VEATIC training fold; keep the sealed tail closed through winner freeze.
- Reuse the verified PCA bases by prefix. Do not refit them unless their substrate, quality mask, split, source, fold ownership, scaler, solver, or numerical implementation changes.
- Use identical folds and seeds for candidate comparisons. Every checkpoint from epoch 1 is merit-eligible; train at least 50 epochs, continue while validation can improve, impose no fixed epoch ceiling, and break exact ties toward the earlier checkpoint.
- Require meaningful value over fresh frozen AR, matched controls, fold/seed stability, leakage checks, and no-harm. AR fallback is a whole fold/seed inner-validation decision, never a row oracle.

## Exact next action

Run:

```bash
rtk uv run python -m neural_bridge.veatic21 screen-event \
  --preregistration '/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/event-spike-v1.json' \
  --calibration '/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/event-spike-v1-calibration.json' \
  --source vjepa_temporal_mean \
  --source tribe_grouped_mean \
  --source tribe_cortical \
  --output '/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/event-target-screen/tribe-cortical-tribe-grouped-mean-vjepa-temporal-mean.json'
```

Accept only a `veatic21_event_target_screen_v12` result covering all 5 folds, 90 targets, and 3 sources with freshly fitted matched AR, `benchmark_test_labels_accessed=false`, and no sealed rows. Summarize target/representation viability from that result, then define the full fold-owned PCA-width experiment; do not start learned-head training first.

Protocol: `/Users/maxsartini/Neural Bridge/internal/active/veatic21-event-preregistration.md`.
