# Current State — VEATIC 2.1 Spike Discovery

## Confirmed state

- Canonical input is only the VEATIC 2.1 TRIBE v2 `cortical_prediction` produced over the
  cached 2 Hz V-JEPA 2.1 encoder outputs:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716/per_video/<video_id>/tribe_v2_cortical_predictions.npz`.
- Exact TRIBE tree SHA-256:
  `0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd`.
- Substrate: 124 videos, 20,657 canonical rows, 923 black/static/end-screen exclusions,
  19,734 usable rows, 13,753 development rows, and a sealed 5,981-row tail.
- Train-only calibration retained all 90 arousal-spike target hypotheses.
- Five fold-owned 512-component cortical PCA payloads are verified and the active manifest
  exposes only candidate widths `64`, `128`, `256`, and `512`.
- The learned residual executor passed one real non-promotable cell and verified resume.
  It trained 84 epochs and selected checkpoint 34, demonstrating that epoch 1 onward is
  merit-eligible and the final checkpoint is not preferred. Its score is not selection
  evidence.
- No target, PCA width, learned representation, head, training recipe, checkpoint panel,
  fallback, or winner has been selected. The sealed tail has not been opened.

## Exact next action

Run the complete fresh-AR benchmark across all 90 targets, five folds, and the three fixed
comparison seeds:

```bash
cd '/Users/maxsartini/Neural Bridge'
uv run python -m neural_bridge.veatic21 benchmark-stage1-ar
```

Output is written resumably by target under:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/stage1-ar-benchmark`

When `summary.json` reports all expected cells, replace the same child plan in place:

```bash
uv run python -m neural_bridge.veatic21 prepare-stage1
```

The new plan must report `purpose: spike_discovery` and bind the AR summary SHA-256 before
learned PCA/head experiments begin.

## Execution order

1. Full target discovery and fresh AR benchmark.
2. Representation and PCA experiments.
3. Model and training experiments.
4. Fixed fold and seed stability.
5. Matched controls, leakage checks, and whole-fold/seed no-harm.
6. Inner-validation winner selection and freeze.
7. One sealed-tail confirmation.
8. Continuous arousal, then valence, then VEATIC zero-label-at-inference.
9. Combine confirmed VEATIC, AGAIN, and future dataset abilities into the production
   generalist for unseen client video.

The complete method and exact artifact paths are in
[`internal/active/veatic21-event-preregistration.md`](../active/veatic21-event-preregistration.md).
