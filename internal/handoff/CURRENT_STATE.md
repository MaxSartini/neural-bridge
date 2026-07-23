# Current State — VEATIC 2.1 Spike Discovery

## Confirmed state

- Canonical input is only the VEATIC 2.1 TRIBE v2 `cortical_prediction` produced over the
  cached 2 Hz V-JEPA 2.1 encoder outputs:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716/per_video/<video_id>/tribe_v2_cortical_predictions.npz`.
- Exact TRIBE tree SHA-256:
  `0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd`.
- Substrate: 124 videos, 20,657 canonical rows, 923 black/static/end-screen exclusions,
  19,734 usable rows, 13,753 development rows, and a sealed 5,981-row tail.
- The complete fresh-AR benchmark finished across all 90 calibrated target hypotheses,
  five grouped-video folds, and seeds `20260722`, `20260723`, and `20260724`.
- All 1,350 expected fresh-AR cells completed and zero cells were invalid. The benchmark
  did not access sealed-tail labels and is not itself promotable evidence.
- Canonical AR summary:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/stage1-ar-benchmark/summary.json`.
- Exact AR summary SHA-256:
  `5a9dfecb2d4c0b1387677c9f02a2c4b1be9692f40cb5b9091ff21946469c8e2a`.
- The replace-in-place Stage-1 child plan now reports `purpose: spike_discovery` and binds
  that AR summary. Exact plan SHA-256:
  `e166d18558b59edbb4633f8e6a6b3abab85c0d5d0eedb3202b4f35eb1fddf7ee`.
- Five fold-owned 512-component cortical PCA payloads remain verified, and the active plan
  exposes only candidate widths `64`, `128`, `256`, and `512`.
- The learned residual executor passed one real non-promotable cell and verified resume.
  It trained 84 epochs and selected checkpoint 34, demonstrating that epoch 1 onward is
  merit-eligible and the final checkpoint is not preferred. Its score is not selection
  evidence.
- A train-only Stage-2 shortlist now contains exactly one target per preregistered quantile.
  Within each quantile, targets were ranked by mean fresh-AR average-precision skill across
  all 15 fold/seed cells, then minimum skill descending, dispersion ascending, and name.
- The six shortlisted targets are:
  `arousal_positive_max_0p5_0p5s_train_q950`,
  `arousal_positive_max_0p5_1s_train_q925`,
  `arousal_positive_max_0p5_1s_train_q900`,
  `arousal_positive_max_0p5_1s_train_q875`,
  `arousal_positive_max_0p5_1s_train_q850`, and
  `arousal_positive_max_1_1s_train_q800`.
- The fixed-PCA screen varies only widths `64/128/256/512` across those six targets, five
  folds, and three comparison seeds: 360 expected cells. The nuisance recipe is copied from
  the current VEATIC executor-validation configuration; its score was not used for selection.
- Canonical Stage-2 screen:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/stage2-pca-screen.json`.
  Exact screen SHA-256:
  `b756bdcbf533466047c2b664aa9dac95f9a2146da3c9491994565b9412241114`.
- No PCA width, learned representation, head, training recipe, checkpoint panel, fallback,
  or winner has been selected. The sealed tail remains unopened.

## Exact next action

Run the complete registered 360-cell fixed-PCA screen:

```bash
cd '/Users/maxsartini/Neural Bridge'
uv run python -m neural_bridge.veatic21 benchmark-stage2-pca
```

Output is written resumably by completed cell under:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/stage2-pca-screen`

Learned residual cells must use the plan-owned MLX capacity and the existing checkpoint
contract. The fresh-AR baseline remains the completed float64 CPU/LBFGS benchmark.

## Execution order

1. Complete target discovery and fresh AR benchmark. **Done.**
2. Register the target shortlist rule, then run representation and PCA experiments.
   **Shortlist and fixed-PCA matrix registered; screen running next.**
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
