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
- All 360 fixed-PCA cells completed. The result summary and every referenced cell metrics
  hash verify; sealed-tail labels were not accessed. Canonical summary:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/stage2-pca-screen/summary.json`.
  Exact summary SHA-256:
  `f0ece1ce929f184793c955354b0a8387af51334b382f32106a83ff174e3c23b4`.
- PCA width `512` won the registered primary key with mean inner average-precision skill
  delta versus fresh AR `0.006312049`. Widths `256`, `64`, and `128` scored `0.002299337`,
  `0.000682527`, and `0.000622105`, respectively. Width 512 was also mean-best for each
  of the six shortlisted targets, so no tie-break was invoked.
- Canonical PCA selection:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/stage2-pca-selection.json`.
  Exact selection SHA-256:
  `5d023c9fcfc822e2a1c1210ed794cf716ca11eb79e5299474fb5a4512b39e5e7`.
- The supervised representation screen is registered as one sequential MLX worker with
  matched batch size 4096. It contains 180 cells: PCA-512 and a fresh supervised 512-wide
  bottleneck on the same six targets, five folds, and three seeds.
- The supervised lane reuses each verified fold-owned cortical source scaler, then learns a
  fresh shared bias-free `20,484 -> 512` projection over the current row and five causal past
  rows. The PCA lane reuses the verified fold-owned PCA-512 projections. Both lanes use the
  same temporal residual head, optimizer recipe, checkpoint contract, AR baseline, and rows.
- Canonical supervised representation screen:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/supervised-projection-screen.json`.
  Exact screen SHA-256:
  `0b97abcb3d1be4139aefe757eb422a7a98d5c9c007c99cb01172f1ff22295873`.
- One real non-promotable supervised-bottleneck cell verified the wide source path,
  checkpointing, and artifact lifecycle. It completed 51 epochs and retained checkpoint 1.
  Its score is not representation-selection evidence.
- All 180 matched representation cells completed with one sequential MLX worker. The result
  summary and every referenced cell metrics hash verify; sealed-tail labels were not accessed.
  Canonical summary:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/supervised-projection-screen/summary.json`.
  Exact summary SHA-256:
  `d5aec6be6364f008b0bf60ef5bd982e3f2384fa12799bcf7f3fab4abec583221`.
- PCA-512 won all `90/90` matched target/fold/seed pairs. Its mean inner average-precision
  skill delta versus fresh AR was `+0.009118552`, positive in `90/90`. The supervised
  bottleneck scored `+0.000196811`, positive in `28/90`; paired supervised-minus-PCA was
  `-0.008921741`. PCA won all `15/15` pairs within every shortlisted target.
- The supervised 512-wide bottleneck is rejected and `fixed_pca512` remains the selected
  Neural Bridge representation. Canonical representation selection:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/supervised-projection-selection.json`.
  Exact selection SHA-256:
  `169fbd766972cada97db5ed5ebd953f3d37be250b6933957a520156e1a31f581`.
- No head family, training recipe, checkpoint panel, fallback, or final winner has been
  selected. The sealed tail remains unopened.

## Exact next action

Register the VEATIC-owned model and training matrix on the selected fold-owned PCA-512
representation. Compare the causal temporal residual and gated multiscale temporal residual
families on identical shortlisted targets, folds, comparison seeds, AR floors, and checkpoint
rules. Calculate architecture widths, optimizer candidates, residual caps, and safe batch
sizes from VEATIC 2.1 only; do not reuse AGAIN numeric winners.

Use exactly one GPU worker for learned cells. Do not launch parallel training processes.

Learned residual cells must use the plan-owned MLX capacity and the existing checkpoint
contract. The fresh-AR baseline remains the completed float64 CPU/LBFGS benchmark.

## Execution order

1. Complete target discovery and fresh AR benchmark. **Done.**
2. Register the target shortlist rule, then run representation and PCA experiments.
   **Done: PCA-512 selected; supervised bottleneck rejected `90/90`.**
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
