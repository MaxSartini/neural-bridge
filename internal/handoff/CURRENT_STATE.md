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
- The head-family screen reuses all 90 verified causal PCA-512 cells and registers only the
  90 missing gated-multiscale cells on identical targets, folds, seeds, batch 4096, optimizer,
  residual cap, AR floors, and checkpoint rules. It uses exactly one sequential MLX worker.
- Canonical head-family screen:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/head-family-screen.json`.
  Exact screen SHA-256:
  `bbd9db2355962628243484e9c82962ca024a421b5c344c18bb37eeaf65b5a9c2`.
- One real non-promotable gated cell verified the multiscale design and artifact lifecycle.
  It completed 58 epochs and retained checkpoint 5. Its score is not head-selection evidence.
- All 90 gated cells completed. The result summary and every referenced cell metrics hash
  verify; sealed-tail labels were not accessed. Canonical summary:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/head-family-screen/summary.json`.
  Exact summary SHA-256:
  `aade8c0bf25a50a0c4b859e7ed8c47e10937e6cab2fc8dbfa5402d024c4ff22e`.
- The causal temporal residual remains selected. Its mean inner average-precision skill
  delta versus fresh AR was `+0.009118552`; gated multiscale scored `+0.007984341`.
  Paired gated-minus-causal was `-0.001134211`; causal won `54/90` pairs and gated won
  `36/90`. Gating helped q950, was nearly neutral at q925, and lost on q900/q875/q850/q800.
- Canonical head-family selection:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/head-family-selection.json`.
  Exact selection SHA-256:
  `d145c7f07dc0e825750532048bbedef4ddba4d3600adb84fa92b4ec6a7319d0a`.
- The VEATIC-owned staged numeric training-recipe program was registered. It reused the 90
  verified PCA-512 causal cells and registered up to 810 new learned cells, batch 4096,
  no artificial memory fraction, and exactly one sequential MLX worker. Canonical plan:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/training-recipe-plan.json`.
  Exact plan SHA-256:
  `f3e67221bb6f8200a1bbd957e9a13e862ac7434af12dbe0884d9a431d778c793`.
- The full sweep was stopped after 82 matched width-128 cells because an interim efficiency
  audit showed material harm and continuing the exhaustive matrix was wasteful. This stopping
  rule was not preregistered, so the partial sweep is not treated as a completed matrix winner.
  Width 128 trailed width 64 by a paired mean `-0.003195706`, and width 64 won `61/82`
  pairs. The complete, already-validated width-64 recipe is retained under the no-harm rule;
  no global numeric optimum is claimed. Canonical resolution:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/training-recipe-selection.json`.
  Exact resolution SHA-256:
  `d102727ca24510269f0b87784c3ac83f78171939a69b2a25cff184d18b142cab`.
- No valid stability expansion, final frozen winner, or sealed confirmation has been
  completed. The fixed nine-seed stability expansion was registered at
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/stability-plan.json`.
  Exact plan SHA-256:
  `ccc380cfa7c18549fce11b161ee020a44625e9526db27de25c293b33feef61b4`.
  It contains 270 cells for the one retained recipe and exactly one sequential MLX worker.
  It must not resume: its 159 completed cells are non-promotable because the prerequisite
  comparison control gate subsequently failed. The sealed tail remains unopened.
- Stability was stopped at `159/270` completed cells after correcting the execution order:
  matched controls must pass before any more stability compute. One interrupted stability cell
  remains non-promotable and is handled by the registered partial-cell quarantine on resume.
- The full procedure-level control crosswalk from Neural Bridge phases 0–7 is now explicit at
  `internal/active/veatic21-lifecycle-control-crosswalk.md`. AGAIN contributes control semantics
  and rigor only; no AGAIN fitted artifact or numeric winner is reused.
- The comparison control matrix is registered at
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/control-plan.json`.
  Exact plan SHA-256:
  `507201a9d064df049fe05d2e475951b8ee3ba012e99fd41183f3bafe15cdd289`.
  It reuses the 90 real causal and fresh-AR cells, then trains five matched residual controls
  plus one current-row ablation across the same targets, folds, and comparison seeds: 540 new
  cells, exactly one MLX worker. Stability cannot resume unless every registered gate passes.
- The comparison control matrix completed `540/540` and failed. Canonical summary:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/matched-controls/summary.json`.
  Exact summary SHA-256:
  `aaef211412e0362614a88d2d6a0c0488104d2a3ebc3e935fd2dfdb98fc1d282c`.
  Real causal residual mean skill delta versus AR was `+0.009118552`. The strongest matched
  control was causal-prefix video mean at `+0.008330933`; aggregate real-minus-control was
  `+0.000787619` and paired median was `+0.001444807`. Label permutation was appropriately
  below AR at `-0.006659042`, and real beat the current-row ablation by `+0.001119056` in
  aggregate. However q950 failed its target gate, folds 2 and 4 failed their fold gates, and
  therefore `all_gates_pass` is false.
- A no-training equal-weight diagnostic over the three comparison checkpoints did not repair
  cross-fold consistency: no target beat its strongest matched control in more than `3/5`
  folds. This is not a lucky-checkpoint failure. The current learned-bridge target shortlist,
  head selection, numeric recipe, and stability evidence are rejected for promotion.
- The earliest clean reusable boundary is the verified canonical substrate, label alignment,
  full fresh-AR benchmark, and label-blind fold-owned PCA cache. The learned bridge must be
  redesigned and evaluated control-complete from its first comparison cells.
- Permanent execution invariant: every learned run includes its real lane, identical frozen
  AR, all applicable matched controls, and its architecture/no-video ablation in the same
  registered matrix. Real-only pilots cannot authorize stability, promotion, confirmation,
  or later control backfill.
- A compact control-complete redesign is registered. It selects q925 and q900 from the failed
  comparison evidence only, removes static video level by subtracting each row's strictly
  prior causal-prefix PCA mean, and trains the real lane beside all five matched controls plus
  a current-innovation-only ablation. Canonical plan:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/innovation-control-plan.json`.
  Exact plan SHA-256:
  `82bd5e90b17305a36ea70431b47fd5909897f70268114b4378ee771529f5791b`.
  The matrix contains 210 cells and exactly one sequential MLX worker.

## Exact next action

Run the registered compact causal-innovation redesign. Reuse only the verified VEATIC
substrate, exact saved fresh-AR floors, fold-owned PCA-512 caches, target definitions, row
ownership, and lifecycle control semantics. Train every real cell beside its matched shuffled,
random, causal-prefix video-mean, diagnostics-only, label-permutation, and current-innovation
ablation lanes. Do not authorize stability unless every registered gate passes. Use exactly
one sequential MLX worker and do not open the sealed tail.

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
