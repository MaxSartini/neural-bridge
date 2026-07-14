# Neural Bridge Roadmap

This roadmap starts from the current Phase 5.5 evidence ladder: VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal, and AGAIN replicated, scaled, validated, and strengthened it with dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residual design, blocked washout-gap confirmation, and updated grouped-video compatibility.

## Best Current Result

AGAIN now has a bounded strict forward-time future-event ranking result:

- target: `future_arousal_max_delta_rows_4_10_train_q90`
- protocol: `blocked_temporal_70_30`
- architecture: `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- frozen AR PR-AUC: `0.2602336231`
- best control: `random_pca_residual`, PR-AUC `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best control: `+0.0077366579`
- seeds positive vs AR: `9/10`
- seeds positive vs best control: `9/10`
- weak / credible / strong confirmation: true
- failed gates: `[]`

The same target/head also passed updated grouped-video compatibility:

- protocol: `grouped_video`
- rows: `350/350`
- real PR-AUC: `0.2313831909`
- AR/frozen PR-AUC: `0.2174953276`
- best matched control: `train_only_video_mean_residual`, PR-AUC `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best matched control: `+0.0139621972`
- fold-seed positives vs best control: `50/50`
- updated grouped compatibility pass: true

## Current Claim Boundary

Proven:

- Controlled future human arousal event-ranking from frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data across VEATIC and AGAIN.
- Bounded strict forward-time future-event ranking on AGAIN for the redesigned washout-gap binary target/head.
- Grouped held-out-video compatibility for the same AGAIN target/head under the updated frozen-AR-residual-aware verdict.

Open:

- Continuous exact arousal forecasting.
- Broad all-target/all-dataset temporal prediction.
- Full multimodal text+audio+video TRIBE coverage.

## Completed Foundations

- VEATIC-124 v2 strict evidence suite and protected snapshot.
- VEATIC raw-representation audit and frozen tensor contract.
- VEATIC trained-head layer over frozen tensors.
- Dense AGAIN H100 V-JEPA 2.1 / TRIBE v2 cache: `995/995` videos, `243,575` video feature rows generated from video by upstream models trained on brain cortical response data, true `2 Hz` labels.
- AGAIN Phase 3 raw predicted cortical/fMRI feature-only negative-control result: blocked `raw_cortical_only` PR-AUC `0.124315` vs AR-only `0.203622`, proving raw predicted cortical/fMRI features alone are not the win.
- AGAIN Phase 5 eval-mode correction.
- AGAIN frozen-AR residual design.
- AGAIN blocked AR decomposition and target redesign audits.
- AGAIN temporal/event-context residual diagnostic.
- AGAIN 10-seed blocked binary confirmation.
- AGAIN updated grouped-video compatibility verdict.
- Full bounded selected-head consolidation: `420/420` audited rows (`70/70` blocked plus `350/350` grouped), with no reruns required.
- One-seed, 16-trial Optuna selected-head pilot on MLX: exact original reproduction at `0.2697372519`, tuned result `0.2718557352`, and `+0.0081646445` vs frozen AR/best control. This is exploratory, not promoted.
- Locked-winner 10-seed Optuna confirmation: tuned beat original in `7/10` seeds but failed the preregistered aggregate-improvement gate (`-0.0011081356` mean delta). It retained positive controlled deltas versus AR and controls. Seed `20260627` was an unusually favorable canonical-original peak, but remains part of the failed verdict.
- Robust multi-seed Optuna Stage A: 24 trials across five development seeds, checked on five reserved inner-validation seeds. Best candidate won `4/5` but was effectively tied on mean and worse on the robust objective; the planned 720-row held-out campaign stopped before Stage B.
- Fresh-seed rescue selected trial 4 and passed inner-only Stage A2, but the 15-seed/8-lane blocked Stage B failed the untouched fresh-five held-out gate after `120/120` rows. Trial 4 was more stable and strongly controlled, but not reliably better; grouped Stage C was not run.
- Fixed 50/50 original/Trial-4 fresh-five pilot: `20/20` blocked rows, positive versus original in `5/5` and AR in `5/5`, but only `+0.0001625` over Trial 4 and `7.21%` worse seed variability. Failed; no control-complete follow-up.
- Fresh-15 three-checkpoint retraining: `60/60` rows. Trial 4 averaging stabilized strongly but lost to the matched original ensemble; original ensemble PR-AUC `0.2717155`, `+0.0116782` over AR. Promising comparator, not promoted.

## Next Work

1. Treat same-family hyperparameter tuning as saturated; the canonical original remains the supported single configuration.
2. Preregister a fresh control-complete blocked confirmation of the original three-checkpoint ensemble; do not retrofit the failed Trial-4 gates.
3. Keep continuous arousal work separate from the confirmed binary event-ranking claim.
4. Do not restart broad secondary-head, all-target, or architecture-zoo sweeps without a narrow diagnostic reason.
5. Treat the historical literal 504 matrix as obsolete; do not recreate it or invent rows to reach 504.
6. Keep full multimodal TRIBE as a separate pilot until audio/text-bearing inputs and local model access are resolved.

## Repo Hygiene Rules

- Keep tracked evidence snapshots under `evidence/`.
- Keep current-facing narrative under `README.md`, `docs/current_project_state.md`, and `docs/neural_bridge_phase5_5_evidence_ladder.md`.
- Keep heavyweight outputs, checkpoints, arrays, dense caches, model weights, videos, and external datasets out of git.
- Preserve historical reports as historical artifacts; add supersession notes only when a file is current-facing.
