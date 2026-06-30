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
- 504-scale confirmation. No 504 has been run or promoted.
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

## Next Work

1. Review the Phase 5.5 evidence ladder and decide whether a 504/broader compatibility confirmation is justified.
2. If 504 is approved, define it explicitly before running anything: target, head, controls, gates, seed/fold matrix, and promotion language.
3. Keep continuous arousal work separate from the confirmed binary event-ranking claim.
4. Do not restart broad secondary-head or all-target sweeps without a narrow diagnostic reason.
5. Keep full multimodal TRIBE as a separate pilot until audio/text-bearing inputs and local model access are resolved.

## Repo Hygiene Rules

- Keep tracked evidence snapshots under `evidence/`.
- Keep current-facing narrative under `README.md`, `docs/current_project_state.md`, and `docs/neural_bridge_phase5_5_evidence_ladder.md`.
- Keep heavyweight outputs, checkpoints, arrays, dense caches, model weights, videos, and external datasets out of git.
- Preserve historical reports as historical artifacts; add supersession notes only when a file is current-facing.
