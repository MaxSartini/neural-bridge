# Neural Bridge Current-State Handoff

Updated: 2026-07-17  
Branch: `codex/veatic21-retraining-foundation`

## Current objective

Finish the privileged VEATIC 2.1 programme before combined VEATIC+AGAIN training or another zero-label campaign: stabilize arousal spike/event first, develop continuous arousal separately, then valence rise/drop/absolute movement and derived direction. Outer-test confirmation remains closed until an inner-only candidate is stable and preregistered.

## Active VEATIC result

The newest authoritative inner-only study is `reports/veatic21_again_parity_inner_discovery_20260717.md`, produced at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_again_parity_inner_discovery_20260717` from commit `e54f20e` with run identity `cab3b6d7f67a348e680798012ced523cb498cb2c551b2f6388ca833fb0376732`.

- Matrix/audit: exactly `270/270` member rows plus `90/90` fixed three-checkpoint ensemble rows; all provenance, PCA, quality, label, zero-event, eval-mode, and outer-closure checks passed.
- Best ensemble: `temporal_mean_2s_pca256_again_clean_joint`, PR-AUC `0.3183913271` versus equally ensembled AR `0.3127116235`; delta `+0.0056797035` (`+1.82%`), `11/15` panel wins, positive median `+0.0060286143`, and `4/5` positive outer-fold means.
- Its individual checkpoints were unstable: real `0.3005580415` versus AR `0.3087399646`; delta `-0.0081819231`, `21/45` wins, and `1/5` positive outer-fold means.
- The ensemble passed its own credibility gate, but the preregistered combined member-plus-ensemble gate failed. `any_credible_recipe: false`; this is an honest partial numerical win, not a promotion or contract failure.
- Runner-up `delta_pca64_again_clean_joint`: ensemble delta `+0.0018407640`, `10/15` wins, `3/5` positive outer-fold means.
- Training was fair and overfit-protected: batch `1024`, minimum eligible epoch `50`, patience `100`, max `5000` runaway fail-safe; no run reached the ceiling. Residual best epochs reached `613`, AR best epochs `646`; only `18/270` residual cells selected exact zero correction.
- Exploratory continuous diagnostic: the event-winning ensemble lost Spearman by `-0.0123756758` and top-5% lift by `-0.0014342785` versus AR. Do not force one recipe across event, continuous, and valence.

The prior no-harm port reduced the corrected six-recipe deficit from `-0.0068741007` to `-0.0026761898`; the parity ensemble then crossed AR. These are system-generation comparisons, not isolated ablations.

## Locked substrate and invariants

- VEATIC has exactly `124` videos and `20,657` dense 2 Hz rows. Fresh compact caches: `$NEURAL_BRIDGE_EXTERNAL_ROOT/cache/veatic_h100_vjepa21_compact_20260716` and `$NEURAL_BRIDGE_EXTERNAL_ROOT/cache/veatic_h100_tribe_v2_mlx_compact_20260716`.
- Apply the locked black/high-duplicate mask consistently: `923` unusable causal-window rows excluded. Across the 15 panels, `95/496` repeated video-panel appearances had zero events; keep their valid negatives in pooled PR-AUC and never fabricate per-video zero scores.
- Use only fresh VEATIC 2.1 fold-safe PCA, labels, AR models, heads, and weights. AGAIN methods may enter as architecture/training priors only; do not reuse fitted AGAIN data or artifacts.
- The three numerical executor integrations and restart/audit contracts are implemented and smoke-tested. Canonical training has not started; no new TRIBE cache is required.
- Preserve the locked AGAIN 299-video pool. `video_supervised_temporal` remains the frozen AGAIN comparator.

## Next executable decision

Preregister a bounded inner-only Optuna stabilization study on the clean temporal-mean-2s PCA256 event family. Keep target/window, quality and zero-event policies, dual frozen AR, full-dense training, overfit protection, and equal checkpoint averaging fixed.

Tune only residual-learning parameters such as learning rate, hidden width, binary-loss weight, alpha cap/init, gate bias, weight decay, and possibly batch size. Optimize a robustness objective rather than peak mean PR-AUC: reward mean/median delta and broad panel wins; penalize variance and bad worst-fold behavior.

Generate candidates on a strict subset of inner panels, freeze exactly one candidate before held-back scoring, then run a fresh-seed tuned-versus-original-versus-AR showdown using equal five-checkpoint ensembles across all 15 inner panels. The tuned model must beat both AR and the untuned original with positive mean/median, broad panel and fold consistency, no worse variability, and positive ensemble uplift. Outer tests stay unopened. Historical AGAIN Optuna results show why fresh-seed comparison is mandatory.

If tuning fails, retain the original ensemble and compare clean temporal PCA128, delta PCA256, and delta PCA64 structural candidates before further hyperparameter search. After event stabilization, start separate continuous discovery; then valence.

## Canonical references

- Current VEATIC report: `reports/veatic21_again_parity_inner_discovery_20260717.md`
- VEATIC parity preregistration: `docs/veatic21_again_parity_arousal_event_preregistration_20260717.md`
- Discovery history and AGAIN method evolution: `docs/how_neural_bridge_was_discovered.md`
- Current Phase 7 evidence: `docs/neural_bridge_phase7_evidence.md`
- Current zero-label evidence: `docs/neural_bridge_zero_label_deployment_evidence.md`
- Executable validation index: `docs/executable_validation_index.md`
- Reviewer dossier: `evidence/current_review/README.md`

## Stable AGAIN ceiling

Phase 7 grouped continuous remains the observed-arousal-assisted research ceiling: `420/420`, failed gates `[]`, real/AR/best-control Spearman `0.2603011121/0.2405371348/0.2402523335`, top-5% lift `0.0975979581/0.0895663763/0.0897088493`, positive `15/15` fold-groups and `5/5` fold means. The locked zero-label direct-supervised lane remains a separate cached-feature AGAIN win: `140/140`, real Spearman/top-5%/event PR-AUC `0.1785132961/0.0766079674/0.1710622218`, with every required endpoint and bootstrap gate passed.

## Validation and handoff

Latest repository verification passed readiness, compilation, `397` tests with one skip, strict VEATIC benchmark dry-run, and frontend production build. For routine handoff, update this file only after canonical evidence is validated, mine only `docs/handoff/` into the `neural_bridge` MemPalace wing, verify one top-3 recall, and keep both repositories clean.
