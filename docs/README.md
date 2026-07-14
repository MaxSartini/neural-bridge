# Neural Bridge Documentation

## Read This First

Phase 7 is the strongest AR-assisted research result, and the locked 299-video
zero-label confirmation is the current deployment headline. Start with:

1. `../README.md` — plain-language and scientific overview.
2. `neural_bridge_zero_label_deployment_evidence.md` — locked 299-video video-only result.
3. `neural_bridge_phase7_evidence.md` — full Phase 7 interpretation and numbers.
4. `current_project_state.md` — compact operating handoff and next task.
5. `current_claim_status.json` — machine-readable claim ledger.
6. `../reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md` — claim-bearing zero-label report.
7. `../reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md` — claim-bearing Phase 7 report.

Phase 7 passed a fresh grouped held-out-video `420/420` matrix. Neural Bridge beat target-specific AR and best matched controls on Spearman and top-5% lift in all `15/15` fold-groups and all `5/5` fold means; failed gates were `[]`.

The larger bridge effect is the headline: original grouped raw cortical `0.136579` became `0.2383409298` (`+74.51%`) under the learned bridge, `+39.95%` above direct AR-plus-raw. Phase 7 then improved the original validated continuous bridge by `+16.61%` Spearman, `+23.59%` top-5% lift, and `+14.52%` top-1% lift, while increasing the top-5% margin beyond AR by `+98.92%`.

In ordinary language: the model is consistently better than recent-response momentum and false-signal controls at identifying where the largest upcoming human-arousal movements will occur on unseen videos.

The video-only lane separately passed `140/140` locked rows on 299 videos with no observed-arousal input at inference. It beat its strongest false-signal/no-video controls by `+77.65%` Spearman, `+70.80%` top-5% lift, and `+26.50%` event PR-AUC, with `5/5` full-video panel wins on every endpoint.

## Current Authority

- `neural_bridge_zero_label_deployment_evidence.md` — current zero-label deployment evidence.
- `neural_bridge_phase7_evidence.md` — current scientific evidence narrative.
- `current_project_state.md` — current state, boundaries, and deployment next step.
- `current_claim_status.json` — structured facts and canonical artifact pointers.
- `neural_bridge_service_as_software.md` — commercial meaning and product path.
- `executable_validation_index.md` — scripts, tests, evidence, and validation commands.
- `executable_validation_manifest.csv` / `.json` — machine-readable executable crosswalk.
- `PROJECT_MEMORY.md` — minimal fresh-session handoff.
- `../AGENTS.md` — agent operating contract.

## Evidence History

- `how_neural_bridge_was_discovered.md` — full discovery sequence from VEATIC through the locked deployment result.
- `neural_bridge_phase5_5_evidence_ladder.md` — historical Phase 5.5 ladder, now superseded as the current ceiling by Phase 7.
- `phase5_selected_head_420_confirmation_plan.md` — completed binary selected-head consolidation protocol.
- `phase6_original_three_checkpoint_grouped_confirmation_plan.md` — completed checkpoint-stabilization confirmation protocol.
- `phase7_continuous_checkpoint_ensemble_preregistration.md` — completed Phase 7 diagnostic protocol.
- `phase7_continuous_checkpoint_ensemble_grouped_preregistration.md` — completed grouped pass protocol.
- `veatic_v2_evidence_summary.md` — foundational VEATIC evidence.
- `again_dense_h100_cache.md` — AGAIN dense substrate handoff.

Historical evidence is preserved because it shows the failures, controls, and design changes that produced the current result. It should not be mistaken for the current performance ceiling.

## Current Claim

Neural Bridge demonstrates controlled future human-arousal event ranking across VEATIC and AGAIN and controlled grouped held-out-video continuous future-arousal movement ranking/lift on AGAIN. Phase 7 independently confirms the continuous result with fixed checkpoint averaging and perfect `15/15` fold-group directional consistency against AR and matched controls.

Exact trajectory values are not claimed. Cached-feature zero-label inference is now confirmed on locked AGAIN videos; end-to-end raw-video runtime, external/cross-domain zero-label confirmation, and prospective client outcomes remain open.

## Next Task

The AGAIN video-only deployment campaign is complete. The direct-supervised temporal lane passed a separately preregistered `140/140` confirmation on the prospectively locked 299-video pool. Read `neural_bridge_zero_label_deployment_evidence.md` before planning follow-up work; method-selection history lives in the Stage A report and discovery narrative.

Next, freeze that method as the comparator, integrate the end-to-end raw-video runtime, and preregister external/cross-domain zero-label confirmation. A bounded V-JEPA 2.1 VEATIC re-encode and balanced VEATIC+AGAIN pilot is now justified, with domain-balanced sampling and leave-one-domain-out evaluation.

## Validation

```bash
npm run verify
npm run audit:repo
npm run verify:research-tooling
```

MLX/MPS is the reference accelerator path. Heavy output roots, caches, checkpoints, arrays, datasets, and model weights remain outside git.
