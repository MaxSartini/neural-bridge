# Neural Bridge Current-State Handoff

Updated: 2026-07-17  
Branch: `main`

## Current objective

Finish the privileged VEATIC 2.1 programme before combined VEATIC+AGAIN training or another zero-label campaign: stabilize arousal spike/event first, develop continuous arousal separately, then valence rise/drop/absolute movement and derived direction. Outer-test confirmation remains closed until an inner-only candidate is stable and preregistered.

## Active VEATIC result

The newest authoritative inner-only study is `reports/veatic21_event_optuna_stabilization_20260717.md`, produced at `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/veatic21_event_optuna_stabilization_20260717` with run identity `4b34b06bd2a1fa95975ac88a0bf797a28ebbe1956628dc250f515264dd3249bb`.

Its stored execution commit `e1b0d9f` is immutable provenance, not the current repository authority. The current executor reproduced all audit checks and exact primary metrics from the stored showdown rows; later executor changes affected only default external-root resolution.

- Search completed exactly `50` trials and froze trial `26` before the showdown. Its search objective was `0.0095972467`; parameters were hidden `48`, learning rate `0.0001494773`, weight decay `0.0003062954`, alpha cap `0.04`, alpha initial logit `-4`, gate bias `6`, and binary-loss weight `1.0`.
- The fresh-seed showdown completed exactly `150/150` member rows and `30/30` five-member ensemble rows. All finite-metric, matrix, frozen-AR-sharing, label-digest, zero-event, and outer-closure audits passed; outer-test scores used: `false`.
- Primary held-back 10 panels: tuned minus original mean `-0.0052929689`, median `-0.0002183180`, `5/10` wins, and only `2/5` positive outer means. The tuned candidate was also more variable than original (`0.0195276406` versus `0.0130045156`).
- Against AR on those same panels, tuned gained `+0.0039664034` with `8/10` wins, while original gained `+0.0092593724` with `9/10` wins. Original therefore remained the stronger ensemble.
- Across all 15 panels, tuned minus original was `-0.0022893491`; tuned versus AR was `+0.0043983825`, while original versus AR was `+0.0066877316`. Individual members of both lanes still lost to AR on average; the useful signal remained ensemble-dependent.
- Promotion gates failed. Trial `26` is rejected, the original parity ensemble remains the incumbent inner-only configuration, and no outer confirmation is authorized.

The preceding parity study remains the incumbent source: `temporal_mean_2s_pca256_again_clean_joint` ensemble PR-AUC `0.3183913271` versus AR `0.3127116235`, delta `+0.0056797035`, but unstable individual checkpoints and failed combined member-plus-ensemble gates. The Optuna result now shows that tuning the transferred AGAIN head does not solve that instability.

## Locked substrate and invariants

- VEATIC has exactly `124` videos and `20,657` dense 2 Hz rows. Fresh compact caches: `$NEURAL_BRIDGE_EXTERNAL_ROOT/cache/veatic_h100_vjepa21_compact_20260716` and `$NEURAL_BRIDGE_EXTERNAL_ROOT/cache/veatic_h100_tribe_v2_mlx_compact_20260716`.
- Apply the locked black/high-duplicate mask consistently: `923` unusable causal-window rows excluded. Across the 15 panels, `95/496` repeated video-panel appearances had zero events; keep their valid negatives in pooled PR-AUC and never fabricate per-video zero scores.
- Use only fresh VEATIC 2.1 fold-safe PCA, labels, AR models, heads, and weights. AGAIN methods may enter as architecture/training priors only; do not reuse fitted AGAIN data or artifacts.
- For learned heads, checkpoints are eligible from epoch `1`; early stopping is forbidden before epoch `50`, with patience `100` and maximum epoch `5000`. Epoch `50` is a minimum training depth, not a checkpoint-selection cutoff.
- The Optuna execution is complete; the next VEATIC-specific discovery programme has not started. No new TRIBE cache is required.
- Preserve the locked AGAIN 299-video pool. `video_supervised_temporal` remains the frozen AGAIN comparator.

## Next executable decision

**Build a new VEATIC-2.1 system from fresh 2.1 data, using end-of-AGAIN scientific rigor.**

- OG VEATIC contributes hypotheses only: temporal change may matter more than raw state, simple heads deserve a baseline, and event/spike should be solved before continuous. It contributes no fitted PCA, tensors, labels, models, thresholds, checkpoints, or exact recipe. Linear ridge is a freshly retrained sanity baseline only; PCA64/128 and temporal windows are candidate priors, not inherited truths.
- Use fresh VEATIC-2.1 V-JEPA/TRIBE features, all eligible dense 2 Hz rows, the current quality mask and zero-event policy, stronger target-specific frozen AR, fold-safe feature fitting, and fresh multiscale causal representations selected on VEATIC 2.1.
- Require matched shuffled, random, video-mean, diagnostic, and label-permutation controls. Keep held-out-video and blocked-temporal evidence distinct.
- Separate discovery, candidate freeze, fresh confirmation, and outer closure. Declare checkpoint ensembles prospectively, confirm event before continuous specialization, and use Optuna only after the VEATIC-specific target/representation/head family is established.

## Canonical references

- Incumbent parity report: `reports/veatic21_again_parity_inner_discovery_20260717.md`
- Optuna stabilization report: `reports/veatic21_event_optuna_stabilization_20260717.md`
- Optuna preregistration: `docs/veatic21_event_optuna_stabilization_preregistration_20260717.md`
- VEATIC parity preregistration: `docs/veatic21_again_parity_arousal_event_preregistration_20260717.md`
- Discovery history and AGAIN method evolution: `docs/how_neural_bridge_was_discovered.md`
- Current Phase 7 evidence: `docs/neural_bridge_phase7_evidence.md`
- Current zero-label evidence: `docs/neural_bridge_zero_label_deployment_evidence.md`
- Executable validation index: `docs/executable_validation_index.md`
- Reviewer dossier: `evidence/current_review/README.md`

## Stable AGAIN ceiling

Phase 7 grouped continuous remains the observed-arousal-assisted research ceiling: `420/420`, failed gates `[]`, real/AR/best-control Spearman `0.2603011121/0.2405371348/0.2402523335`, top-5% lift `0.0975979581/0.0895663763/0.0897088493`, positive `15/15` fold-groups and `5/5` fold means. The locked zero-label direct-supervised lane remains a separate cached-feature AGAIN win: `140/140`, real Spearman/top-5%/event PR-AUC `0.1785132961/0.0766079674/0.1710622218`, with every required endpoint and bootstrap gate passed.

## Validation and handoff

Latest repository verification passed readiness, compilation, `400` tests with one skip, strict VEATIC benchmark dry-run, and frontend production build. For routine handoff, update this file only after canonical evidence is validated, index this file and `docs/tokless_agent_workflow.md` into Context-Mode, verify one top-3 search result, and keep the repository clean.
