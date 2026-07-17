# Neural Bridge Agent Guide

## Source Rules

- Use current repo/workspace files and current user prompts as benchmark truth.
- Do not use prior chat memory, Claude/VS Code state, old plans, compacted context, or Spark-era outputs as authority.
- `codebase-memory-mcp` is for code navigation only, not benchmark authority.
- Keep context use compact. Do not dump full reports, logs, or generated metadata unless asked.
- Claim boundaries are evidence-state markers, not permanent bans. When a new valid executable result passes its gates, update the canonical files and state the win plainly; never call a passed result a failure merely because older documentation was cautious. Likewise, do not promote a failed gate without new valid evidence. Current executable evidence outranks stale wording in either direction.

## Continuity And Context Efficiency

- Use `AGENTS.md` and the current canonical artifacts below as the small always-loaded handoff.
- Use MemPalace only when a task depends on earlier conversations, decisions, paths, or unfinished work. Recall the smallest relevant set (normally 3-5 results); never inject or summarize whole transcripts by default.
- For routine handoff, never run a full-project or full-evidence MemPalace mine. Update `docs/handoff/CURRENT_STATE.md` after canonical files are committed and validated, mine only `docs/handoff/` into the `neural_bridge` wing, then verify it with one top-3 recall query. This is the default seconds-to-one-minute path.
- Reserve a full MemPalace corpus mine for an explicitly authorized structural rebuild of the evidence corpus or conversation archive. Scope it to the changed directory where possible; do not block ordinary result/documentation handoff on re-embedding the historical evidence tree.
- Keep `codebase-memory-mcp` for structural code discovery and call/path tracing. MemPalace history and its entity graph do not replace the code knowledge graph.
- Current files and executable evidence override MemPalace whenever they conflict. Save durable claim changes to canonical repo files before treating historical memory as updated.
- Use RTK-compressed shell output and targeted snippets to avoid spending context on raw logs or large files.
- After substantive code changes, refresh the internal and external `codebase-memory-mcp` projects before handoff. Do not pay the reindex cost for conversation-only or documentation-only turns.
- Only mark the MemPalace wing fresh after canonical files are updated, validation is complete, relevant graph indexes are refreshed, and both repositories are clean. A Git-head or dirty-worktree mismatch means recalled project status is stale until verified.

## Current Claim

Canonical claim: Neural Bridge demonstrates controlled future human arousal event-ranking across VEATIC and AGAIN, plus controlled grouped held-out-video continuous future-arousal movement ranking/lift on AGAIN, from frozen predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data.

VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal. AGAIN replicated, scaled, validated, and strengthened it using 995 videos, 2 Hz dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

The frozen video-side features are predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data. Treat them as predicted neuro-response video features, not generic video embeddings and not direct neural recordings from the benchmark viewer rows.

Raw predicted cortical/fMRI features alone fail badly on AGAIN. On the original Phase 3 spike target `arousal_spike_rows_2_6_train_q90`, blocked `raw_cortical_only` PR-AUC was `0.124315` versus AR-only `0.203622`, and direct `AR_plus_raw_cortical` dropped to `0.167731`. Neural Bridge is the current result, not raw predicted cortical/fMRI features by themselves.

Beating AR is the core technical hurdle. AR is recent/past arousal persistence, and it is intentionally strong. The current blocked result beats matched seed-specific frozen AR by `+0.0068399399` PR-AUC (`+2.63%` relative lift) and the best matched control by `+0.0077366579` PR-AUC (`+2.98%` relative lift), with `9/10` positive seeds vs both. Updated grouped compatibility beats matched fold/seed-specific AR/frozen by `+0.0138878634` PR-AUC (`+6.39%` relative lift) and the best matched control by `+0.0139621972` PR-AUC (`+6.42%` relative lift), with `50/50` fold-seed positives vs the best matched control.

Do not frame the Phase 7 result as “only 8% better.” Its `8.22%` Spearman and `8.97%` top-5% lifts are the difficult residual gain over a trained target-specific AR model that already captures the dominant persistence signal. On the early same-target raw ablation, raw cortical-only was `38.95%` below trained AR and direct AR-plus-raw remained `17.63%` below AR. The bridge's value is the reversal from negative incremental value to positive signal in every `15/15` Phase 7 fold-group. Do not collapse the early PR-AUC and later continuous metrics into one cross-task percentage; state the valid qualitative transformation and the within-protocol numbers separately.

For current-vs-original system value, use the like-for-like stored grouped continuous metrics: Phase 7 versus the original validated Phase 5 eval-mode bridge is `+16.61%` Spearman (`0.2232222830` → `0.2603011121`), `+23.59%` top-5% lift (`0.0789694843` → `0.0975979581`), and `+14.52%` top-1% lift (`0.1359465244` → `0.1556892559`). The top-5% margin over AR grew `+98.92%`. Label this a full system-generation comparison, not a controlled single-component ablation.

Also convey spike/event progress. On the original grouped spike target, raw cortical-only `0.136579` progressed to the Phase 5 frozen-AR residual bridge at `0.2383409298` (`+74.51%`), which was `+39.95%` above direct AR-plus-raw. The later fresh grouped redesigned-target binary ensemble reached `0.2343675680` vs AR `0.2180497906`, won `15/15`, and increased the bridge margin over AR by `17.50%` versus the promoted single model. Phase 7's continuous prediction has supporting event PR-AUC `0.2231895329` vs AR `0.2088047413` and strongest control `0.2096090680`, positive `15/15`; state explicitly that this is a secondary stored metric, not the primary Phase 7 gate.

Terminology: `AR-only baseline` is a standalone autoregressive comparison lane. `Frozen AR` is a seed- or fold-specific AR-only score/logit fixed before residual/control training and reused identically across real and matched controls inside that seed/fold. It may be reused from an existing compatible AR checkpoint/score or newly trained for that exact seed/fold when missing.

More precisely, trained AR uses observed current arousal, lag-1/2/4 arousal, and recent deltas and is fitted for the exact target/fold/seed with training-side inner validation. `AR_plus_raw_cortical` is direct feature concatenation and previously damaged the AR path. Neural Bridge instead learns a fold-safe causal residual correction over the frozen AR score; the Phase 7 ensemble averages three independently trained prespecified residual checkpoints.

Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`. Grouped held-out-video compatibility for the same target/head is proven under the updated frozen-AR-residual-aware verdict.

The full bounded selected-head confirmation is now assembled and promoted: exactly `420/420` scored rows (`70/70` blocked plus `350/350` grouped) passed matrix completeness, uniqueness, target/head/window, split/PCA provenance, frozen-AR score-cache identity, executable control-policy, eval-mode checkpoint, canonical blocked-gate, and updated grouped-gate audits. All `420` rows were provenance-compatible reuse; rerun count was `0` and failed gates were `[]`.

The original three-checkpoint ensemble is now promoted under both bounded protocols. Its fresh control-complete blocked confirmation passed `140/140` rows, and its separate fresh grouped-video confirmation passed `420/420` rows (`315` member plus `105` ensemble). In the grouped confirmation, real ensemble PR-AUC was `0.2343675680` versus AR `0.2180497906` and best matched control `0.2179716645`; all `15/15` fold-groups and `5/5` fold means were positive versus both, ensembling added `+0.0082200727` over the 45 real-member mean, and failed gates were `[]`.

The deterministic Phase 5 eval-mode `regression_plus_binary` lane also passed its grouped continuous-ranking/lift gate. Across 15 grouped fold-seed evaluations, real future-movement Spearman was `0.2232222830` versus AR-only `0.1982207591`, `ar_plus_shuffled_pca` `0.1938183619`, and `ar_plus_random_pca` `0.1931781163`; real top-1% average-true-movement lift was `0.1359465244` versus `0.1115815364`, `0.1125842464`, and `0.1136304212`, respectively. This proves controlled grouped continuous future-movement ranking/lift for that lane. It does not prove calibrated exact continuous values.

Phase 7 then applied the selected washout continuous target, proven short-temporal-conv recipe, and fixed three-checkpoint averaging in a separately preregistered grouped held-out-video validation. It passed the full `420/420` matrix (`315` member plus `105` ensemble) with failed gates `[]`. Real ensemble Spearman was `0.2603011121` versus target-specific AR `0.2405371348` and best control `0.2402523335`; real top-5% lift was `0.0975979581` versus `0.0895663763` and `0.0897088493`. Real-minus-AR / best-control was `+0.0197639773` / `+0.0200487786` Spearman and `+0.0080315818` / `+0.0078891089` top-5% lift. All `15/15` fold-groups and `5/5` fold means were positive for both metrics versus both comparators. Ensembling added `+0.0077966938` Spearman and `+0.0025021192` top-5% lift over members. This is the authoritative current Phase 7 result and promotes grouped continuous future-movement ranking/lift for `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`.

The video-only deployment bridge is now separately confirmed. The prespecified direct-supervised temporal lane was selected on development data, then prospectively locked before opening the 299-video pool. The locked run completed `140/140` rows on MLX `Device(gpu, 0)`, passed every target/split/PCA/prediction-seal/inference audit, and failed no Tier 1 gates. With no observed-arousal input at inference, real Spearman / top-5% lift / event PR-AUC were `0.1785132961` / `0.0766079674` / `0.1710622218` versus the metric-specific strongest false-signal/no-video controls `0.1004882655` / `0.0448520122` / `0.1352295369`: relative gains `+77.65%` / `+70.80%` / `+26.50%`. Every endpoint won `5/5` full-video panels, every paired whole-video bootstrap lower bound was positive, and the first-30-second cold-start tier passed. Treat this as a locked cached-feature zero-label-at-inference win on AGAIN. Development-path details live in the Stage A report and discovery history, not the current claim.

Deployment boundary: Phase 7 consumes observed current/past arousal through `arousal`, lag-1/2/4, and recent-delta features and remains the privileged research ceiling. The locked direct-supervised lane removes those inputs and validates cold-start cached-feature zero-label inference on AGAIN. It is supervised during training. It does not yet validate end-to-end raw-video feature generation/runtime, external/cross-domain zero-label transfer, prospective client outcomes, or calibrated exact-value prediction.

Not currently proven by the evidence below. These are current boundaries, not permanent prohibitions; remove or revise an item immediately when new valid executable evidence proves it:

- exact continuous future arousal forecasting is solved
- end-to-end raw-video runtime or external/client deployment validity is established
- broad all-target/all-dataset temporal prediction is solved
- treat Phase 5b/5c/Spark/max-capacity/deep/chimera outputs as canonical
- use `holy_shit_pass` as a valid gate

Canonical adversarial review:
`docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`

Canonical deterministic eval-mode correction report:
`reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`

Canonical frozen-AR residual report:
`reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`

Canonical current Phase 7 evidence:
`docs/neural_bridge_phase7_evidence.md`

Canonical zero-label deployment evidence:
`docs/neural_bridge_zero_label_deployment_evidence.md`
`reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md`
`evidence/zero_label_video_only_direct_supervised_locked_confirmation_20260715/README.md`

Historical Phase 5.5 evidence ladder:
`docs/neural_bridge_phase5_5_evidence_ladder.md`

Canonical discovery and washout-gap evolution:
`docs/how_neural_bridge_was_discovered.md`

Canonical reviewer evidence dossier:
`evidence/current_review/README.md`

Canonical executable validation index:
`docs/executable_validation_index.md`

Canonical blocked temporal binary confirmation:
`reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`

Canonical updated grouped compatibility verdict:
`reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`

Canonical full bounded selected-head confirmation:
`reports/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953.md`

Canonical Phase 7 grouped confirmation:
`reports/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440.md`

Current bottom line, newest first: the direct-supervised video-only bridge passes a locked 299-video `140/140` zero-label-at-inference confirmation across continuous, top-5%, and event endpoints. Phase 7 is the observed-arousal-assisted research ceiling with an authoritative grouped `420/420` / `15/15` pass and failed gates `[]`. Phase 6 checkpoint stabilization and the Phase 5.5 binary selected-head confirmations remain promoted foundations; VEATIC established the original cross-domain phenomenon. Exact continuous values, end-to-end raw-video runtime, external zero-label transfer, and broad universal prediction remain open.

## Commercial Language

Neural Bridge should be described as a Service-as-Software product direction for neuro-response video intelligence: software intended to automate the first-pass expert service of pre-release video response evaluation, creative diagnostics, variant comparison, and response-readiness reporting using predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data.

Keep product vision separate from validated deployment state. Cached-feature zero-label inference is validated on locked AGAIN videos. End-to-end raw-video operation, external/client validity, calibrated outputs, and production reliability remain to be proved.

Use bounded commercial wording: population-level response-event ranking, pre-release response intelligence, controlled future arousal event signal, creative decision support, and Service as Software for response evaluation.

Do not frame Neural Bridge as generic SaaS, a simple dashboard, mind reading, individual profiling, medical inference, exact continuous arousal solved, universal emotion prediction, guaranteed campaign outcomes, or a replacement for editors/researchers.

## Project Ownership And Dependency Boundary

Neural Bridge is sole-founder work. Treat the Neural Bridge-specific ideas, target and washout design, bridge architecture, implementation, experiments, controls, training and evaluation procedures, trained heads, derived caches, evidence packages, reporting logic, and product surfaces built across VEATIC and AGAIN as the founder's work. Do not attribute that IP to the benchmark datasets or to the upstream encoders.

V-JEPA, TRIBE, and the source datasets are third-party upstream dependencies or research inputs governed by their own licenses and terms. They are replaceable inputs to Neural Bridge, not the commercial output and not part of the product payload. VEATIC and AGAIN are used for training, benchmarking, and evidence; the eventual product does not ship those source datasets. Generated caches and post-encoder artifacts created by the Neural Bridge project are project assets, subject to any applicable upstream and dataset terms.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence/phase_0_to_5_historical_ladder_20260625/`
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence/phase_5_1_frozen_ar_residual/`
- Current reviewer dossier: `evidence/current_review/`
- Historical detailed dossier: `evidence/current_phase_5_5_review/`
- Blocked binary confirmation evidence snapshot: `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- Grouped compatibility evidence snapshot: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`
- Selected-head 420 confirmation output root: `outputs/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953/`
- Selected-head 420 confirmation evidence snapshot: `evidence/phase_5_5_selected_head_420_confirmation_20260714_124953/`
- Original three-checkpoint grouped confirmation output root: `outputs/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024/`
- Original three-checkpoint grouped confirmation evidence snapshot: `evidence/phase_6_original_three_checkpoint_grouped_confirmation_20260714_163024/`
- Phase 7 grouped continuous output root: `outputs/again_dense_2hz_phase7_continuous_checkpoint_ensemble_grouped_20260714_181440/`
- Zero-label direct-supervised locked output root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_zero_label_direct_supervised_locked_confirm_20260715/`
- Zero-label direct-supervised evidence snapshot: `evidence/zero_label_video_only_direct_supervised_locked_confirmation_20260715/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Do not force-add ignored output roots.

## Current Numbers

AGAIN locked zero-label direct-supervised confirmation:

- target: `future_arousal_max_delta_rows_4_10` plus training-q90 event endpoint
- training / prospectively locked videos: `696` / `299`
- matrix: `140/140` (`105` member plus `35` ensemble)
- real / strongest-control Spearman: `0.1785132961` / `0.1004882655`; delta `+0.0780250306` (`+77.65%`)
- real / strongest-control top-5% lift: `0.0766079674` / `0.0448520122`; delta `+0.0317559552` (`+70.80%`)
- real / strongest-control event PR-AUC: `0.1710622218` / `0.1352295369`; delta `+0.0358326849` (`+26.50%`)
- full-video panel wins: `5/5` on every required endpoint
- one-sided whole-video bootstrap lower 95%: `+0.0606787212` / `+0.0187740072` / `+0.0235455194`
- first-30 cold-start tier: passed
- primary-minus-current-row: `+0.0111227287` Spearman / `+0.0114229070` top-5% / `+0.0075797964` event PR-AUC
- Tier 1 / Tier 2 / Tier 3: true / true / true; failed Tier 1 gates: `[]`
- inference uses no observed arousal; training is supervised; current scope is cached-feature AGAIN confirmation

AGAIN Phase 7 grouped continuous checkpoint ensemble:

- target/head: `residual_future_max_delta_rows_4_10` / `short_temporal_conv_residual`
- matrix: `420/420` (`315` member plus `105` ensemble) over five grouped-video folds, nine untouched seeds, and three fixed groups
- real / target-specific AR / best-control Spearman: `0.2603011121` / `0.2405371348` / `0.2402523335`
- delta vs AR / best control Spearman: `+0.0197639773` / `+0.0200487786`
- real / AR / best-control top-5% lift: `0.0975979581` / `0.0895663763` / `0.0897088493`
- delta vs AR / best control top-5% lift: `+0.0080315818` / `+0.0078891089`
- fold-group wins vs AR / best control: `15/15` / `15/15` for Spearman and `15/15` / `15/15` for top-5% lift
- positive fold means: `5/5`
- ensemble uplift over members: `+0.0077966938` Spearman / `+0.0025021192` top-5% lift
- grouped continuous ranking/lift pass: true; failed gates: `[]`
- calibrated exact-value forecasting: not promoted

AGAIN original three-checkpoint blocked ensemble confirmation:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- matrix: `140/140` rows over 15 untouched seeds and five fixed groups
- real / AR / best-control PR-AUC: `0.2668905427` / `0.2597235728` / `0.2589301730`
- delta vs AR / best control: `+0.0071669699` / `+0.0079603697`
- positive groups vs AR / per-group best control: `5/5` / `5/5`
- ensemble uplift over 15 real members: `+0.0057164681`, positive `5/5`
- blocked control-complete pass: true; failed gates: `[]`

AGAIN original three-checkpoint grouped ensemble confirmation:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- matrix: `420/420` rows (`315` member plus `105` ensemble) over five grouped-video folds, nine untouched seeds, and three fixed groups
- real / AR / best-control PR-AUC: `0.2343675680` / `0.2180497906` / `0.2179716645`
- best aggregate matched control: `train_only_video_mean_residual`
- delta vs AR / best control: `+0.0163177774` / `+0.0163959035`
- positive fold-groups vs AR / per-fold-group best control: `15/15` / `15/15`
- positive fold means vs AR / best control: `5/5` / `5/5`
- ensemble uplift over 45 real members: `+0.0082200727`, positive `15/15`
- grouped control-complete pass: true; failed gates: `[]`

AGAIN full bounded selected-head confirmation:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- matrix completeness: `420/420` (`70/70` blocked, `350/350` grouped)
- rows reused / rerun: `420` / `0`
- provenance, frozen-AR checksum identity, executable control policies, and checkpoint restoration: pass
- canonical blocked confirmation and updated grouped compatibility: pass
- overall selected-head confirmation pass: true
- failed gates: `[]`

AGAIN blocked temporal binary confirmation:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- matched seed-specific frozen AR PR-AUC: `0.2602336231`
- best matched control `random_pca_residual` PR-AUC: `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best matched control: `+0.0077366579`
- seeds positive vs frozen AR and best matched control: `9/10`, `9/10`
- weak / credible / strong confirmation: true
- failed gates: `[]`

AGAIN updated grouped compatibility:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- real PR-AUC: `0.2313831909`
- matched fold/seed-specific AR/frozen PR-AUC: `0.2174953276`
- best matched control `train_only_video_mean_residual` PR-AUC: `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best matched control: `+0.0139621972`
- fold-seed positives vs best matched control: `50/50`
- real minus label permutation: `+0.0160732134`
- label permutation minus AR: `-0.0021853501`
- updated grouped compatibility pass: true

AGAIN dense substrate:

- `995/995` videos complete, `243,575` row-level video feature rows generated from video by upstream models trained on brain cortical response data, true `2 Hz` labels, `256 px`, float16, official V-JEPA 2.1 ViT-G, TRIBE v2 cache-only postpass

AGAIN eval-mode and frozen-AR design path:

- eval-mode grouped real `regression_plus_binary` PR-AUC: `0.2300639382`
- eval-mode grouped best matched control `ar_plus_shuffled_pca` PR-AUC: `0.2042740689`
- eval-mode grouped real-minus-control delta: `+0.0257898694`
- eval-mode grouped fold-seed delta: positive in `15/15`
- eval-mode grouped continuous ranking/lift pass: true
- eval-mode grouped future-movement Spearman real / AR-only / shuffled / random: `0.2232222830` / `0.1982207591` / `0.1938183619` / `0.1931781163`
- eval-mode grouped top-1% average-true-movement lift real / AR-only / shuffled / random: `0.1359465244` / `0.1115815364` / `0.1125842464` / `0.1136304212`
- continuous boundary: grouped future-movement ranking/lift passed; calibrated exact continuous values are not claimed
- frozen-AR residual grouped frozen AR PR-AUC: `0.2246816187`
- frozen-AR residual grouped best real residual PR-AUC: `0.2383409298`
- frozen-AR residual grouped matched control PR-AUC: `0.2248361805`
- frozen-AR residual grouped delta vs frozen AR: `+0.0136593110`
- frozen-AR residual grouped delta vs matched control: `+0.0135047493`
- frozen-AR residual do_no_harm_blocked_pass: yes
- frozen-AR residual full_forward_time_pass: no for the older rows 2-6 lane

VEATIC-124 v2:

- strongest blocked full-frame spike row `cortical_pca64_delta` / `arousal__future_spike_1_3s` PR-AUC: `0.2536`
- same-row AR/shuffled/random controls: `0.1969` / `0.1840` / `0.1944`
- balanced event-vs-stable `arousal__future_spike_1_3s@0.05` PR-AUC: `0.3394`
- balanced deltas over AR/shuffled/random: `+0.0609` / `+0.0631` / `+0.0476`

VEATIC 2.1 corrected-depth arousal-event inner discovery:

- scope: full-dense inner-only development, `270/270` rows = five outer partitions × three inner folds × six recipes × three seeds
- matrix execution/audit: passed; outer-test scores used: false; explicitly nonpromotable: true; confirmation authorized: false
- `canonical_gates_passed: false` is expected because no confirmation stage ran; it is not the numerical verdict
- unanimous recipe winner: `delta_pca64_short_conv`, selected `5/5` outer panels
- real / matched freshly trained frozen AR PR-AUC: `0.3014044849` / `0.3082785856`
- delta vs AR: `-0.0068741007` (`-2.23%` relative); paired median delta `-0.0059385296`
- paired positives vs AR: `10/45`; outer-panel mean positives: `0/5`
- training: batch `1,024`, minimum epoch `50`, patience `100`, 5,000-epoch runaway fail-safe, binary checkpoint selection by inner-validation PR-AUC
- cap hits: `0`; video best epoch range `50`–`734`; AR best epoch range `50`–`945`
- interpretation: execution passed and recipe selection was decisive, but Neural Bridge lost to AR; arousal spike/event is not cracked
- boundary: fresh VEATIC PCA/AR/heads/thresholds only; no fitted AGAIN models and no raw-cortical direct concatenation

## Next Task

The AGAIN deployment bridge is complete. The direct-supervised temporal lane is frozen after its prospectively locked 299-video `140/140` pass. Development-path details belong in `docs/how_neural_bridge_was_discovered.md` and the Stage A report, not current result summaries.

The highest-value next milestone is the bounded VEATIC 2.1 programme itself, before either the combined cross-domain model or another zero-label campaign. Follow a sequential inner-only discovery ladder unless executable inner-validation evidence supports one shared method without compromise: first crack the arousal train-q90 spike/event problem, transfer the resulting target/head/feature/training insight into continuous arousal, then adapt the proven mechanism to valence rise/drop/absolute movement and derived direction. Do not force one recipe across these endpoints; selections remain target/protocol specific. A shared or multitask candidate may enter only as a preregistered inner-discovery recipe and must earn its place before any outer-test confirmation. Zero-label VEATIC remains a later deployment bonus and must not drive the immediate recipe or claim priority. Only after the privileged VEATIC result is complete should the project start the harmonized VEATIC+AGAIN pilot with domain-balanced sampling and leave-one-domain-out evaluation. Keep `video_supervised_temporal` frozen as the AGAIN comparator and never tune on the locked 299-video pool.

The authoritative corrected-depth arousal-event inner discovery is complete: `270/270` full-dense rows, no outer-test scores, explicitly nonpromotable, and confirmation unauthorized. Matrix execution and audit passed; `canonical_gates_passed: false` is expected because that summary field is sourced only from a confirmation stage, which this bounded discovery scope deliberately did not run. Numerically, Neural Bridge lost to AR. `delta_pca64_short_conv` won the six-recipe comparison in all `5/5` outer development panels, but scored `0.3014044849` versus matched freshly trained frozen AR `0.3082785856`: `-0.0068741007` absolute, `-2.23%` relative, paired median `-0.0059385296`, positive only `10/45`, and negative on mean in all five panels. This was a real frozen-AR residual bridge using fresh VEATIC fits, not raw-cortical concatenation and not fitted AGAIN models. No run hit the 5,000-epoch fail-safe; video best epochs ranged `50`–`734` and AR ranged `50`–`945`, so more epochs is not the next fix. The residual scale remained near-closed around `0.00310`, and `203/270` video checkpoints selected the first eligible epoch. Do not open outer confirmation or start continuous. Preregister a focused inner-only two-stage residual-learning branch around `delta_pca64_short_conv`: learn the cortical head explicitly against frozen-AR errors with an open learning path, then fit a train-only bounded correction coefficient that may choose zero. Keep AR, delta PCA256, current-row PCA256, and the current joint-gate method as bounded comparators. Exact report: `reports/veatic21_arousal_event_first_six_recipe_inner_discovery_20260717.md`.

Phase 7 continuous status is now fixed by the authoritative grouped `420/420` validation: every gate passed, with `15/15` fold-groups positive versus both AR and controls on Spearman and top-5% lift. Current-facing summaries must lead with that win and its exact scope.

The approved full bounded 420-row selected-head confirmation remains canonical. Optuna and checkpoint-combination development history belongs in `docs/how_neural_bridge_was_discovered.md` and the Phase 6 reports; it is not the current claim.

Seed `20260627` dominated the mean failure: its canonical original beat tuned by `+0.0178629568`. The stored original curve has an unusually favorable `+0.0272760719` inner-validation peak at epoch 14, roughly `73%` above the original runs' median best peak, then declines. A post-hoc 80-epoch diagnostic reproduced the tuned score exactly, so tuned under-training is not the explanation. Do not delete the seed or relabel the preregistered verdict as a pass.

The staged `docs/phase6_robust_optuna_720_plan.md` stopped fail-closed after Stage B. Its original Stage A failed; a pre-held-out sensitivity analysis identified trial 4, and Stage A2 then passed on five fresh inner-validation seeds. Stage B completed `120/120` blocked rows across 15 seeds and 8 lanes. Trial 4 was slightly higher than original on all-seed mean/median (`+0.0000512741` / `+0.0002953952`) and reduced seed-level PR-AUC standard deviation by `19.41%`, but the untouched fresh-five panel failed (`-0.0013756950` mean, `3/5` wins) and single-seed contribution exceeded its cap. It remained positive versus AR and controls in `15/15`. The planned `600` grouped rows were not authorized or run; do not claim 720.

Stage A2 passed on five new inner-validation seeds, authorizing Stage B. Stage B then completed `120/120` blocked rows across 15 seeds and 8 lanes but failed: all-seed candidate-minus-original mean/median were `+0.0000512741` / `+0.0002953952` with `10/15` wins, while the untouched fresh-five panel was `-0.0013756950` mean with `3/5` wins. Trial 4 nevertheless beat AR and best controls in `15/15`, with `+0.0074160357` and `+0.0085749406` mean deltas. Stage C's 600 grouped rows were not authorized or run; do not claim 720.

Trial 4 reduced seed-level PR-AUC standard deviation by `19.41%` versus original but did not reliably improve fresh held-out performance. This historical result motivated the now-completed locked ensemble/checkpoint-stabilization campaign. Do not do more same-family Optuna tuning, delete stress seeds, or run Stage C.

Do not describe this as a broad seed problem. Most seed-to-seed differences are small. The precise issue is rare favorable-original checkpoint sensitivity: seed `20260627` and later fresh seed `20260636` produced unusually large original-over-Trial-4 gaps, while Trial 4 won `7/10` across fresh seeds `20260635`–`20260644` with a positive paired median but a negative mean dominated by seed `20260636`.

The preregistered fixed 50/50 original/Trial-4 fresh-five pilot then completed `20/20` blocked rows on seeds `20260640`–`20260644`. It beat original in `5/5`, Trial 4 in `3/5`, and AR in `5/5`, but improved over the stronger component by only `+0.0001625462` and increased rather than reduced seed variability by `7.21%`. The pilot failed its locked minimum-gain, median, and stability gates; no matched-control or grouped follow-up is authorized. This rules out the simple within-seed 50/50 blend, not multi-checkpoint seed averaging.

The subsequent larger retraining used 15 untouched seeds in five fixed three-checkpoint groups (`60/60` rows). Trial-4 ensembling improved over its member mean by `+0.0027137975`, reduced variability by `85.89%`, and beat AR in `5/5`, but lost to the equally ensembled original recipe by `-0.0029710659` with only `2/5` wins. The prespecified original ensemble reached `0.2717155074`, `+0.0044814318` over its member mean and `+0.0116782360` over the AR ensemble. The Trial-4 candidate failed; this made the original ensemble the comparator subsequently tested and promoted by fresh blocked and grouped confirmations.

That fresh original three-checkpoint control-complete blocked confirmation passed. Across `140/140` rows on untouched seeds `20260660`–`20260674`, real ensemble PR-AUC was `0.2668905427` versus AR ensemble `0.2597235728` and best aggregate matched control `random_pca_residual` at `0.2589301730`. Real-minus-AR / best-control was `+0.0071669699` / `+0.0079603697`, all `5/5` groups were positive versus both, ensemble uplift over the 15 real members was `+0.0057164681`, and failed gates were `[]`.

The separately preregistered fresh grouped-video confirmation then passed `420/420` rows on seeds `20260675`–`20260683`. Real ensemble PR-AUC was `0.2343675680` versus AR `0.2180497906` and best aggregate matched control `train_only_video_mean_residual` at `0.2179716645`. Real-minus-AR / best-control was `+0.0163177774` / `+0.0163959035`; all `15/15` fold-groups and all `5/5` fold means were positive versus both. Ensembling added `+0.0082200727` over the 45 real members and won `15/15`; failed gates were `[]`. This promotes bounded grouped-video control-complete evidence for the original three-checkpoint ensemble. Do not generalize it beyond the selected target/head or into exact continuous forecasting.

The real upstream Optuna, Polars, MLflow, and SHAP integrations are installed as the `research-tooling` backend extra and documented in `docs/research_tooling_integrations.md`. Run `npm run verify:research-tooling` to exercise all four with verified MLX GPU/MPS hardware. Their runs remain exploratory and cannot promote canonical evidence.

## Test And Script Validation

`npm test` runs the deterministic contract suite: `python3 -m pytest -q tests`. On `2026-07-17`, `npm run verify` passed `391` tests with one skip and also passed repository readiness, strict-benchmark dry run, and the frontend production build. The separate research-tooling verification passed Optuna, Polars, MLflow, and SHAP on MLX `Device(gpu, 0)` / MPS.

Relevant AGAIN and VEATIC v2 runners, tests, benchmark artifacts, and runtime-only tools are indexed in `docs/executable_validation_manifest.csv`. The current review entrypoint is `evidence/current_review/`; the older executable mirror remains under `evidence/current_phase_5_5_review/14_executable_validation_and_code/`. Do not add placeholder smoke tests as validation. Add tests only when they protect a real split, target, leakage, control, manifest, scorer, checkpoint, or claim-boundary contract.

## Code Discovery

Prefer MCP graph tools for code-level lookup: `search_graph`, `trace_path`, `get_code_snippet`. Use `rg` for literal text, docs, configs, and generated artifacts.
