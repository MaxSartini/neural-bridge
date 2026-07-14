# Neural Bridge Agent Guide

## Source Rules

- Use current repo/workspace files and current user prompts as benchmark truth.
- Do not use prior chat memory, Claude/VS Code state, old plans, compacted context, or Spark-era outputs as authority.
- `codebase-memory-mcp` is for code navigation only, not benchmark authority.
- Keep context use compact. Do not dump full reports, logs, or generated metadata unless asked.

## Continuity And Context Efficiency

- Use `AGENTS.md` and the current canonical artifacts below as the small always-loaded handoff.
- Use MemPalace only when a task depends on earlier conversations, decisions, paths, or unfinished work. Recall the smallest relevant set (normally 3-5 results); never inject or summarize whole transcripts by default.
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

Terminology: `AR-only baseline` is a standalone autoregressive comparison lane. `Frozen AR` is a seed- or fold-specific AR-only score/logit fixed before residual/control training and reused identically across real and matched controls inside that seed/fold. It may be reused from an existing compatible AR checkpoint/score or newly trained for that exact seed/fold when missing.

Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`. Grouped held-out-video compatibility for the same target/head is proven under the updated frozen-AR-residual-aware verdict.

The full bounded selected-head confirmation is now assembled and promoted: exactly `420/420` scored rows (`70/70` blocked plus `350/350` grouped) passed matrix completeness, uniqueness, target/head/window, split/PCA provenance, frozen-AR score-cache identity, executable control-policy, eval-mode checkpoint, canonical blocked-gate, and updated grouped-gate audits. All `420` rows were provenance-compatible reuse; rerun count was `0` and failed gates were `[]`.

The original three-checkpoint ensemble is now promoted under both bounded protocols. Its fresh control-complete blocked confirmation passed `140/140` rows, and its separate fresh grouped-video confirmation passed `420/420` rows (`315` member plus `105` ensemble). In the grouped confirmation, real ensemble PR-AUC was `0.2343675680` versus AR `0.2180497906` and best matched control `0.2179716645`; all `15/15` fold-groups and `5/5` fold means were positive versus both, ensembling added `+0.0082200727` over the 45 real-member mean, and failed gates were `[]`.

The deterministic Phase 5 eval-mode `regression_plus_binary` lane also passed its grouped continuous-ranking/lift gate. Across 15 grouped fold-seed evaluations, real future-movement Spearman was `0.2232222830` versus AR-only `0.1982207591`, `ar_plus_shuffled_pca` `0.1938183619`, and `ar_plus_random_pca` `0.1931781163`; real top-1% average-true-movement lift was `0.1359465244` versus `0.1115815364`, `0.1125842464`, and `0.1136304212`, respectively. This proves controlled grouped continuous future-movement ranking/lift for that lane. It does not prove exact continuous values or blocked continuous generalization: the old fused blocked lane lost to AR/controls, and the later washout continuous diagnostic improved Spearman but failed its full top-5%/seed-consistency gate.

Do not claim:

- exact continuous future arousal forecasting is solved
- broad all-target/all-dataset temporal prediction is solved
- 504 has been run or promoted
- treat Phase 5b/5c/Spark/max-capacity/deep/chimera outputs as canonical
- use `holy_shit_pass` as a valid gate

Canonical adversarial review:
`docs/reviews/neural_bridge_phase5_adversarial_review_20260625.html`

Canonical deterministic eval-mode correction report:
`reports/again_dense_2hz_phase5_evalmode_rescore_summary_.md`

Canonical frozen-AR residual report:
`reports/again_dense_2hz_phase5_frozen_ar_residual_summary_.md`

Canonical Phase 5.5 evidence ladder:
`docs/neural_bridge_phase5_5_evidence_ladder.md`

Canonical discovery and washout-gap evolution:
`docs/how_neural_bridge_was_discovered.md`

Canonical reviewer evidence dossier:
`evidence/current_phase_5_5_review/README.md`

Canonical executable validation index:
`docs/executable_validation_index.md`

Canonical blocked temporal binary confirmation:
`reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`

Canonical updated grouped compatibility verdict:
`reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`

Canonical full bounded selected-head confirmation:
`reports/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953.md`

Current bottom line: VEATIC is foundational, and AGAIN is now the scaled confirmation/current main result. Raw predicted cortical/fMRI features alone were weak; the Neural Bridge pipeline makes the difference. The old fused AGAIN lane passed grouped event and continuous future-movement ranking/lift controls but failed blocked AR/control checks; frozen-AR residual and temporal/event-context residual designs established the current design path. The redesigned washout-gap target/head beats the strong AR persistence baseline, passes blocked temporal confirmation and updated grouped-video compatibility, and now passes the unified bounded `420/420` selected-head confirmation audit. Exact continuous-value forecasting, blocked continuous generalization, and broad universal temporal prediction remain open; do not erase the grouped continuous-ranking/lift pass when stating those boundaries.

## Commercial Language

Neural Bridge should be described as Service as Software for neuro-response video intelligence: software that automates the first-pass expert service of pre-release video response evaluation, creative diagnostics, variant comparison, and response-readiness reporting using predicted cortical/fMRI response features generated from video by upstream models trained on brain cortical response data.

Use bounded commercial wording: population-level response-event ranking, pre-release response intelligence, controlled future arousal event signal, creative decision support, and Service as Software for response evaluation.

Do not frame Neural Bridge as generic SaaS, a simple dashboard, mind reading, individual profiling, medical inference, exact continuous arousal solved, universal emotion prediction, guaranteed campaign outcomes, or a replacement for editors/researchers.

## Canonical Artifacts

- Dense root: `.cache/h100_drive_downloads/again_tribe_v2_postpass_float16_256_2hz/`
- Phase 4 root: `$NEURAL_BRIDGE_EXTERNAL_ROOT/outputs/again_dense_2hz_phase4_pca_bridge_20260625_full/`
- Evidence bundle: `evidence/phase_0_to_5_historical_ladder_20260625/`
- Frozen-AR residual output root: `outputs/again_dense_2hz_phase5_frozen_ar_residual_/`
- Frozen-AR residual evidence snapshot: `evidence/phase_5_1_frozen_ar_residual/`
- Current reviewer dossier: `evidence/current_phase_5_5_review/`
- Blocked binary confirmation evidence snapshot: `evidence/phase_5_5_binary_blocked_confirmation_20260630_025437/`
- Grouped compatibility evidence snapshot: `evidence/phase_5_5_grouped_compatibility_20260630_033520/`
- Selected-head 420 confirmation output root: `outputs/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953/`
- Selected-head 420 confirmation evidence snapshot: `evidence/phase_5_5_selected_head_420_confirmation_20260714_124953/`
- Original three-checkpoint grouped confirmation output root: `outputs/again_dense_2hz_phase6_original_three_checkpoint_grouped_confirmation_20260714_163024/`
- Original three-checkpoint grouped confirmation evidence snapshot: `evidence/phase_6_original_three_checkpoint_grouped_confirmation_20260714_163024/`

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Do not force-add ignored output roots.

## Current Numbers

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
- continuous boundary: grouped future-movement ranking/lift passed; old fused blocked continuous ranking did not beat AR/controls; later washout continuous Spearman improved but the full continuous gate failed
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

## Next Task

The approved full bounded 420-row selected-head confirmation remains canonical. The one-seed Optuna pilot was promising, but its preregistered locked-winner 10-seed confirmation did not establish an aggregate improvement: tuned/original PR-AUC was `0.2659654274` / `0.2670735630`, all-seed delta `-0.0011081356`, and nine-follow-up-seed delta `-0.0014666488`. Tuned beat original in `7/10` seeds and had a positive paired median (`+0.0004281433`), while retaining a controlled bridge win over frozen AR (`+0.0057318043`) and the best tuned matched control (`+0.0067636509`) with `8/10` positive seeds versus both.

Seed `20260627` dominated the mean failure: its canonical original beat tuned by `+0.0178629568`. The stored original curve has an unusually favorable `+0.0272760719` inner-validation peak at epoch 14, roughly `73%` above the original runs' median best peak, then declines. A post-hoc 80-epoch diagnostic reproduced the tuned score exactly, so tuned under-training is not the explanation. Do not delete the seed or relabel the preregistered verdict as a pass.

The staged `docs/phase6_robust_optuna_720_plan.md` stopped fail-closed after Stage B. Its original Stage A failed; a pre-held-out sensitivity analysis identified trial 4, and Stage A2 then passed on five fresh inner-validation seeds. Stage B completed `120/120` blocked rows across 15 seeds and 8 lanes. Trial 4 was slightly higher than original on all-seed mean/median (`+0.0000512741` / `+0.0002953952`) and reduced seed-level PR-AUC standard deviation by `19.41%`, but the untouched fresh-five panel failed (`-0.0013756950` mean, `3/5` wins) and single-seed contribution exceeded its cap. It remained positive versus AR and controls in `15/15`. The planned `600` grouped rows were not authorized or run; do not claim 720.

Stage A2 passed on five new inner-validation seeds, authorizing Stage B. Stage B then completed `120/120` blocked rows across 15 seeds and 8 lanes but failed: all-seed candidate-minus-original mean/median were `+0.0000512741` / `+0.0002953952` with `10/15` wins, while the untouched fresh-five panel was `-0.0013756950` mean with `3/5` wins. Trial 4 nevertheless beat AR and best controls in `15/15`, with `+0.0074160357` and `+0.0085749406` mean deltas. Stage C's 600 grouped rows were not authorized or run; do not claim 720.

Trial 4 reduced seed-level PR-AUC standard deviation by `19.41%` versus original but did not reliably improve fresh held-out performance. The next bounded modeling hypothesis should be a locked ensemble or checkpoint-stabilization method that addresses complementary favorable original/candidate seed behavior. Do not do more same-family Optuna tuning, delete stress seeds, run Stage C, restore 504, or begin continuous development without a new explicit plan.

Do not describe this as a broad seed problem. Most seed-to-seed differences are small. The precise issue is rare favorable-original checkpoint sensitivity: seed `20260627` and later fresh seed `20260636` produced unusually large original-over-Trial-4 gaps, while Trial 4 won `7/10` across fresh seeds `20260635`–`20260644` with a positive paired median but a negative mean dominated by seed `20260636`.

The preregistered fixed 50/50 original/Trial-4 fresh-five pilot then completed `20/20` blocked rows on seeds `20260640`–`20260644`. It beat original in `5/5`, Trial 4 in `3/5`, and AR in `5/5`, but improved over the stronger component by only `+0.0001625462` and increased rather than reduced seed variability by `7.21%`. The pilot failed its locked minimum-gain, median, and stability gates; no matched-control or grouped follow-up is authorized. This rules out the simple within-seed 50/50 blend, not multi-checkpoint seed averaging.

The subsequent larger retraining used 15 untouched seeds in five fixed three-checkpoint groups (`60/60` rows). Trial-4 ensembling improved over its member mean by `+0.0027137975`, reduced variability by `85.89%`, and beat AR in `5/5`, but lost to the equally ensembled original recipe by `-0.0029710659` with only `2/5` wins. The prespecified original ensemble reached `0.2717155074`, `+0.0044814318` over its member mean and `+0.0116782360` over the AR ensemble. The Trial-4 candidate failed; this made the original ensemble the comparator subsequently tested and promoted by fresh blocked and grouped confirmations.

That fresh original three-checkpoint control-complete blocked confirmation passed. Across `140/140` rows on untouched seeds `20260660`–`20260674`, real ensemble PR-AUC was `0.2668905427` versus AR ensemble `0.2597235728` and best aggregate matched control `random_pca_residual` at `0.2589301730`. Real-minus-AR / best-control was `+0.0071669699` / `+0.0079603697`, all `5/5` groups were positive versus both, ensemble uplift over the 15 real members was `+0.0057164681`, and failed gates were `[]`.

The separately preregistered fresh grouped-video confirmation then passed `420/420` rows on seeds `20260675`–`20260683`. Real ensemble PR-AUC was `0.2343675680` versus AR `0.2180497906` and best aggregate matched control `train_only_video_mean_residual` at `0.2179716645`. Real-minus-AR / best-control was `+0.0163177774` / `+0.0163959035`; all `15/15` fold-groups and all `5/5` fold means were positive versus both. Ensembling added `+0.0082200727` over the 45 real members and won `15/15`; failed gates were `[]`. This promotes bounded grouped-video control-complete evidence for the original three-checkpoint ensemble. Do not generalize it beyond the selected target/head or into exact continuous forecasting.

The real upstream Optuna, Polars, MLflow, and SHAP integrations are installed as the `research-tooling` backend extra and documented in `docs/research_tooling_integrations.md`. Run `npm run verify:research-tooling` to exercise all four with verified MLX GPU/MPS hardware. Their runs remain exploratory and cannot promote canonical evidence.

## Test And Script Validation

`npm test` runs the deterministic contract suite: `python3 -m pytest -q tests`. On `2026-07-14`, the fully provisioned backend environment passed `138` tests, including the blocked and grouped control-complete checkpoint-ensemble contracts. The default `npm run verify` environment passed `130` tests and skipped one optional research-tooling module; repository readiness, strict-benchmark dry run, and frontend production build passed.

Relevant AGAIN and VEATIC v2 runners, tests, benchmark artifacts, and runtime-only tools are indexed in `docs/executable_validation_manifest.csv` and mirrored under `evidence/current_phase_5_5_review/14_executable_validation_and_code/`. Do not add placeholder smoke tests as validation. Add tests only when they protect a real split, target, leakage, control, manifest, scorer, checkpoint, or claim-boundary contract.

## Code Discovery

Prefer MCP graph tools for code-level lookup: `search_graph`, `trace_path`, `get_code_snippet`. Use `rg` for literal text, docs, configs, and generated artifacts.
