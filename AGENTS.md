# Neural Bridge Agent Guide

## Source Rules

- Use current repo/workspace files and current user prompts as benchmark truth.
- Do not use prior chat memory, Claude/VS Code state, old plans, compacted context, or Spark-era outputs as authority.
- `codebase-memory-mcp` is for code navigation only, not benchmark authority.
- Keep context use compact. Do not dump full reports, logs, or generated metadata unless asked.

## Current Claim

Canonical claim: Neural Bridge demonstrates controlled future human arousal event-ranking from frozen video-derived predictions generated from brain cortical response data across VEATIC and AGAIN.

VEATIC-124 v2 established the original controlled future arousal spike/event-ranking signal. AGAIN replicated, scaled, validated, and strengthened it using 995 videos, 2 Hz dense V-JEPA 2.1 / TRIBE v2 features, frozen-AR residuals, a redesigned washout-gap future arousal event target, blocked temporal confirmation, and grouped-video compatibility.

The frozen video-side features are learned predictions generated from brain cortical response data. Treat them as neuro-response-derived video features, not generic video embeddings and not direct neural recordings from the benchmark viewer rows.

Raw cortical-derived features alone fail badly on AGAIN. On the original Phase 3 spike target `arousal_spike_rows_2_6_train_q90`, blocked `raw_cortical_only` PR-AUC was `0.124315` versus AR-only `0.203622`, and direct `AR_plus_raw_cortical` dropped to `0.167731`. Neural Bridge is the current result, not raw cortical features by themselves.

Bounded strict forward-time future-event ranking is proven on AGAIN for `future_arousal_max_delta_rows_4_10_train_q90` with `short_temporal_conv_residual`. Grouped held-out-video compatibility for the same target/head is proven under the updated frozen-AR-residual-aware verdict.

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

Canonical reviewer evidence dossier:
`evidence/current_phase_5_5_review/README.md`

Canonical executable validation index:
`docs/executable_validation_index.md`

Canonical blocked temporal binary confirmation:
`reports/again_dense_2hz_phase5_temporal_residual_binary_big_confirm_20260630_025437.md`

Canonical updated grouped compatibility verdict:
`reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520_UPDATED_VERDICT.md`

Current bottom line: VEATIC is foundational, and AGAIN is now the scaled confirmation/current main result. Raw cortical-derived features alone were weak; the Neural Bridge pipeline makes the difference. The old fused AGAIN lane passed grouped eval-mode controls but failed blocked AR/control checks; frozen-AR residual and temporal/event-context residual designs fixed the design path. The redesigned washout-gap target/head passes blocked temporal confirmation and updated grouped-video compatibility. Continuous exact arousal forecasting and broad universal temporal prediction remain open.

## Commercial Language

Neural Bridge should be described as Service as Software for neuro-response video intelligence: software that automates the first-pass expert service of pre-release video response evaluation, creative diagnostics, variant comparison, and response-readiness reporting using video-derived predictions generated from brain cortical response data.

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

Do not touch dense cache files, Phase 4 outputs, original Phase 5 output roots, or evidence bundle contents unless explicitly asked. Do not force-add ignored output roots.

## Current Numbers

AGAIN blocked temporal binary confirmation:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- real PR-AUC: `0.2670735630`
- frozen AR PR-AUC: `0.2602336231`
- best control `random_pca_residual` PR-AUC: `0.2593369051`
- delta vs frozen AR: `+0.0068399399`
- delta vs best control: `+0.0077366579`
- seeds positive vs AR and best control: `9/10`, `9/10`
- weak / credible / strong confirmation: true
- failed gates: `[]`

AGAIN updated grouped compatibility:

- target/head: `future_arousal_max_delta_rows_4_10_train_q90` / `short_temporal_conv_residual`
- real PR-AUC: `0.2313831909`
- AR/frozen PR-AUC: `0.2174953276`
- best matched control `train_only_video_mean_residual` PR-AUC: `0.2174209937`
- delta vs AR/frozen: `+0.0138878634`
- delta vs best control: `+0.0139621972`
- fold-seed positives vs best control: `50/50`
- real minus label permutation: `+0.0160732134`
- label permutation minus AR: `-0.0021853501`
- updated grouped compatibility pass: true

AGAIN dense substrate:

- `995/995` videos complete, `243,575` row-level brain-cortical-response-data-generated video feature rows, true `2 Hz` labels, `256 px`, float16, official V-JEPA 2.1 ViT-G, TRIBE v2 cache-only postpass

AGAIN eval-mode and frozen-AR design path:

- eval-mode grouped real `regression_plus_binary` PR-AUC: `0.2300639382`
- eval-mode grouped best matched control `ar_plus_shuffled_pca` PR-AUC: `0.2042740689`
- eval-mode grouped real-minus-control delta: `+0.0257898694`
- eval-mode grouped fold-seed delta: positive in `15/15`
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

Next work is explicit review and any intentionally approved 504/broader compatibility plan, not broad uncontrolled secondary heads. Do not claim 504 or universal temporal prediction until that later confirmation is actually run and promoted.

## Test And Script Validation

`npm test` runs the full deterministic contract suite: `python3 -m pytest -q tests`. The current validated suite is `93 passed in 5.52s` on `2026-06-30`.

Relevant AGAIN and VEATIC v2 runners, tests, benchmark artifacts, and runtime-only tools are indexed in `docs/executable_validation_manifest.csv` and mirrored under `evidence/current_phase_5_5_review/14_executable_validation_and_code/`. Do not add placeholder smoke tests as validation. Add tests only when they protect a real split, target, leakage, control, manifest, scorer, checkpoint, or claim-boundary contract.

## Code Discovery

Prefer MCP graph tools for code-level lookup: `search_graph`, `trace_path`, `get_code_snippet`. Use `rg` for literal text, docs, configs, and generated artifacts.
