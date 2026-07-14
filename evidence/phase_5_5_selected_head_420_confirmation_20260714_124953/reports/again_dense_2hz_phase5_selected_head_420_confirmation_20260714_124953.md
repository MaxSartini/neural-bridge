# AGAIN Phase 5 Selected-Head 420-Row Confirmation

## Verdict

**PASS.** Matrix completeness is `420/420`: `70/70` strict blocked temporal rows plus `350/350` grouped held-out-video rows. All scored rows were reused; no training, scoring, PCA fitting, or rerun was required.

Neural Bridge passes a full bounded 420-row selected-head confirmation for future arousal event ranking on AGAIN across strict blocked temporal validation and grouped held-out-video compatibility, using the redesigned washout-gap target, short temporal convolution residual, matched frozen AR, and matched controls.

## Fixed Scope

- Target: `future_arousal_max_delta_rows_4_10_train_q90`
- Head: `short_temporal_conv_residual`
- Feature: `temporal_mean_2s_then_pca256`
- Seeds: `20260625` through `20260634`
- Lanes: seven matched lanes per protocol/fold/seed
- Historical : not run and not part of this confirmation

## Canonical Protocol Results

| Protocol | Real PR-AUC | AR/frozen PR-AUC | Best matched control | Control PR-AUC | Delta vs AR | Delta vs control | Consistency |
|---|---:|---:|---|---:|---:|---:|---:|
| Blocked temporal | 0.2670735630 | 0.2602336231 | `random_pca_residual` | 0.2593369051 | +0.0068399399 | +0.0077366579 | 9/10 |
| Grouped video | 0.2313831909 | 0.2174953276 | `train_only_video_mean_residual` | 0.2174209937 | +0.0138878634 | +0.0139621972 | 50/50 |

The grouped label-permutation verdict remains frozen-AR-residual-aware: real minus label permutation is `+0.0160732134`, while label permutation minus AR is `-0.0021853501`. The superseded raw-prevalence-near-chance gate was not restored.

## Integrity And Reuse Audit

- Matrix keys complete and unique: `true` / `true`
- Target, head, target-window, split, PCA, and row provenance: `true`
- Frozen-AR score-cache hashes and within-group checksum identity: `true`
- Executable control policies: `true`
- Eval-mode checkpoint restoration/checksums: `true`
- Rows reused / rerun: `420` / `0`
- Failed gates: `[]`

## Artifacts

- Output root: `outputs/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953`
- Evidence snapshot: `evidence/phase_5_5_selected_head_420_confirmation_20260714_124953`
- Row manifest: `outputs/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953/metrics/selected_head_420_row_manifest.csv`
- Gate JSON: `outputs/again_dense_2hz_phase5_selected_head_420_confirmation_20260714_124953/promotion/selected_head_420_gates.json`

## Claim Boundary

This is a bounded selected-head binary future-event ranking confirmation. It is  and does not establish exact continuous-value forecasting, blocked continuous generalization, broad all-target/all-dataset prediction, or universal temporal generalization. The prior grouped continuous future-movement ranking/lift pass remains a separate bounded result.
