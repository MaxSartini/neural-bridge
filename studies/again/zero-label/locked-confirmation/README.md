# Zero-label video-only direct-supervised locked confirmation

The fixed direct-supervised temporal Neural Bridge passed its prospectively
locked 299-video AGAIN confirmation on `2026-07-15`.

## Verdict

- exact matrix: `140/140`;
- target/split/PCA/prediction-seal/inference/hardware audit: pass;
- Tier 1 baseline-beating deployment signal: pass;
- Tier 2 high-consistency confirmation: pass;
- Tier 3 first-30-second cold-start confirmation: pass;
- failed Tier 1 gates: `[]`;
- accelerator: MLX `Device(gpu, 0)`; no model-training CPU fallback.

The model was trained on 696 development videos and scored once on the frozen
299-video pool. It received no observed current/past arousal, teacher scores, or
response labels at any held-out inference row. Training remained supervised;
this is zero-label inference, not label-free training.

## Locked aggregate result

| Required endpoint | Video-only Neural Bridge | Strongest false-signal/no-video control | Absolute gain | Relative gain | Panel wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spearman future-movement ranking | `0.1785132961` | `0.1004882655` diagnostics-only | `+0.0780250306` | `+77.65%` | `5/5` |
| Top-5% true-movement lift | `0.0766079674` | `0.0448520122` no-video | `+0.0317559552` | `+70.80%` | `5/5` |
| Future-event PR-AUC | `0.1710622218` | `0.1352295369` diagnostics-only | `+0.0358326849` | `+26.50%` | `5/5` |

The one-sided paired whole-video bootstrap lower 95% bounds were respectively
`+0.0606787212`, `+0.0187740072`, and `+0.0235455194`—all above zero.

The fixed temporal model also beat the strong current-row video model on all
three endpoints by `+0.0111227287` Spearman, `+0.0114229070` top-5% lift, and
`+0.0075797964` event PR-AUC. It beat diagnostics-only, no-video,
sequence-shuffled, and hard-label-permuted controls.

## Cold start

Within the first 30 seconds, aggregate gains over the strongest control were
`+0.0803990394` Spearman, `+0.0350573783` top-5% lift, and `+0.0310250147`
event PR-AUC. Panel wins were `5/5`, `5/5`, and `4/5`, satisfying the locked
Tier 3 rule.

## Phase 7 ceiling

The observed-arousal-assisted ceiling remained higher at `0.2992983823`
Spearman, `0.1140998111` top-5% lift, and `0.2355919160` event PR-AUC. That is
expected because it receives response history unavailable on raw pre-release
video. On teacher-compatible rows, the video-only model retained `47.32%`,
`45.08%`, and `45.49%` of the ceiling's added gain over the no-video anchor.
These retention figures are diagnostic, not pass gates.

## Confirmed scope and next proof points

This confirms cold-start cached-feature zero-label inference on prospectively
locked AGAIN videos. The next independent proof points are end-to-end raw-video
feature generation/runtime, external-dataset transfer, and prospective client
outcomes. Exact trajectories, label-free training, and universal emotion
prediction are outside this study's claim.

Machine summary: `locked_confirmation_summary.json`.

Full runtime registry:
[`again-zero-label-locked-confirmation-20260715.json`](../../../../registry/artifacts/again-zero-label-locked-confirmation-20260715.json).
