# Neural Bridge Zero-Label Deployment Evidence

## Bottom line

Neural Bridge now has a prospectively locked video-only result, not only an
AR-assisted research benchmark. The fixed direct-supervised temporal model was
trained on 696 AGAIN development videos and then scored once on a frozen
299-video pool. At held-out inference it used cached video-derived predicted
cortical/fMRI features and causal video metadata, with no observed arousal,
response history, teacher scores, or labeled warm start.

It completed `140/140` rows, passed every audit, and passed all three locked
verdict tiers.

## What it beat

| Endpoint | Neural Bridge | Strongest false-signal/no-video control | Gain |
| --- | ---: | ---: | ---: |
| Future-movement Spearman | `0.1785132961` | `0.1004882655` | `+77.65%` |
| Top-5% true-movement lift | `0.0766079674` | `0.0448520122` | `+70.80%` |
| Future-event PR-AUC | `0.1710622218` | `0.1352295369` | `+26.50%` |

Every full-video endpoint won in all `5/5` frozen panels. All three paired
whole-video bootstrap lower 95% bounds were positive. The first-30-second
cold-start tier also passed.

Controls answer different failure explanations:

- diagnostics-only asks whether the 53 generic video diagnostics are enough;
- no-video retains only masks/time metadata;
- sequence shuffle preserves realistic-looking temporal input while breaking
  content/outcome alignment;
- hard-label permutation breaks the training relationship;
- current-row keeps real video information but removes causal video history.

The real temporal bridge beats all of them on all three aggregate endpoints.
It also beats current-row by `+0.0111227287` Spearman, `+0.0114229070` top-5%
lift, and `+0.0075797964` event PR-AUC, supporting the value of temporal video
context rather than only static/current-frame signal.

## Product meaning

The remaining deployment gap is no longer “can useful signal survive without
response labels at inference?” On locked AGAIN videos, it can. The remaining
work is translation and generalization:

1. run the frozen feature extractor and bridge end to end from an actual raw
   client-style video rather than a precomputed cache;
2. confirm the video-only lane on a genuinely external/cross-domain or
   prospective client-style set;
3. calibrate output bands, uncertainty, latency, and response-readiness reports
   without turning ranking scores into false exact-value claims.

This remains supervised training. It is not mind reading, individual profiling,
medical inference, exact trajectory prediction, or a universal emotion model.

## Authority

- Preregistration: `preregistration/locked-confirmation.md`
- Claim-bearing report:
  `locked-confirmation/reports/again_dense_2hz_zero_label_direct_supervised_locked_confirmation_20260715.md`
- Checksum-anchored evidence: `locked-confirmation/`
- Full-run registry:
  `../../../registry/artifacts/again-zero-label-locked-confirmation-20260715.json`
- Portable verifier: `src/neural_bridge/zero_label/`
