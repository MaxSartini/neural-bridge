# VEATIC-124 v2 Evidence Summary

## Headline

VEATIC-124 v2 established the first controlled Neural Bridge result for ranking future arousal events from video-generated cortical/TRIBE features. Real causal feature modes beat autoregressive, shuffled, random, timestamp, and video/time controls under blocked and grouped-video evaluation.

Here `cortical/TRIBE` means frozen predicted cortical/fMRI response features generated from video by upstream models trained on cortical-response data. It does not mean direct viewer neural recordings or generic video embeddings.

## Key numbers

- Strongest blocked full-frame event row: `cortical_pca64_delta`, `arousal__future_spike_1_3s`, threshold `0.05`, PR-AUC `0.2536`.
- Same-row controls: AR `0.1969`, shuffled `0.1840`, random `0.1944`.
- Balanced event-vs-stable PR-AUC: `0.3394`, gains of `+0.0609` over AR, `+0.0631` over shuffled, and `+0.0476` over random.
- Grouped-video spike F1 gain over AR: PCA64 `+0.0256`; PCA64-delta `+0.0177`.
- Four of four focused temporal rows improved over current-only by more than `0.005` PR-AUC; the strongest focused window was causal past 2 seconds.
- The alignment audit found target-dependent offsets, so current `0s` alignment remained primary and offset sweeps remained diagnostic.
- The cache was video-dominant: `122/124` entries were video-only and `2/124` contained text, audio, and video.

## What this supports

1. Video-generated cortical/TRIBE features carry target-dependent signal for future arousal event ranking.
2. PCA feature modes can materially outperform the six-feature global baseline.
3. Balanced event-vs-stable evaluation can expose signal hidden by full-frame continuous metrics.
4. Short causal temporal context can outperform current-only state on selected event rows.
5. A single global lag correction was not justified.

## Boundaries

- Continuous future-change and exact-value prediction remained unresolved.
- Zero-change baselines still beat cortical features in most continuous checks.
- This was not a full multimodal TRIBE result, a finished product model, or evidence of external deployment validity.
- The result is foundational historical evidence, not authority for VEATIC 2.1 fitted PCA, tensors, labels, models, thresholds, checkpoints, windows, or head recipes.

AGAIN later extended this event signal with dense 2 Hz data, stronger frozen-AR controls, held-out-video confirmation, continuous future-movement ranking, and a locked zero-label-at-inference study. [Read the AGAIN result ladder](../../again/phase-07-continuous/evidence-summary.md).

## Canonical evidence

- [confirmatory benchmark](results/veatic_124_confirmatory_benchmark_report_20260616.md)
- [event-conditioned retest](results/veatic_124_event_conditioned_retest_20260616.md)
- [event/spike core retest](results/veatic_124_retest_event_spike_core_20260616.md)
- [alignment audit](results/veatic_124_alignment_lag_audit_20260616.md)
- [causal-window audit](results/veatic_124_alignment_causal_window_audit.md)
- [evidence manifest](results/veatic_v2_evidence_manifest.json)
- [strict reproduction runner](reproduction/backend/scripts/run_veatic_strict_benchmark.py)
