# VEATIC-124 v2 Evidence Summary

Generated from the current v2 reports imported into the cleaned Neural Bridge repo.

## Headline

VEATIC-124 v2 proves specific Neural Bridge hypotheses for arousal event/spike ranking. It shows that cortical/TRIBE PCA feature modes can improve future arousal spike/event ranking over autoregressive, shuffled, random, timestamp, and video/time controls under blocked and grouped-video validation.

The claim remains bounded: this is event/spike ranking and temporal-context evidence, not exact continuous arousal-value prediction or a finished downstream product model.

## Proven Or Supported Hypotheses

1. Real cortical/TRIBE features carry stimulus-specific signal for future arousal spike ranking.
2. PCA feature modes are materially stronger than the 6-feature global baseline for spike/event ranking.
3. Balanced event-vs-stable evaluation exposes signal that full-frame continuous MAE can hide.
4. Short causal temporal context can improve selected spike-ranking rows over current-only evaluation.
5. Single-frame 0s evaluation can underfeed the bridge head for spike/event tasks.
6. Timing/alignment policy is resolved: current 0s alignment stays primary; offset-grid and train-selected timing checks are diagnostics.

## Key Numbers

- Strongest blocked full-frame spike row: `cortical_pca64_delta`, `arousal__future_spike_1_3s`, threshold `0.05`, PR-AUC `0.2536`.
- Same row controls: AR `0.1969`, shuffled `0.1840`, random `0.1944`.
- Official split spike rows pass controls across current feature families.
- Grouped-video aggregate spike F1 improves over AR for PCA modes: `cortical_pca_64` `+0.0256`, `cortical_pca64_delta` `+0.0177`.
- Balanced event-vs-stable `arousal__future_spike_1_3s@0.05`: `cortical_pca64_delta` PR-AUC `0.3394`, `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random.
- Temporal context v2: 4/4 focused feature-target rows improved over current-only by more than `0.005` PR-AUC; best focused windows were `causal_past_2s`.
- Alignment repair: best offsets vary by target/mode, so no global lag correction was selected; final policy is `keep_current_0s_as_primary_plus_report_offset_diagnostics`.

## Boundaries

- Continuous future-change MAE remains diagnostic only.
- Zero-change baselines still beat real cortical features in most continuous checks.
- Offset diagnostics should not be promoted into final scores unless a future train-only policy survives controls and grouped validation.
- Legacy validation branches and retired secondary model expansion are not active validation requirements for the v2 claim.
- Simulation/LLM-agent integration is outside the current v2 evidence roadmap.

## Source Reports

- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md`
- `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md`
- `outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md`
- `benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md`
- `benchmarks/veatic/veatic_124_alignment_candidate_fixes.md`
- `benchmarks/veatic/veatic_124_alignment_causal_window_audit.md`
