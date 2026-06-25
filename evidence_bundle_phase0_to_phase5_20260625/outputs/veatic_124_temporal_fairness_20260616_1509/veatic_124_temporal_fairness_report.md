# VEATIC-124 Temporal Fairness Benchmark

Offset convention: `feature_time = label_anchor_time + offset_seconds`. Negative offsets use earlier TRIBE/cortical features; positive offsets use later TRIBE/cortical features.

## Executive Verdict

This temporal fairness benchmark evaluates whether the current 0s single-timestep TRIBE interface is fair under grouped-video validation. Taxonomy counts: fair at 0s=0, context-starved=2, earlier-feature advantage=0, later-feature advantage=0, control-driven=0, unstable/video-specific=6. Causal windows and train-selected offsets are valid predictive diagnostics; future-inclusive windows are diagnostic only.

## Conservative Headline Rows

| Feature | Target | Thr | Taxonomy | 0s PR-AUC | Best causal | Causal gain | Best offset | Offset gain |
|---|---|---:|---|---:|---|---:|---:|---:|
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.050 | Unstable/video-specific | 0.4221 | causal_past_3s | -0.0547 | 4.0000 | 0.0040 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.050 | Unstable/video-specific | 0.4457 | causal_past_3s | -0.0368 | 0.0000 | 0.0000 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.075 | Unstable/video-specific | 0.2981 | causal_past_3s | -0.0408 | 4.0000 | 0.0162 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.075 | Unstable/video-specific | 0.3189 | causal_past_3s | -0.0197 | 4.0000 | 0.0072 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.050 | Unstable/video-specific | 0.3859 | causal_past_3s | -0.0279 | -2.0000 | 0.0257 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.050 | Context-starved | 0.4000 | causal_past_3s | 0.0114 | -1.5000 | 0.0368 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.075 | Unstable/video-specific | 0.2777 | causal_past_3s | -0.0165 | 4.0000 | 0.0299 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.075 | Context-starved | 0.2975 | causal_past_3s | 0.0137 | -1.5000 | 0.0355 |

## Benchmark Arms

- Arm 1: current 0s single-timestep baseline.
- Arm 2: causal temporal context windows ending at the label anchor.
- Arm 3: offset sensitivity sweep, diagnostic unless selected train-only.
- Arm 4: train-only selected alignment under grouped-video folds.
- Arm 5: future-inclusive diagnostic context, excluded from final predictive claims.

## Leakage Audit

All final predictive arms keep grouped video holdout, train-only PCA/scalers, train-only thresholds, and no future-inclusive feature context. Future-inclusive windows are explicitly diagnostic.

## Timing Fairness Taxonomy

- Context-starved: 2
- Unstable/video-specific: 6

## Recommended Final Claim

Best defensible claim: TRIBE/cortical features should be judged with input-timing fairness checks. If causal windows or train-selected offsets outperform 0s while controls do not, the single-timestep 0s interface likely underestimates the representation. Do not promote any test-selected or per-video lag correction to a final benchmark score.

## Forbidden Claims

- Do not claim universal early warning.
- Do not claim lag fixed the benchmark.
- Do not claim exact future arousal prediction.
- Do not use test-selected lag correction as a final score.
- Do not use per-video corrected final scores.

## Output Index

- `zero_offset_results.csv`
- `causal_context_window_results.csv`
- `offset_sweep_results.csv`
- `train_selected_offset_results.csv`
- `diagnostic_future_context_results.csv`
- `real_vs_control_context_specificity.csv`
- `real_vs_control_offset_specificity.csv`
- `balanced_event_stable_results.csv`
- `bootstrap_ci.csv`
- `timing_fairness_taxonomy.csv`
- `selected_offsets_by_fold.csv`
- `ar_behavior_audit.csv`
- `leakage_audit.csv`
- `veatic_124_temporal_fairness_report.md`
- `veatic_124_temporal_fairness_summary.json`