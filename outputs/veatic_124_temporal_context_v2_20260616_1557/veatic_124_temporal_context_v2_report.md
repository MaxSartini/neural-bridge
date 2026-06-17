# VEATIC-124 Temporal Context v2

This is a narrow reuse-first temporal context sufficiency benchmark for future arousal spike ranking.

## Executive Verdict

V2 narrowed to future spike ranking and found best causal windows {'causal_past_2s': 4} with best representations {'last': 2, 'mean': 2}. 4/4 feature-target rows improved over current-only by more than 0.005 PR-AUC, and 0/4 had positive context-specific gain versus AR above 0.003. This supports temporal context sufficiency checks where gains are real-feature-specific, but does not justify universal early-warning or label-shift claims.

## Reuse

- `outputs/veatic_124_temporal_fairness_20260616_1509/causal_context_window_results.csv`: prior 0s, 3s, 5s causal context summary rows; directly comparable grouped-video v1 rows reused for matching cells.
- `outputs/veatic_124_temporal_fairness_20260616_1509/offset_sweep_results.csv`: prior focused offset sweep; v2 compares against prior best offset without rerunning offsets.
- `outputs/veatic_124_temporal_fairness_20260616_1509/train_selected_offset_results.csv`: prior train-only selected offset results; v2 compares against prior selected-offset outcomes without rerunning offsets.
- `outputs/veatic_124_temporal_fairness_20260616_1509/leakage_audit.csv`: prior leakage checks; v2 extends same leakage policy.
- `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`: VEATIC-124 manifest rows and split metadata; same labels/video ids as prior run.
- `benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json`: VEATIC-124 manifest report and complete video ids; same 124-video accepted set.
- `<external-assets-root>/benchmarks/veatic/tribe_cache`: cached TRIBE/cortical raw outputs; tribe_raw_output.npz count=124; source for cache-only PCA/context matrices; no video re-encoding.
- `outputs/veatic_124_temporal_fairness_20260616_1509/causal_context_window_results.csv`: 60 directly reused summary rows; avoid recomputing matching v1 context summary cells.

## Best Windows

| Feature | Target | Thr | Best window | Representation | PR-AUC | Real gain | Specific gain vs AR |
|---|---|---:|---|---|---:|---:|---:|
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.050 | causal_past_2s | last | 0.4188 | 0.0330 | -0.0051 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.075 | causal_past_2s | last | 0.3119 | 0.0342 | -0.0020 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.050 | causal_past_2s | mean | 0.4365 | 0.0366 | -0.0015 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.075 | causal_past_2s | mean | 0.3339 | 0.0365 | 0.0003 |

## Top Real Minus AR Rows

- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_5s mean: real_minus_AR=0.0929, real_PR_AUC=0.3796
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_4s mean: real_minus_AR=0.0928, real_PR_AUC=0.3947
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_3s mean: real_minus_AR=0.0862, real_PR_AUC=0.3968
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_2s mean: real_minus_AR=0.0814, real_PR_AUC=0.3965
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_5s mean_slope: real_minus_AR=0.0808, real_PR_AUC=0.3418
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_5s mean_std_last_slope: real_minus_AR=0.0793, real_PR_AUC=0.3402
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_5s mean_last: real_minus_AR=0.0792, real_PR_AUC=0.3401
- cortical_pca64_delta arousal__future_spike_1_3s thr=0.075 causal_past_5s last: real_minus_AR=0.0769, real_PR_AUC=0.3378
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_5s last_slope: real_minus_AR=0.0752, real_PR_AUC=0.3361
- cortical_pca_64 arousal__future_spike_1_3s thr=0.075 causal_past_4s mean_slope: real_minus_AR=0.0744, real_PR_AUC=0.3471

## Top Context-Specific Rows

- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s mean vs timestamp: window_specific_gain=0.1402
- cortical_pca64_delta arousal__future_spike_1_3s thr=0.05 causal_past_5s last vs timestamp: window_specific_gain=0.1337
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s mean_slope vs timestamp: window_specific_gain=0.1333
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s last vs timestamp: window_specific_gain=0.1327
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s mean_std_last_slope vs timestamp: window_specific_gain=0.1317
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s last_slope vs timestamp: window_specific_gain=0.1274
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s mean_last vs timestamp: window_specific_gain=0.1260
- cortical_pca64_delta arousal__future_spike_1_3s thr=0.05 causal_past_5s mean vs timestamp: window_specific_gain=0.1206
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_5s slope vs timestamp: window_specific_gain=0.1178
- cortical_pca_64 arousal__future_spike_1_3s thr=0.05 causal_past_4s mean vs timestamp: window_specific_gain=0.1121

## Recommended Claim

Best defensible claim: short causal context can modestly improve future arousal spike ranking for selected TRIBE/cortical PCA modes, so single-timestep 0s evaluation may underfeed the bridge head. The effect must be reported with controls because some context changes can track label/timing structure rather than TRIBE-specific information.

## Forbidden Claims

- Do not claim universal early warning.
- Do not claim exact future arousal prediction.
- Do not claim TRIBE is globally early or late from this v2 alone.
- Do not use symmetric diagnostic windows as final predictive claims.