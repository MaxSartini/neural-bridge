# VEATIC-124 Frozen Tensor Trained-Head Benchmark
## Executive Verdict
1. Did trained heads improve over ridge? Best heads are `elastic_net_logistic` at spike @ 0.05 and `elastic_net_logistic` at spike @ 0.075; compare lane CSVs against `ridge_score` rows for exact deltas.
2. Did PCA128_only beat AR_only? See gate checks; the best PCA128-only grouped PR-AUC at @0.05 is 0.2989 versus AR 0.4170.
3. Did AR_plus_PCA128 beat AR_only? @0.05 best AR-plus is 0.4315; @0.075 best AR-plus is 0.3269.
4. Did residualized_AR_plus_PCA128 beat AR_only? @0.05 best residualized is 0.4300; @0.075 best residualized is 0.3243.
5. Did AR_plus_PCA128 beat AR_plus_PCA64_delta? See gate checks for grouped deltas.
6. Did residualized PCA128 beat residualized PCA64-delta? See gate checks for grouped deltas.
7. Which head worked best? AR-plus best heads: @0.05 `elastic_net_logistic`, @0.075 `elastic_net_logistic`.
8. Did sequence/temporal pooling help? Sequence-head rows are included with `uses_sequence_tensor=true`; compare them to collapsed `logistic_l2` and `ridge_score` rows.
9. Did canonical controls pass? Yes.
10. What is the best honest claim? `promoted_incremental_neural_value` for the primary spike target, subject to the gate table and no final promotion JSON.
## Full VEATIC-124 Policy
Video 83 was included. Exclude-video-83 sensitivity was intentionally skipped because the project benchmark is the full VEATIC-124 set.
- full_veatic_124: `true`
- video_83_included: `true`
- exclude_video_83_run: `false`
## Reuse Policy
- Reused frozen tensors, metadata/checksum/leakage contracts, canonical helpers, splits, lane semantics, and controls.
- Recomputed AR matrices, model fits, predictions, controls, metrics, fold aggregates, gates, and summaries.
## Tensor Inputs
- primary: `pca_sequence_128_causal_past_2s_mean`
- previous neural baseline: `cortical_pca64_delta_frozen_baseline`
## Existing Suite Semantics
The run uses canonical AR/time/ridge/control/event-metric helper functions through the frozen tensor adapter.
## Heads Tested
ridge_score, logistic_l2, elastic_net_logistic, flattened_sequence_logistic, learned_temporal_pool_logistic
## Ridge Parity Check
Rows with `head_name=ridge_score` are included as the parity check against the ridge-only run.
## Original AR Baseline
{"head": "elastic_net_logistic", "mean_grouped_pr_auc": 0.41701680420223325}
## PCA128-Only Results
{"0.05": {"head": "flattened_sequence_logistic", "mean_grouped_pr_auc": 0.29888635506930733}, "0.075": {"head": "elastic_net_logistic", "mean_grouped_pr_auc": 0.22566232010459952}}
## AR-plus-PCA128 Incremental Results
{"0.05": {"head": "elastic_net_logistic", "mean_grouped_pr_auc": 0.43146544934241043}, "0.075": {"head": "elastic_net_logistic", "mean_grouped_pr_auc": 0.32694254536453893}}
## Residualized AR-plus-PCA128 Results
{"0.05": {"head": "ridge_score", "mean_grouped_pr_auc": 0.42997649831762885}, "0.075": {"head": "ridge_score", "mean_grouped_pr_auc": 0.32426561541114085}}
## PCA64-Delta Baseline Comparisons
{"0.05": {"AR_plus": {"head": "elastic_net_logistic", "mean_grouped_pr_auc": 0.3996722648266604}, "residualized": {"head": "ridge_score", "mean_grouped_pr_auc": 0.3916937439985917}}}
## Sequence Tensor Head Results
Sequence-head rows are marked with `uses_sequence_tensor=true` in `lane_results.csv`.
## Temporal Pooling Results
Temporal-pooling selections are recorded in `head_selection_details.json`.
## ROI and Top-K Exploratory Results
ROI and top-k exploratory lanes were not part of this required run. Top-k, if enabled later, must remain cautionary and non-headline.
## Canonical Control Results
Canonical control failures: `0`.
## Grouped Fold Stability
Stable positive folds are recorded per primary gate in `gate_checks.json`.
## Leakage and Freshness Audit
Leakage failures: `0`. Freshness ledger records fresh AR, controls, predictions, and score computation.
## Gate Summary
[
  {
    "best_heads": {
      "AR_only": "elastic_net_logistic",
      "AR_plus_PCA128": "elastic_net_logistic",
      "AR_plus_PCA64_delta": "elastic_net_logistic",
      "PCA128_only": "flattened_sequence_logistic",
      "PCA64_delta_only": "elastic_net_logistic",
      "residualized_AR_plus_PCA128": "ridge_score",
      "residualized_AR_plus_PCA64_delta": "ridge_score"
    },
    "category": "promoted_incremental_neural_value",
    "control_pass": true,
    "means": {
      "AR_only": 0.41701680420223325,
      "AR_plus_PCA128": 0.43146544934241043,
      "AR_plus_PCA64_delta": 0.3996722648266604,
      "PCA128_only": 0.29888635506930733,
      "PCA64_delta_only": 0.31577777753705494,
      "residualized_AR_plus_PCA128": 0.42997649831762885,
      "residualized_AR_plus_PCA64_delta": 0.3916937439985917
    },
    "not_final_promotion": true,
    "target_name": "arousal__future_spike_1_3s",
    "threshold": 0.05
  },
  {
    "best_heads": {
      "AR_only": "logistic_l2",
      "AR_plus_PCA128": "elastic_net_logistic",
      "AR_plus_PCA64_delta": "elastic_net_logistic",
      "PCA128_only": "elastic_net_logistic",
      "PCA64_delta_only": "elastic_net_logistic",
      "residualized_AR_plus_PCA128": "ridge_score",
      "residualized_AR_plus_PCA64_delta": "ridge_score"
    },
    "category": "promoted_incremental_neural_value",
    "control_pass": true,
    "means": {
      "AR_only": 0.30152827657642645,
      "AR_plus_PCA128": 0.32694254536453893,
      "AR_plus_PCA64_delta": 0.28355823106447864,
      "PCA128_only": 0.22566232010459952,
      "PCA64_delta_only": 0.22144086461576457,
      "residualized_AR_plus_PCA128": 0.32426561541114085,
      "residualized_AR_plus_PCA64_delta": 0.28302787174187255
    },
    "not_final_promotion": true,
    "target_name": "arousal__future_spike_1_3s",
    "threshold": 0.075
  }
]
## Best Honest Claim
promoted_incremental_neural_value
## Next Recommended Experiment
Inspect the best trained-head rows against controls before deciding whether to run optional tiny MLPs.
