# Phase 5 Gate/Fusion Checkpoint Diagnostic

Source output root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`

## No New Training

This pass loaded saved best checkpoints and re-forwarded held-out benchmark rows only. It did not train models, start secondary heads, start secondary targets, rerun the 702 matrix, rerun V-JEPA/TRIBE/PCA, or modify Phase 4/original Phase 5 outputs.

## Checkpoints Loaded

- Checkpoints loaded: `72`
- Manifest: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/instrumentation/gate_fusion_checkpoint_only/gate_fusion_manifest.json`
- Selected controls: `real_ar_pca_diag`, `ar_plus_random_pca`, `ar_plus_shuffled_pca`, `ar_only_head`
- Protocols: `grouped_video`, `blocked_temporal_70_30`
- Loss/model: `regression_plus_binary`, `gated_ar_pca_mlp`

## Metric Reproduction Audit

- Deterministic eval-mode reproduction pass: `False`
- Max absolute diffs: `{"pr_auc": 0.00595432286551173, "roc_auc": 0.006158307800924967, "top_1pct_recall": 0.006783625730994197, "top_5pct_recall": 0.006424412768520432, "top_10pct_recall": 0.009055026700719726, "continuous_pearson": 0.07569545409953551}`
- Note: `Original repair scoring did not call model.eval(); dropout made exact train-mode metric reproduction non-deterministic from checkpoint alone. Instrumentation uses deterministic eval mode.`

## Grouped Gate Behavior

| validation_protocol | control_type | checkpoints | mean_gate | mean_saturation_low_rate | mean_saturation_high_rate | mean_pca_input_norm | mean_pca_rep_norm | mean_corr_gate_pca_norm | mean_corr_prediction_pca_score | mean_corr_pca_score_future_target | mean_pr_auc | mean_roc_auc | mean_spearman_future_movement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_video | ar_only_head | 15 | 0.494348 | 0.000000 | 0.000000 |  |  |  |  |  | 0.224682 | 0.709493 | 0.198221 |
| grouped_video | ar_plus_random_pca | 15 | 0.458435 | 0.000000 | 0.000000 | 15.981445 | 7.562963 | 0.036993 | 0.228707 | -0.001094 | 0.202896 | 0.672095 | 0.193178 |
| grouped_video | ar_plus_shuffled_pca | 15 | 0.456377 | 0.000000 | 0.000000 | 137.406031 | 7.593328 | -0.001883 | 0.218921 | 0.001413 | 0.204274 | 0.673843 | 0.193818 |
| grouped_video | real_ar_pca_diag | 15 | 0.472855 | 0.000000 | 0.000000 | 137.406031 | 7.394460 | -0.074443 | 0.458853 | 0.092332 | 0.230064 | 0.709750 | 0.223222 |

Grouped real remains the best matched-control result. Gate diagnostics show whether real PCA differs from random/shuffled controls under deterministic eval-mode re-forward, but the corrected grouped claim still rests on the committed repair metrics.

## Blocked Gate Behavior

| validation_protocol | control_type | checkpoints | mean_gate | mean_saturation_low_rate | mean_saturation_high_rate | mean_pca_input_norm | mean_pca_rep_norm | mean_corr_gate_pca_norm | mean_corr_prediction_pca_score | mean_corr_pca_score_future_target | mean_pr_auc | mean_roc_auc | mean_spearman_future_movement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blocked_temporal_70_30 | ar_only_head | 3 | 0.490206 | 0.000000 | 0.000000 |  |  |  |  |  | 0.265472 | 0.739745 | 0.215649 |
| blocked_temporal_70_30 | ar_plus_random_pca | 3 | 0.460704 | 0.000000 | 0.000000 | 15.981096 | 7.527515 | 0.041332 | 0.225554 | -0.004485 | 0.231185 | 0.702823 | 0.189220 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | 3 | 0.457097 | 0.000000 | 0.000000 | 140.868301 | 9.984815 | 0.009742 | 0.378370 | -0.004573 | 0.225291 | 0.684793 | 0.184308 |
| blocked_temporal_70_30 | real_ar_pca_diag | 3 | 0.474024 | 0.000000 | 0.000000 | 140.868301 | 10.696252 | -0.243560 | 0.611432 | 0.046816 | 0.221866 | 0.683568 | 0.177648 |

Blocked AR-only remains strongest among inspected controls in eval-mode re-forward. Random/shuffled PCA do not beat AR-only; they mainly show that real PCA is more harmful than random/shuffled PCA inside the fused head under blocked validation.

## Mechanism Verdict

- `ar_time_dominance_supported`: `True`
- `random_pca_regularization_supported`: `True`
- `harmful_real_pca_fusion_supported`: `True`
- `gate_routing_supported`: `True`
- `control_bug_supported`: `False`
- `split_prevalence_artifact_supported`: `False`
- `strict_forward_time_temporal_generalization_proven`: `False`

Interpretation: AR/time dominance remains the strongest mechanism. The checkpoint-only gate/fusion view supports harmful real-PCA fusion as a plausible blocked mechanism because the real fused head underperforms random/shuffled PCA while its prediction is more coupled to the PCA branch. The PCA branch score is not anticorrelated with the future target, so this is a fusion/routing issue rather than simple negative PCA signal. It does not rescue strict forward-time temporal generalization.

## Retrain Needed?

No retrain is needed for the corrected claim. A minimal retrain is only needed if exact original train-mode metric reproduction or full gate-vector/per-row branch behavior is required with scoring semantics fixed.

Exact next recommendation: Do not expand to secondary heads. If mechanism-level proof is required, run only the minimal instrumented blocked/grouped rerun with eval-mode scoring fixed and full gate vectors/per-row branch logs.
