# Phase 5 Eval-Mode Checkpoint Rescore Summary

Source root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`

Eval-mode root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_evalmode_rescore_/`

## Scoring Contract

The original repair matrix trained correctly and saved best checkpoints. The original repair scoring was legacy train-mode/dropout-active because `model.eval()` was not called before scoring. This eval-mode checkpoint rescore is the canonical deterministic metric pass: it loads saved best checkpoints, disables dropout with `model.eval()`, and scores only original held-out rows.

No training, secondary heads, secondary targets, V-JEPA/TRIBE/PCA reruns, PCA refits, dense-cache writes, Phase 4 output edits, or original Phase 5 output edits were performed.

## Corrected Eval-Mode Result

- Grouped real PR-AUC: `0.2300639382`
- Grouped best matched control: `ar_plus_shuffled_pca` PR-AUC `0.2042740689`
- Grouped real-minus-control delta: `+0.0257898694`
- Grouped AR-only PR-AUC: `0.2246816187`
- Grouped fold-seed positive: `15/15`
- Blocked real PR-AUC: `0.2218656156`
- Blocked best matched control: `ar_plus_random_pca` PR-AUC `0.2311845051`
- Blocked real-minus-control delta: `-0.0093188895`
- Blocked AR-only PR-AUC: `0.2654721820`

Grouped support survives eval-mode scoring. Blocked support does not survive. AR-only still dominates blocked. Real PCA remains useful cross-video, but strict forward-time temporal generalization is not proven. Secondary heads should remain paused until the blocked mechanism is resolved.

## Legacy Train-Mode vs Eval-Mode

| validation_protocol | control_type | legacy_train_mode_pr_auc | eval_mode_pr_auc | delta_eval_minus_legacy_pr_auc | legacy_roc_auc | eval_roc_auc | delta_eval_minus_legacy_roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked_temporal_70_30 | ar_only_head | 0.260997 | 0.265472 | 0.004475 | 0.736023 | 0.739745 | 0.003723 |
| blocked_temporal_70_30 | ar_plus_random_pca | 0.228869 | 0.231185 | 0.002316 | 0.700280 | 0.702823 | 0.002543 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | 0.222617 | 0.225291 | 0.002673 | 0.682082 | 0.684793 | 0.002711 |
| blocked_temporal_70_30 | real_ar_pca_diag | 0.219998 | 0.221866 | 0.001868 | 0.681336 | 0.683568 | 0.002232 |
| blocked_temporal_70_30 | video_mean_pca_oracle_diagnostic | 0.192681 | 0.195527 | 0.002846 | 0.640864 | 0.642091 | 0.001227 |
| blocked_temporal_70_30 | label_permutation | 0.110305 | 0.110129 | -0.000176 | 0.499028 | 0.499267 | 0.000239 |
| grouped_video | real_ar_pca_diag | 0.228280 | 0.230064 | 0.001784 | 0.707675 | 0.709750 | 0.002075 |
| grouped_video | ar_only_head | 0.221552 | 0.224682 | 0.003130 | 0.705361 | 0.709493 | 0.004133 |
| grouped_video | ar_plus_shuffled_pca | 0.202163 | 0.204274 | 0.002111 | 0.671920 | 0.673843 | 0.001923 |
| grouped_video | ar_plus_random_pca | 0.200705 | 0.202896 | 0.002191 | 0.669607 | 0.672095 | 0.002489 |
| grouped_video | label_permutation | 0.105665 | 0.105805 | 0.000140 | 0.509730 | 0.510005 | 0.000276 |
| grouped_video | video_mean_pca_oracle_diagnostic | 0.104482 | 0.105481 | 0.000999 | 0.527003 | 0.527693 | 0.000690 |

## Required Answers

1. Grouped real remained above best matched grouped control: `True`.
2. Grouped real remained above AR-only: `True`.
3. Grouped fold-seed positivity survived: `15/15`.
4. Blocked real remained below best matched blocked control: `True`.
5. Blocked real remained below AR-only: `True`.
6. AR-only dominates blocked under eval-mode: `True`.
7. Label permutation remained near chance and below real: `True`.
8. Video-mean PCA remained unable to explain grouped: `True`.
9. Eval-mode scoring strengthens the corrected claim by making deterministic scoring canonical while preserving grouped support and blocked caveat.
10. The eval-mode numbers in this report and output root should now be considered canonical for deterministic checkpoint scoring.

## Ranking Shift Audit

Ranking shifts are recorded in `diagnostics/evalmode_control_ranking_shift_audit.json`.

```json
[
  {
    "validation_protocol": "blocked_temporal_70_30",
    "loss_name": "binary",
    "legacy_order": [
      "ar_only_head",
      "ar_plus_random_pca",
      "ar_plus_shuffled_pca",
      "real_ar_pca_diag",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_real",
      "diag_only",
      "quality_only",
      "shuffled_diag_only",
      "pca_only_shuffled",
      "pca_only_random",
      "label_permutation",
      "time_only"
    ],
    "eval_mode_order": [
      "ar_only_head",
      "ar_plus_random_pca",
      "ar_plus_shuffled_pca",
      "real_ar_pca_diag",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_real",
      "diag_only",
      "quality_only",
      "pca_only_shuffled",
      "shuffled_diag_only",
      "pca_only_random",
      "time_only",
      "label_permutation"
    ],
    "ranking_changed": true,
    "legacy_top_control": "ar_only_head",
    "eval_mode_top_control": "ar_only_head"
  },
  {
    "validation_protocol": "blocked_temporal_70_30",
    "loss_name": "regression",
    "legacy_order": [
      "ar_only_head",
      "ar_plus_random_pca",
      "ar_plus_shuffled_pca",
      "real_ar_pca_diag",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_real",
      "diag_only",
      "shuffled_diag_only",
      "time_only",
      "pca_only_shuffled",
      "pca_only_random",
      "quality_only",
      "label_permutation"
    ],
    "eval_mode_order": [
      "ar_only_head",
      "ar_plus_random_pca",
      "ar_plus_shuffled_pca",
      "real_ar_pca_diag",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_real",
      "diag_only",
      "time_only",
      "shuffled_diag_only",
      "quality_only",
      "pca_only_random",
      "pca_only_shuffled",
      "label_permutation"
    ],
    "ranking_changed": true,
    "legacy_top_control": "ar_only_head",
    "eval_mode_top_control": "ar_only_head"
  },
  {
    "validation_protocol": "blocked_temporal_70_30",
    "loss_name": "regression_plus_binary",
    "legacy_order": [
      "ar_only_head",
      "ar_plus_random_pca",
      "ar_plus_shuffled_pca",
      "real_ar_pca_diag",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_real",
      "diag_only",
      "quality_only",
      "pca_only_shuffled",
      "shuffled_diag_only",
      "pca_only_random",
      "label_permutation",
      "time_only"
    ],
    "eval_mode_order": [
      "ar_only_head",
      "ar_plus_random_pca",
      "ar_plus_shuffled_pca",
      "real_ar_pca_diag",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_real",
      "diag_only",
      "quality_only",
      "pca_only_shuffled",
      "shuffled_diag_only",
      "pca_only_random",
      "label_permutation",
      "time_only"
    ],
    "ranking_changed": false,
    "legacy_top_control": "ar_only_head",
    "eval_mode_top_control": "ar_only_head"
  },
  {
    "validation_protocol": "grouped_video",
    "loss_name": "binary",
    "legacy_order": [
      "real_ar_pca_diag",
      "ar_only_head",
      "ar_plus_shuffled_pca",
      "ar_plus_random_pca",
      "pca_only_real",
      "quality_only",
      "diag_only",
      "time_only",
      "label_permutation",
      "video_mean_pca_oracle_diagnostic",
      "shuffled_diag_only",
      "pca_only_random",
      "pca_only_shuffled"
    ],
    "eval_mode_order": [
      "real_ar_pca_diag",
      "ar_only_head",
      "ar_plus_shuffled_pca",
      "ar_plus_random_pca",
      "pca_only_real",
      "quality_only",
      "diag_only",
      "time_only",
      "video_mean_pca_oracle_diagnostic",
      "label_permutation",
      "pca_only_shuffled",
      "pca_only_random",
      "shuffled_diag_only"
    ],
    "ranking_changed": true,
    "legacy_top_control": "real_ar_pca_diag",
    "eval_mode_top_control": "real_ar_pca_diag"
  },
  {
    "validation_protocol": "grouped_video",
    "loss_name": "regression",
    "legacy_order": [
      "ar_only_head",
      "real_ar_pca_diag",
      "ar_plus_shuffled_pca",
      "ar_plus_random_pca",
      "pca_only_real",
      "diag_only",
      "quality_only",
      "video_mean_pca_oracle_diagnostic",
      "time_only",
      "label_permutation",
      "pca_only_random",
      "pca_only_shuffled",
      "shuffled_diag_only"
    ],
    "eval_mode_order": [
      "real_ar_pca_diag",
      "ar_only_head",
      "ar_plus_shuffled_pca",
      "ar_plus_random_pca",
      "pca_only_real",
      "diag_only",
      "quality_only",
      "time_only",
      "video_mean_pca_oracle_diagnostic",
      "label_permutation",
      "shuffled_diag_only",
      "pca_only_shuffled",
      "pca_only_random"
    ],
    "ranking_changed": true,
    "legacy_top_control": "ar_only_head",
    "eval_mode_top_control": "real_ar_pca_diag"
  },
  {
    "validation_protocol": "grouped_video",
    "loss_name": "regression_plus_binary",
    "legacy_order": [
      "real_ar_pca_diag",
      "ar_only_head",
      "ar_plus_shuffled_pca",
      "ar_plus_random_pca",
      "pca_only_real",
      "diag_only",
      "time_only",
      "quality_only",
      "label_permutation",
      "video_mean_pca_oracle_diagnostic",
      "shuffled_diag_only",
      "pca_only_random",
      "pca_only_shuffled"
    ],
    "eval_mode_order": [
      "real_ar_pca_diag",
      "ar_only_head",
      "ar_plus_shuffled_pca",
      "ar_plus_random_pca",
      "pca_only_real",
      "diag_only",
      "time_only",
      "quality_only",
      "label_permutation",
      "video_mean_pca_oracle_diagnostic",
      "pca_only_random",
      "shuffled_diag_only",
      "pca_only_shuffled"
    ],
    "ranking_changed": true,
    "legacy_top_control": "real_ar_pca_diag",
    "eval_mode_top_control": "real_ar_pca_diag"
  }
]
```

## Corrected Claim

Robust cross-video future arousal spike / emotional moment ranking is strengthened under deterministic eval-mode scoring. Strict forward-time temporal generalization remains unproven.
