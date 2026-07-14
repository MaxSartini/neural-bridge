# Phase 5 Continuous Residual Blocked Summary

> [!NOTE]
> **HISTORICAL, PROTOCOL-SPECIFIC FAILURE.** This blocked residual diagnostic failed and remains valid for this exact setup. It does not erase the separate deterministic eval-mode grouped continuous future-movement ranking/lift pass: real Spearman `0.2232222830` beat AR-only `0.1982207591`, shuffled `0.1938183619`, and random `0.1931781163`; real top-1% lift `0.1359465244` beat `0.1115815364`, `0.1125842464`, and `0.1136304212`. Exact-value forecasting and blocked continuous generalization remain open.

Output root: `outputs/again_dense_2hz_phase5_continuous_residual_blocked_20260630_000219`

This is a targeted continuous future arousal movement residual experiment over frozen AR. It uses only `blocked_temporal_70_30`, only `monotonic_do_no_harm_residual`, five prespecified seeds, and seven controls for a maximum of 35 rows. It does not rerun binary spike confirmation, grouped 5-fold, secondary targets, AR training, V-JEPA/TRIBE, or PCA.

## Result

- Continuous residual pass: `False`
- Credible continuous pass: `False`
- Recommendation: `continuous_residual_failed_do_not_run_grouped`
- Strict forward-time spike prediction claimed: `False`

## Continuous Metrics

- Real Spearman: `0.2484145880`
- Frozen AR Spearman: `0.2695371538`
- Spearman delta: `-0.0211225658`
- Real top 1pct continuous lift: `0.1430029988`
- Frozen AR top 1pct continuous lift: `0.1494930923`
- Real top 5pct continuous lift: `0.0951335132`
- Frozen AR top 5pct continuous lift: `0.0981196761`
- Top 5pct lift delta vs frozen AR: `-0.0029861629`
- Best matched control: `random_pca_continuous_residual` top 5pct lift `0.0976563662`
- Delta vs best matched control: `-0.0025228530`
- Label permutation top 5pct lift: `0.0668185577`
- Train-only video mean top 5pct lift: `0.0980083317`

## Gates

- `real_gt_frozen_ar_top5_pass`: `False`
- `real_gt_controls_top5_pass`: `False`
- `spearman_delta_vs_frozen_ar_positive`: `False`
- `top5_seed_positive_count`: `0/5`
- `no_single_seed_over_60pct_pass`: `False`
- `binary_pr_auc_do_no_harm_pass`: `True`
- `frozen_ar_integrity_pass`: `True`
- `checkpoint_restore_pass`: `True`
- `eval_mode_scoring_pass`: `True`

Binary spike metrics are secondary only in this run. Do not claim strict forward-time spike prediction unless binary PR-AUC gates pass in a separate binary confirmation.
