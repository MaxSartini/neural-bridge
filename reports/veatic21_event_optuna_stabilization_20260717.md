# VEATIC 2.1 Event Optuna Stabilization

Status: `FAIL`
Run identity: `4b34b06bd2a1fa95975ac88a0bf797a28ebbe1956628dc250f515264dd3249bb`
Execution commit: `e1b0d9f8d2a1c702ec440fe0c25dd23131bc3106` (immutable run provenance, not the current repository authority)
Output root: `/Volumes/onn. Drive/Neural Bridge/outputs/veatic21_event_optuna_stabilization_20260717`
Outer-test scores used: `false`
Training contract: checkpoints were eligible from epoch `1`; early stopping was forbidden before epoch `50`, with patience `100` and maximum epoch `5000`. Epoch `50` was a minimum training depth, not a checkpoint-selection cutoff.

The current executor was reapplied to the stored showdown rows after later path/workflow-only changes and reproduced every audit check and the exact primary metrics below.

## Frozen Optuna candidate

- trials: `50`
- best trial: `26`
- development objective: `0.0095972467`
- parameters: `{"alpha_cap": 0.04, "alpha_initial_logit": -4.0, "gate_bias": 6.0, "hidden": 48, "lambda_binary": 1.0, "learning_rate": 0.00014947725114481127, "weight_decay": 0.0003062953828798376}`

## Primary held-back 10 panels

- tuned minus original: mean `-0.0052929689`, median `-0.0002183180`, wins `5/10`, positive outer means `2/5`;
- tuned minus AR: mean `+0.0039664034`, median `+0.0092828944`, wins `8/10`, positive outer means `4/5`;
- original minus AR: mean `+0.0092593724`, median `+0.0115551847`.

## Secondary all 15 panels

- tuned minus original: mean `-0.0022893491`, median `+0.0005086129`, wins `8/15`;
- tuned minus AR: mean `+0.0043983825`, median `+0.0092219848`, wins `12/15`.

## Gate and audit

- gate checks: `{"tuned_delta_variability_no_worse_than_original": false, "tuned_mean_ensemble_uplift_positive": true, "tuned_mean_exceeds_ar": true, "tuned_mean_exceeds_original": false, "tuned_median_exceeds_ar": true, "tuned_median_exceeds_original": false, "tuned_member_mean_exceeds_ar": false, "tuned_member_mean_exceeds_original": true, "tuned_member_wins_at_least_30_of_50_vs_ar": false, "tuned_member_wins_at_least_30_of_50_vs_original": false, "tuned_outer_means_at_least_4_of_5_vs_ar": true, "tuned_outer_means_at_least_4_of_5_vs_original": false, "tuned_wins_at_least_7_of_10_vs_ar": true, "tuned_wins_at_least_7_of_10_vs_original": false}`
- audit: `{"ensemble_matrix_complete": true, "finite_metrics": true, "frozen_ar_shared_across_lanes": true, "label_digest_aligned_across_lanes": true, "member_matrix_complete": true, "outer_test_closed": true, "zero_event_policy_preserved": true}`

This is inner-only exploratory evidence. It does not authorize outer confirmation.

Trial `26` is rejected. Next, stop tuning the transferred AGAIN head and run fresh VEATIC 2.1 representation/head discovery with fold-safe fitting, matched controls, frozen AR, and no reused fitted OG VEATIC or AGAIN artifacts.
