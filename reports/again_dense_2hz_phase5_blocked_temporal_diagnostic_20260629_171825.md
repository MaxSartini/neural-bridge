# Phase 5 Blocked-Temporal Diagnostic

Output root: `outputs/again_dense_2hz_phase5_adversarial_repair_fixplus_20260629_171825/`

## Executive Summary

The primary repair checkpoint is committed as `9757383c7e30d759fd15911e4ab87ee60b73fd86` (`phase5: add primary adversarial repair checkpoint`). The run completed `702/702` primary trainings with best-checkpoint restoration passing for `702/702` scored jobs.

The best explanation from existing artifacts is a combination of **A. AR/time autocorrelation dominance**, **D. real PCA being useful cross-video but not under strict forward-time blocked evaluation**, and a weaker version of **B. random/shuffled PCA acting as a less harmful perturbation than real PCA inside the fused head**. Blocked `ar_only_head` is stronger than both real and matched PCA controls, so random PCA does not improve over AR-only; it only beats the real PCA fused lane. There is no evidence of a control construction/evaluation bug from the normalized artifact checks, but the missing gate activations prevent a direct C/fusion-routing verdict.

Grouped-video is strong after repair: real `regression_plus_binary` PR-AUC `0.2282800471` versus matched control `0.2021634578`, delta `+0.0261165893`, positive in `15/15` fold-seed comparisons. Blocked-temporal still fails: real PR-AUC `0.2199977456` versus best matched blocked control `0.2288685899`, delta `-0.0088708442`. Strict forward-time temporal generalization remains unproven.

## Confirmed Commit Checkpoint

- Commit SHA: `9757383c7e30d759fd15911e4ab87ee60b73fd86`
- Commit message: `phase5: add primary adversarial repair checkpoint`
- Current run commit metadata predates that checkpoint because the run happened before commit, but the checkpoint commit now tracks the repair scripts, reports, and lightweight evidence bundle.

## What Grouped Proves Now

Grouped-video supports robust cross-video future arousal spike / emotional moment ranking for the primary lane. Real AR+PCA+diagnostics beats the best matched shuffled/random PCA control across every grouped fold-seed comparison for `regression_plus_binary`.

Weakest grouped `regression_plus_binary` delta: fold `4`, seed `20260627`, delta `0.013592392`. Strongest grouped delta: fold `2`, seed `20260625`, delta `0.036908738`.

## What Blocked Still Fails

Blocked-temporal does not pass the corrected matched-control gate. The best matched blocked control is `ar_plus_random_pca`, and it beats real AR+PCA+diagnostics under `regression_plus_binary` by `0.008870844` PR-AUC.

## Metric Table By Protocol / Loss / Control

| validation_protocol | control_type | loss_name | mean_pr_auc | mean_roc_auc | mean_top_1pct_recall | mean_top_5pct_recall | mean_top_10pct_recall | mean_continuous_pearson | mean_spearman_future_movement | folds | seeds | real_minus_control_pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blocked_temporal_70_30 | ar_only_head | binary | 0.258737 | 0.733992 | 0.039068 | 0.163173 | 0.287534 | 0.248056 | 0.207367 | 1 | 3 | -0.044303 |
| blocked_temporal_70_30 | ar_plus_random_pca | binary | 0.224888 | 0.697520 | 0.038684 | 0.147452 | 0.238454 | 0.254423 | 0.181916 | 1 | 3 | -0.010454 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | binary | 0.217377 | 0.674654 | 0.039409 | 0.143661 | 0.235898 | 0.234745 | 0.175367 | 1 | 3 | -0.002943 |
| blocked_temporal_70_30 | real_ar_pca_diag | binary | 0.214434 | 0.677554 | 0.035106 | 0.132498 | 0.231893 | 0.225382 | 0.169979 | 1 | 3 | 0.000000 |
| blocked_temporal_70_30 | video_mean_pca_oracle_diagnostic | binary | 0.190847 | 0.638309 | 0.030803 | 0.126789 | 0.215832 | 0.088948 | -0.043116 | 1 | 3 | 0.023587 |
| blocked_temporal_70_30 | pca_only_real | binary | 0.131355 | 0.549236 | 0.015678 | 0.069700 | 0.134288 | 0.060715 | 0.042894 | 1 | 3 | 0.083079 |
| blocked_temporal_70_30 | diag_only | binary | 0.119292 | 0.514559 | 0.011929 | 0.061776 | 0.119248 | 0.012746 | 0.017496 | 1 | 3 | 0.095142 |
| blocked_temporal_70_30 | quality_only | binary | 0.117327 | 0.519288 | 0.011205 | 0.053511 | 0.106339 | 0.020590 | 0.024739 | 1 | 3 | 0.097107 |
| blocked_temporal_70_30 | pca_only_shuffled | binary | 0.112788 | 0.504418 | 0.009756 | 0.049421 | 0.103442 | 0.005093 | 0.004108 | 1 | 3 | 0.101646 |
| blocked_temporal_70_30 | pca_only_random | binary | 0.110367 | 0.497886 | 0.008947 | 0.048696 | 0.098415 | -0.003122 | -0.003323 | 1 | 3 | 0.104067 |
| blocked_temporal_70_30 | label_permutation | binary | 0.107676 | 0.491833 | 0.007882 | 0.043882 | 0.092621 | -0.004666 | -0.004565 | 1 | 3 | 0.106758 |
| blocked_temporal_70_30 | time_only | binary | 0.107291 | 0.484240 | 0.009501 | 0.046183 | 0.094496 | -0.016596 | -0.006142 | 1 | 3 | 0.107143 |
| blocked_temporal_70_30 | ar_only_head | regression | 0.258812 | 0.731281 | 0.038684 | 0.165815 | 0.287108 | 0.337297 | 0.256275 | 1 | 3 | -0.072523 |
| blocked_temporal_70_30 | ar_plus_random_pca | regression | 0.208481 | 0.653046 | 0.038301 | 0.139230 | 0.227548 | 0.242245 | 0.165155 | 1 | 3 | -0.022192 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | regression | 0.204412 | 0.642887 | 0.036852 | 0.138420 | 0.226994 | 0.220482 | 0.159614 | 1 | 3 | -0.018122 |
| blocked_temporal_70_30 | real_ar_pca_diag | regression | 0.186290 | 0.622189 | 0.033913 | 0.121805 | 0.202624 | 0.189450 | 0.132964 | 1 | 3 | 0.000000 |
| blocked_temporal_70_30 | video_mean_pca_oracle_diagnostic | regression | 0.176557 | 0.616654 | 0.024242 | 0.115031 | 0.206203 | 0.121252 | 0.000110 | 1 | 3 | 0.009733 |
| blocked_temporal_70_30 | pca_only_real | regression | 0.125143 | 0.534689 | 0.014358 | 0.065908 | 0.125511 | 0.044481 | 0.035210 | 1 | 3 | 0.061146 |
| blocked_temporal_70_30 | diag_only | regression | 0.120287 | 0.517749 | 0.011673 | 0.063863 | 0.120612 | 0.023532 | 0.025935 | 1 | 3 | 0.066003 |
| blocked_temporal_70_30 | time_only | regression | 0.112020 | 0.500829 | 0.010481 | 0.051508 | 0.102633 | -0.000328 | -0.003595 | 1 | 3 | 0.074270 |
| blocked_temporal_70_30 | pca_only_shuffled | regression | 0.112006 | 0.502632 | 0.010182 | 0.050060 | 0.099480 | 0.003777 | 0.000523 | 1 | 3 | 0.074284 |
| blocked_temporal_70_30 | pca_only_random | regression | 0.111911 | 0.501381 | 0.010608 | 0.050571 | 0.100801 | 0.001542 | 0.001044 | 1 | 3 | 0.074379 |
| blocked_temporal_70_30 | quality_only | regression | 0.110984 | 0.502184 | 0.009373 | 0.047546 | 0.097520 | 0.003547 | 0.005673 | 1 | 3 | 0.075306 |
| blocked_temporal_70_30 | label_permutation | regression | 0.110679 | 0.489882 | 0.012100 | 0.048995 | 0.095475 | -0.011150 | -0.003131 | 1 | 3 | 0.075611 |
| blocked_temporal_70_30 | ar_only_head | regression_plus_binary | 0.260997 | 0.736023 | 0.039494 | 0.165772 | 0.288045 | 0.335784 | 0.209502 | 1 | 3 | -0.040999 |
| blocked_temporal_70_30 | ar_plus_random_pca | regression_plus_binary | 0.228869 | 0.700280 | 0.039238 | 0.149540 | 0.244632 | 0.204298 | 0.186961 | 1 | 3 | -0.008871 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | regression_plus_binary | 0.222617 | 0.682082 | 0.038940 | 0.148219 | 0.241309 | 0.171461 | 0.182004 | 1 | 3 | -0.002619 |
| blocked_temporal_70_30 | real_ar_pca_diag | regression_plus_binary | 0.219998 | 0.681336 | 0.037491 | 0.136887 | 0.236537 | 0.119582 | 0.175058 | 1 | 3 | 0.000000 |
| blocked_temporal_70_30 | video_mean_pca_oracle_diagnostic | regression_plus_binary | 0.192681 | 0.640864 | 0.031271 | 0.126406 | 0.220220 | 0.106911 | -0.041137 | 1 | 3 | 0.027316 |
| blocked_temporal_70_30 | pca_only_real | regression_plus_binary | 0.131689 | 0.550318 | 0.016871 | 0.070978 | 0.131944 | 0.026685 | 0.042842 | 1 | 3 | 0.088309 |
| blocked_temporal_70_30 | diag_only | regression_plus_binary | 0.118284 | 0.513848 | 0.010566 | 0.059901 | 0.115670 | 0.013971 | 0.014895 | 1 | 3 | 0.101714 |
| blocked_temporal_70_30 | quality_only | regression_plus_binary | 0.115668 | 0.514122 | 0.011460 | 0.052360 | 0.106297 | 0.007272 | 0.014574 | 1 | 3 | 0.104330 |
| blocked_temporal_70_30 | pca_only_shuffled | regression_plus_binary | 0.112918 | 0.504779 | 0.011077 | 0.052019 | 0.103315 | -0.001734 | 0.001886 | 1 | 3 | 0.107080 |
| blocked_temporal_70_30 | pca_only_random | regression_plus_binary | 0.111361 | 0.499793 | 0.010353 | 0.048355 | 0.098969 | 0.000916 | -0.003877 | 1 | 3 | 0.108636 |
| blocked_temporal_70_30 | label_permutation | regression_plus_binary | 0.110305 | 0.499028 | 0.008521 | 0.048100 | 0.099225 | -0.013685 | 0.004150 | 1 | 3 | 0.109693 |
| blocked_temporal_70_30 | time_only | regression_plus_binary | 0.109516 | 0.490724 | 0.009927 | 0.049676 | 0.098500 | -0.005127 | -0.000079 | 1 | 3 | 0.110482 |
| grouped_video | real_ar_pca_diag | binary | 0.227134 | 0.707012 | 0.044197 | 0.160720 | 0.269931 | 0.269692 | 0.220248 | 5 | 3 | 0.000000 |
| grouped_video | ar_only_head | binary | 0.221356 | 0.707076 | 0.039024 | 0.159404 | 0.273998 | 0.208772 | 0.187479 | 5 | 3 | 0.005777 |
| grouped_video | ar_plus_shuffled_pca | binary | 0.201402 | 0.671094 | 0.039204 | 0.152494 | 0.249593 | 0.230357 | 0.190213 | 5 | 3 | 0.025731 |
| grouped_video | ar_plus_random_pca | binary | 0.201390 | 0.670772 | 0.039018 | 0.153297 | 0.249849 | 0.231478 | 0.190217 | 5 | 3 | 0.025744 |
| grouped_video | pca_only_real | binary | 0.141760 | 0.587046 | 0.025448 | 0.097036 | 0.171052 | 0.121770 | 0.109673 | 5 | 3 | 0.085374 |
| grouped_video | quality_only | binary | 0.119196 | 0.538623 | 0.019872 | 0.074467 | 0.130327 | 0.061030 | 0.045641 | 5 | 3 | 0.107938 |
| grouped_video | diag_only | binary | 0.117972 | 0.543924 | 0.017095 | 0.070405 | 0.134253 | 0.062135 | 0.054976 | 5 | 3 | 0.109162 |
| grouped_video | time_only | binary | 0.110212 | 0.520669 | 0.015142 | 0.066967 | 0.123953 | 0.037656 | 0.039540 | 5 | 3 | 0.116922 |
| grouped_video | label_permutation | binary | 0.105957 | 0.508892 | 0.012699 | 0.061301 | 0.116675 | 0.005758 | 0.017611 | 5 | 3 | 0.121177 |
| grouped_video | video_mean_pca_oracle_diagnostic | binary | 0.105325 | 0.527894 | 0.008538 | 0.045448 | 0.098314 | 0.022073 | 0.016810 | 5 | 3 | 0.121809 |
| grouped_video | pca_only_random | binary | 0.100275 | 0.501068 | 0.010080 | 0.049470 | 0.098419 | 0.000104 | 0.000128 | 5 | 3 | 0.126859 |
| grouped_video | pca_only_shuffled | binary | 0.100143 | 0.499474 | 0.010538 | 0.049604 | 0.099687 | 0.001017 | -0.001512 | 5 | 3 | 0.126991 |
| grouped_video | ar_only_head | regression | 0.209870 | 0.679920 | 0.039174 | 0.157510 | 0.268527 | 0.281455 | 0.256303 | 5 | 3 | -0.003988 |
| grouped_video | real_ar_pca_diag | regression | 0.205882 | 0.657660 | 0.044046 | 0.153646 | 0.248533 | 0.240746 | 0.192767 | 5 | 3 | 0.000000 |
| grouped_video | ar_plus_shuffled_pca | regression | 0.187865 | 0.645159 | 0.037489 | 0.142250 | 0.233242 | 0.224625 | 0.186269 | 5 | 3 | 0.018016 |
| grouped_video | ar_plus_random_pca | regression | 0.184153 | 0.638111 | 0.037893 | 0.141122 | 0.227159 | 0.214253 | 0.176100 | 5 | 3 | 0.021729 |
| grouped_video | pca_only_real | regression | 0.129304 | 0.562686 | 0.021988 | 0.085111 | 0.151732 | 0.087855 | 0.084072 | 5 | 3 | 0.076577 |
| grouped_video | diag_only | regression | 0.119929 | 0.546295 | 0.018749 | 0.073133 | 0.135869 | 0.069271 | 0.063665 | 5 | 3 | 0.085953 |
| grouped_video | quality_only | regression | 0.111602 | 0.524490 | 0.017342 | 0.065786 | 0.121263 | 0.047542 | 0.038684 | 5 | 3 | 0.094279 |
| grouped_video | video_mean_pca_oracle_diagnostic | regression | 0.107084 | 0.519736 | 0.013689 | 0.053436 | 0.108535 | 0.016303 | 0.006244 | 5 | 3 | 0.098797 |
| grouped_video | time_only | regression | 0.105045 | 0.511365 | 0.012213 | 0.056881 | 0.110899 | 0.017670 | 0.018233 | 5 | 3 | 0.100836 |
| grouped_video | label_permutation | regression | 0.102104 | 0.492515 | 0.014438 | 0.055883 | 0.106855 | -0.006462 | -0.005436 | 5 | 3 | 0.103778 |
| grouped_video | pca_only_random | regression | 0.100409 | 0.500745 | 0.010155 | 0.050566 | 0.100195 | -0.000166 | -0.001128 | 5 | 3 | 0.105473 |
| grouped_video | pca_only_shuffled | regression | 0.100231 | 0.500393 | 0.009818 | 0.050242 | 0.099105 | 0.000190 | 0.000458 | 5 | 3 | 0.105651 |
| grouped_video | real_ar_pca_diag | regression_plus_binary | 0.228280 | 0.707675 | 0.045100 | 0.162678 | 0.270057 | 0.214564 | 0.220956 | 5 | 3 | 0.000000 |
| grouped_video | ar_only_head | regression_plus_binary | 0.221552 | 0.705361 | 0.039548 | 0.159238 | 0.275126 | 0.288399 | 0.193642 | 5 | 3 | 0.006728 |
| grouped_video | ar_plus_shuffled_pca | regression_plus_binary | 0.202163 | 0.671920 | 0.039421 | 0.152002 | 0.249574 | 0.201227 | 0.191390 | 5 | 3 | 0.026117 |
| grouped_video | ar_plus_random_pca | regression_plus_binary | 0.200705 | 0.669607 | 0.039259 | 0.153543 | 0.248030 | 0.190904 | 0.190498 | 5 | 3 | 0.027575 |
| grouped_video | pca_only_real | regression_plus_binary | 0.140774 | 0.585336 | 0.024910 | 0.097149 | 0.170026 | 0.079556 | 0.106377 | 5 | 3 | 0.087506 |
| grouped_video | diag_only | regression_plus_binary | 0.115385 | 0.537276 | 0.015705 | 0.065812 | 0.128519 | 0.052583 | 0.044859 | 5 | 3 | 0.112895 |
| grouped_video | time_only | regression_plus_binary | 0.111418 | 0.524016 | 0.015538 | 0.068267 | 0.124674 | 0.029792 | 0.038974 | 5 | 3 | 0.116862 |
| grouped_video | quality_only | regression_plus_binary | 0.108342 | 0.516253 | 0.014741 | 0.060146 | 0.113891 | 0.039511 | 0.022891 | 5 | 3 | 0.119938 |
| grouped_video | label_permutation | regression_plus_binary | 0.105665 | 0.509730 | 0.011916 | 0.060971 | 0.116611 | -0.006067 | 0.007729 | 5 | 3 | 0.122615 |
| grouped_video | video_mean_pca_oracle_diagnostic | regression_plus_binary | 0.104482 | 0.527003 | 0.008285 | 0.045312 | 0.094439 | 0.016569 | 0.017020 | 5 | 3 | 0.123798 |
| grouped_video | pca_only_random | regression_plus_binary | 0.100288 | 0.501332 | 0.009367 | 0.049487 | 0.099515 | -0.000265 | 0.000289 | 5 | 3 | 0.127992 |
| grouped_video | pca_only_shuffled | regression_plus_binary | 0.100039 | 0.498552 | 0.010095 | 0.051072 | 0.100899 | -0.001931 | -0.002101 | 5 | 3 | 0.128241 |

Requested controls not separately present: `ar_plus_diag`; `ar_plus_diag_plus_random_pca` is implemented as `ar_plus_random_pca`; `ar_plus_diag_plus_shuffled_pca` is implemented as `ar_plus_shuffled_pca`.

## Loss-Specific Diagnosis

| validation_protocol | loss_name | real_pr_auc | ar_only_pr_auc | best_matched_control_type | best_matched_control_pr_auc | real_minus_ar_only | real_minus_best_matched_control | best_matched_minus_ar_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_video | regression | 0.205882 | 0.209870 | ar_plus_shuffled_pca | 0.187865 | -0.003988 | 0.018016 | -0.022005 |
| grouped_video | binary | 0.227134 | 0.221356 | ar_plus_shuffled_pca | 0.201402 | 0.005777 | 0.025731 | -0.019954 |
| grouped_video | regression_plus_binary | 0.228280 | 0.221552 | ar_plus_shuffled_pca | 0.202163 | 0.006728 | 0.026117 | -0.019388 |
| blocked_temporal_70_30 | regression | 0.186290 | 0.258812 | ar_plus_random_pca | 0.208481 | -0.072523 | -0.022192 | -0.050331 |
| blocked_temporal_70_30 | binary | 0.214434 | 0.258737 | ar_plus_random_pca | 0.224888 | -0.044303 | -0.010454 | -0.033849 |
| blocked_temporal_70_30 | regression_plus_binary | 0.219998 | 0.260997 | ar_plus_random_pca | 0.228869 | -0.040999 | -0.008871 | -0.032129 |

Blocked real loses to the best matched control for every loss: `True`. Grouped real beats the best matched control for every loss: `True`.

## Fold / Seed Consistency

Blocked `regression_plus_binary` fold-seed deltas:

| fold | seed | real_pr_auc | best_control_type | best_control_pr_auc | real_minus_best_control_pr_auc | real_beats_best_control |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20260625 | 0.221386 | ar_plus_random_pca | 0.227763 | -0.006377 | False |
| 1 | 20260626 | 0.213926 | ar_plus_random_pca | 0.226692 | -0.012766 | False |
| 1 | 20260627 | 0.224681 | ar_plus_random_pca | 0.232151 | -0.007470 | False |

Grouped `regression_plus_binary` fold-seed deltas:

| fold | seed | real_pr_auc | best_control_type | best_control_pr_auc | real_minus_best_control_pr_auc | real_beats_best_control |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20260625 | 0.229618 | ar_plus_shuffled_pca | 0.208768 | 0.020850 | True |
| 1 | 20260626 | 0.224899 | ar_plus_shuffled_pca | 0.205802 | 0.019097 | True |
| 1 | 20260627 | 0.226692 | ar_plus_shuffled_pca | 0.208823 | 0.017870 | True |
| 2 | 20260625 | 0.238223 | ar_plus_shuffled_pca | 0.201315 | 0.036909 | True |
| 2 | 20260626 | 0.236333 | ar_plus_shuffled_pca | 0.204218 | 0.032116 | True |
| 2 | 20260627 | 0.239357 | ar_plus_shuffled_pca | 0.202903 | 0.036454 | True |
| 3 | 20260625 | 0.219069 | ar_plus_shuffled_pca | 0.187800 | 0.031269 | True |
| 3 | 20260626 | 0.220329 | ar_plus_shuffled_pca | 0.186615 | 0.033714 | True |
| 3 | 20260627 | 0.223421 | ar_plus_random_pca | 0.189707 | 0.033715 | True |
| 4 | 20260625 | 0.213969 | ar_plus_shuffled_pca | 0.197870 | 0.016098 | True |
| 4 | 20260626 | 0.217360 | ar_plus_random_pca | 0.201622 | 0.015738 | True |
| 4 | 20260627 | 0.216723 | ar_plus_shuffled_pca | 0.203131 | 0.013592 | True |
| 5 | 20260625 | 0.239581 | ar_plus_shuffled_pca | 0.212572 | 0.027009 | True |
| 5 | 20260626 | 0.241577 | ar_plus_shuffled_pca | 0.213242 | 0.028335 | True |
| 5 | 20260627 | 0.237050 | ar_plus_random_pca | 0.214946 | 0.022104 | True |

## Evidence For / Against AR Dominance

The blocked result is heavily AR-driven. Under `regression_plus_binary`, `ar_only_head` is the strongest blocked lane in the focused comparison: PR-AUC `0.260997`, above real `0.219998` and above `ar_plus_random_pca` `0.228869`. Adding random PCA to AR+diagnostics does not improve over AR-only; it reduces performance relative to AR-only but degrades less than the real PCA fused lane. Adding real PCA does not improve over AR-only and does not beat matched controls under blocked-temporal validation.

| validation_protocol | control_type | mean_pr_auc | mean_roc_auc | mean_top_1pct_recall | mean_continuous_pearson | real_minus_control_pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| blocked_temporal_70_30 | ar_only_head | 0.260997 | 0.736023 | 0.039494 | 0.335784 | -0.040999 |
| blocked_temporal_70_30 | ar_plus_random_pca | 0.228869 | 0.700280 | 0.039238 | 0.204298 | -0.008871 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | 0.222617 | 0.682082 | 0.038940 | 0.171461 | -0.002619 |
| blocked_temporal_70_30 | real_ar_pca_diag | 0.219998 | 0.681336 | 0.037491 | 0.119582 | 0.000000 |
| blocked_temporal_70_30 | pca_only_real | 0.131689 | 0.550318 | 0.016871 | 0.026685 | 0.088309 |
| blocked_temporal_70_30 | diag_only | 0.118284 | 0.513848 | 0.010566 | 0.013971 | 0.101714 |
| blocked_temporal_70_30 | quality_only | 0.115668 | 0.514122 | 0.011460 | 0.007272 | 0.104330 |
| blocked_temporal_70_30 | pca_only_shuffled | 0.112918 | 0.504779 | 0.011077 | -0.001734 | 0.107080 |
| blocked_temporal_70_30 | pca_only_random | 0.111361 | 0.499793 | 0.010353 | 0.000916 | 0.108636 |
| blocked_temporal_70_30 | time_only | 0.109516 | 0.490724 | 0.009927 | -0.005127 | 0.110482 |
| grouped_video | real_ar_pca_diag | 0.228280 | 0.707675 | 0.045100 | 0.214564 | 0.000000 |
| grouped_video | ar_only_head | 0.221552 | 0.705361 | 0.039548 | 0.288399 | 0.006728 |
| grouped_video | ar_plus_shuffled_pca | 0.202163 | 0.671920 | 0.039421 | 0.201227 | 0.026117 |
| grouped_video | ar_plus_random_pca | 0.200705 | 0.669607 | 0.039259 | 0.190904 | 0.027575 |
| grouped_video | pca_only_real | 0.140774 | 0.585336 | 0.024910 | 0.079556 | 0.087506 |
| grouped_video | diag_only | 0.115385 | 0.537276 | 0.015705 | 0.052583 | 0.112895 |
| grouped_video | time_only | 0.111418 | 0.524016 | 0.015538 | 0.029792 | 0.116862 |
| grouped_video | quality_only | 0.108342 | 0.516253 | 0.014741 | 0.039511 | 0.119938 |
| grouped_video | pca_only_random | 0.100288 | 0.501332 | 0.009367 | -0.000265 | 0.127992 |
| grouped_video | pca_only_shuffled | 0.100039 | 0.498552 | 0.010095 | -0.001931 | 0.128241 |

This supports A: AR/time autocorrelation dominance is a major mechanism in blocked-temporal scoring.

## Evidence For / Against Random PCA Regularization

Training diagnostics show the random/shuffled matched controls behave more like perturbations of the AR path than meaningful visual representation controls. Existing artifacts do not include train-vs-validation PR-AUC gaps, but they do include best-vs-final inner validation drops, epoch counts, overfit flags, and validation-curve volatility.

| validation_protocol | control_type | loss_name | mean_epochs_run | median_epochs_run | overfit_flags | runs | mean_best_minus_final_inner_pr_auc | mean_best_inner_validation_pr_auc | mean_final_inner_validation_pr_auc | mean_validation_curve_diff_std | mean_validation_curve_final_minus_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blocked_temporal_70_30 | ar_only_head | regression_plus_binary | 150.666667 | 177.000000 | 1 | 3 | 0.001435 | 0.235264 | 0.233829 | 0.006364 | -0.001435 |
| blocked_temporal_70_30 | ar_plus_random_pca | regression_plus_binary | 34.666667 | 35.000000 | 3 | 3 | 0.069532 | 0.214756 | 0.145223 | 0.009999 | -0.069532 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | regression_plus_binary | 35.000000 | 35.000000 | 3 | 3 | 0.058312 | 0.216880 | 0.158568 | 0.009667 | -0.058312 |
| blocked_temporal_70_30 | real_ar_pca_diag | regression_plus_binary | 31.666667 | 31.000000 | 3 | 3 | 0.086993 | 0.220008 | 0.133015 | 0.011125 | -0.086993 |
| grouped_video | ar_only_head | regression_plus_binary | 148.733333 | 149.000000 | 10 | 15 | 0.002949 | 0.282099 | 0.279150 | 0.003814 | -0.002949 |
| grouped_video | ar_plus_random_pca | regression_plus_binary | 34.333333 | 34.000000 | 15 | 15 | 0.057424 | 0.251370 | 0.193946 | 0.011826 | -0.057424 |
| grouped_video | ar_plus_shuffled_pca | regression_plus_binary | 36.133333 | 35.000000 | 15 | 15 | 0.036489 | 0.253568 | 0.217079 | 0.011450 | -0.036489 |
| grouped_video | real_ar_pca_diag | regression_plus_binary | 31.800000 | 32.000000 | 15 | 15 | 0.082123 | 0.262611 | 0.180488 | 0.013425 | -0.082123 |

This supports only a cautious version of B: random PCA improves blocked PR-AUC over real while not explaining grouped-video improvement, but it does not improve over AR-only. It may be less damaging than real PCA in the blocked fused head, or it may perturb the gate/branch path differently. Direct proof requires gate and branch logging.

## Evidence For / Against Gate-Routing Issue

Gate activations, branch weights, gate saturation, branch norms, and gate correlations were not logged. Therefore C cannot be directly adjudicated from existing artifacts.

Minimal no-sweep instrumented rerun if needed: Instrument gate logging only for controls real_ar_pca_diag, ar_plus_random_pca, ar_plus_shuffled_pca, and ar_only_head; protocols grouped_video and blocked_temporal_70_30; loss regression_plus_binary; seeds 20260625, 20260626, 20260627; model gated_ar_pca_mlp only; same target and feature; no secondary heads or targets. Log gate mean, saturation, gate-vs-AR-score correlation, gate-vs-PCA-norm correlation, branch norms, and per-row prediction files for the blocked fold.

## Evidence For / Against Split / Prevalence Artifact

Blocked split prevalence and coverage:

- Train positive rate: `0.100523`
- Test positive rate: `0.111245`
- Train event count: `16361`
- Test event count: `7824`
- Test rows: `70331`
- Videos with test positives: `683/995`
- Video positive-rate quantiles: `{"0.0": 0.0, "0.25": 0.0, "0.5": 0.08333333333333333, "0.75": 0.18440643863179074, "0.9": 0.28410462776659967, "0.99": 0.4647887323943662, "1.0": 0.6056338028169014}`

Per-video within-video metrics exist, but per-row predictions and quality/motion/luma-by-video prediction joins do not. Random-control advantage is measurable by per-video PR-AUC deltas, but attribution to quality/motion/luma requires additional logged joins. Existing per-video deltas were written to `diagnostics/ar_random_control_advantage_analysis.csv`.

Video-level blocked real-minus-random summary: real wins on `319` videos, random wins on `364` videos, mean video delta `-0.003766`, event-rate correlation `-0.017940`.

This makes F possible but not established as the dominant mechanism.

## Label Permutation Sanity

Label permutation collapses near chance/prevalence: grouped `regression_plus_binary` PR-AUC `0.1056653365`; blocked `regression_plus_binary` PR-AUC `0.1103050201`. This argues against gross leakage.

## Video-Mean Diagnostic

The video-mean PCA oracle diagnostic does not explain grouped real performance: grouped `regression_plus_binary` PR-AUC is `0.1044820659`, near chance and far below real `0.2282800471`. For blocked, video-mean PCA is non-trivial at `0.1926814853`, below real and below the best matched control. Static/video mean content may partially align with blocked behavior but does not fully explain it.

## Within-Video And Continuous Support

Within-video and continuous ranking metrics support a ranking signal, but they do not override the corrected blocked matched-control PR-AUC failure.

| validation_protocol | control_type | videos | mean_video_pr_auc | mean_video_spearman | mean_top_1pct_lift | mean_top_5pct_lift | mean_top_10pct_lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blocked_temporal_70_30 | ar_only_head | 995 | 0.322461 | 0.156832 | 1.945454 | 1.947070 | 1.834672 |
| blocked_temporal_70_30 | ar_plus_random_pca | 995 | 0.301638 | 0.143283 | 1.795303 | 1.739076 | 1.659782 |
| blocked_temporal_70_30 | ar_plus_shuffled_pca | 995 | 0.288451 | 0.123762 | 1.967972 | 1.701285 | 1.584521 |
| blocked_temporal_70_30 | real_ar_pca_diag | 995 | 0.297872 | 0.148796 | 1.739234 | 1.637272 | 1.585381 |
| grouped_video | ar_only_head | 995 | 0.240682 | 0.118863 | 3.292907 | 2.711142 | 2.230597 |
| grouped_video | ar_plus_random_pca | 995 | 0.221927 | 0.105572 | 3.020545 | 2.370055 | 2.008600 |
| grouped_video | ar_plus_shuffled_pca | 995 | 0.222291 | 0.107779 | 3.075790 | 2.373798 | 2.004893 |
| grouped_video | real_ar_pca_diag | 995 | 0.260242 | 0.242530 | 3.436479 | 2.869537 | 2.499711 |

Known real-lane anchors: grouped within-video PR-AUC `0.2602418556`, Spearman `0.2425295289`, top-1% lift `3.4364789478`; blocked within-video PR-AUC `0.2978716136`, Spearman `0.1487958181`, top-1% lift `1.7392340268`.

## Artifact Quality Check

| path | exists | json_parses |
| --- | --- | --- |
| promotion/corrected_promotion_gates.json | True | True |
| promotion/promotion_gates.json | True | True |
| promotion/adversarial_verdict.json | True | True |
| promotion/failure_reasons.json | True | True |
| diagnostics/training_headroom_audit.json | True | True |
| diagnostics/adversarial_repair_artifact_completeness_audit.json | True | True |

`corrected_promotion_gates.json`, `promotion_gates.json`, `adversarial_verdict.json`, `failure_reasons.json`, `training_headroom_audit.json`, and artifact completeness audit are present and parse cleanly. No missing artifact blocks interpretation.

## Best Explanation For Blocked Random-Control Win

The most defensible explanation is mixed:

1. Blocked-temporal scoring is dominated by AR/time structure; `ar_only_head` is stronger than real and matched PCA fused controls in blocked `regression_plus_binary`.
2. Random/shuffled PCA does not improve over AR-only, but it degrades less than real PCA inside the blocked fused head.
3. Real PCA adds robust cross-video information, but that information does not yet prove strict forward-time temporal generalization.
4. Gate/fusion routing could explain why real PCA is more harmful than random/shuffled PCA under blocked validation, but existing artifacts do not log gates or branch behavior.

## Missing Instrumentation

Missing artifacts: per-row predictions, gate activations, branch norms, gate saturation, gate-vs-AR-score correlations, gate-vs-PCA-norm correlations, and quality/motion/luma joins for per-video prediction deltas.

## Minimal Next Rerun Recommendation

New training is needed only if we want mechanism-level confirmation, especially for C/gate routing and B/random-regularization. Do not rerun the full 702 matrix. Minimal next run:

`Instrument gate logging only for controls real_ar_pca_diag, ar_plus_random_pca, ar_plus_shuffled_pca, and ar_only_head; protocols grouped_video and blocked_temporal_70_30; loss regression_plus_binary; seeds 20260625, 20260626, 20260627; model gated_ar_pca_mlp only; same target and feature; no secondary heads or targets. Log gate mean, saturation, gate-vs-AR-score correlation, gate-vs-PCA-norm correlation, branch norms, and per-row prediction files for the blocked fold.`

## Corrected Claim

Robust cross-video future arousal spike / emotional moment ranking is strengthened. Strict forward-time temporal generalization remains unproven.

## Do-Not-Do List

- Do not expand to secondary heads until blocked mechanism is understood.
- Do not claim strict forward-time temporal generalization.
- Do not use old holy_shit gate.
- Do not compare regression_plus_binary real against binary-only controls.
- Do not rerun the full 702 matrix unless diagnostics require it.
