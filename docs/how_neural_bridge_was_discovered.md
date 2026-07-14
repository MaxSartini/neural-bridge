# How Neural Bridge Was Discovered

Neural Bridge was not a single model win. It emerged from failures, corrections, target redesigns, and controlled confirmations. The invention is the combination of predicted neuro-response features, fold-safe compression, a matched frozen-AR floor, future-event target design, temporal residual learning, controls, and validation discipline.

## 1. VEATIC Established The Phenomenon

VEATIC-124 v2 showed that cortical-prediction-derived video features could rank future arousal spikes beyond AR and matched controls. The strongest blocked full-frame spike result reached PR-AUC `0.2536` versus AR `0.1969`, shuffled `0.1840`, and random `0.1944`. The balanced event-vs-stable result reached `0.3394`, beating AR by `+0.0609`.

## 2. AGAIN Made The Problem Harder

AGAIN scaled the work to `995` videos and `243,575` dense 2 Hz feature rows. Raw predicted cortical/fMRI features failed badly: blocked `raw_cortical_only` scored `0.124315` versus AR-only `0.203622`, while direct `AR_plus_raw_cortical` reached only `0.167731`. This established that the upstream features alone were not the result.

## 3. The Eval-Mode Correction Preserved Two Grouped Wins

Correct deterministic evaluation recovered grouped event-ranking and a separate controlled grouped continuous future-movement ranking/lift result. Across 15 fold-seed evaluations, real continuous Spearman was `0.2232222830` versus AR-only `0.1982207591`, shuffled `0.1938183619`, and random `0.1931781163`. Real top-1% average-true-movement lift was `0.1359465244` versus `0.1115815364`, `0.1125842464`, and `0.1136304212`. The stored `continuous_ranking_lift_pass` is `true`.

This is a real continuous-ranking victory. It is not exact-value trajectory forecasting and it is not a blocked continuous pass.

## 4. Blocked Evaluation Exposed Persistence

The older fused lane lost to AR and controls under blocked temporal evaluation. That failure showed that recent arousal persistence was the dominant hurdle. The project therefore stopped treating a grouped win as sufficient proof of forward-time generalization.

## 5. Frozen AR Made The Hurdle Explicit

Each seed or fold received its own fixed AR-only score. Real residual lanes and matched controls reused that identical frozen AR floor. This prevented the bridge from receiving an easier baseline than its controls and separated bridge-added signal from arousal momentum.

## 6. Washout-Gap Targets Changed The Scientific Question

Exact future values are noisy and highly autocorrelated. The redesigned target asked a sharper question: after an explicit gap from the current state, can the system rank a future response-relevant arousal event beyond persistence?

The promoted target, `future_arousal_max_delta_rows_4_10_train_q90`, leaves a washout gap before measuring future maximum arousal movement. The `short_temporal_conv_residual` head then learns causal event context over the frozen AR floor.

## 7. The Blocked Washout-Gap Result Passed

The 10-seed blocked confirmation reached PR-AUC `0.2670735630` versus matched frozen AR `0.2602336231` and best control `0.2593369051`. It was positive in `9/10` seeds versus both; weak, credible, and strong confirmation gates all passed.

This proves bounded strict forward-time future-event ranking for the promoted target/head.

## 8. The Same Head Passed Grouped Compatibility

Across five grouped-video folds and ten seeds, real PR-AUC was `0.2313831909` versus AR/frozen `0.2174953276` and best control `0.2174209937`. Real beat the best matched control in `50/50` fold-seeds. The current frozen-AR-residual-aware verdict passes with no failed gates.

The original grouped report appeared to fail because it used a raw-prevalence label-permutation-near-chance gate that does not fit a residual null retaining the frozen AR floor. The updated verdict corrected interpretation from existing artifacts; it did not retrain or rerun the benchmark.

## 9. What The Later Washout Continuous Diagnostic Did And Did Not Do

The redesigned washout continuous diagnostic improved Spearman by `+0.0055230967` versus frozen AR, but missed its top-5% lift and seed-consistency gates. That diagnostic did not fully pass. It also does not erase the earlier grouped continuous-ranking/lift pass, because the protocols and claims differ.

## Current Result

Neural Bridge now has three bounded victories:

1. Controlled future arousal event/spike ranking across VEATIC and AGAIN.
2. Bounded blocked washout-gap future-event ranking on AGAIN, beyond frozen AR and controls.
3. Controlled grouped continuous future-movement ranking/lift on AGAIN.

Exact continuous values, blocked continuous generalization, broad all-target/all-dataset temporal prediction, and 504 confirmation remain open.
