# VEATIC 124 Confirmatory Benchmark Report

Generated: 2026-06-16

## Executive Verdict

**Promising event/ranking signal, not a broad continuous-MAE win.** On the 124-video confirmatory run, cortical/TRIBE features improve blocked arousal spike ranking over autoregressive, shuffled, random, timestamp, and video/time controls. The strongest blocked full-frame spike row is `cortical_pca64_delta` at threshold 0.05 with PR-AUC 0.2536 versus AR 0.1969, shuffled 0.1840, and random 0.1944.

The continuous future-change rows remain diagnostic only: zero-change MAE beats real cortical on most continuous checks, which is expected when stable/zero-change frames dominate. This report therefore treats PR-AUC, balanced event-vs-stable sampling, top-k/recall diagnostics, and grouped video generalization as the main evidence.

## Run Policy

- No new architectures were added.
- No feature extraction was changed and no videos were re-encoded.
- Cached TRIBE cortical outputs were reused from `<external-assets-root>/benchmarks/veatic/tribe_cache`.
- PCA backend: `mps_gram` with exact MPS Gram products and CPU top-eigenpair solve.
- Ridge/scoring backend: `cpu_pinv`.
- PCA is fit once per `feature_mode + split/fold`, then reused across all targets, thresholds, masks, and metrics for that split/fold.
- Video-generalization split: grouped video 5-fold, no train/test rows from the same video in a fold.

## Manifest Validation

- Videos: 124/124 complete.
- Rows: 10357 total.
- Rejected for cache: 0.
- Nonfinite labels: 0.
- Feature alignment counts: `{'exact': 123, 'linear_resampled_by_benchmark': 1}`.
- Video `83` has raw predictions [263, 20484] vs 143 manifest rows; no fixed alternate cache was found, so the benchmark included it with the existing linear-resample policy.

Primary event-count caveat: zero-event videos exist for stricter thresholds, so per-video and MAE-adjacent summaries can be skewed. Use PR-AUC/top-k and balanced event-vs-stable rows for the main signal read.

| Target | Rows | Events | Positive rate | Zero-event videos |
|---|---:|---:|---:|---:|
| `arousal__future_spike_1_3s@0.05` | 10233 | 1681 | 0.1643 | 0 |
| `arousal__future_spike_1_3s@0.075` | 10233 | 992 | 0.0969 | 7 |
| `arousal__future_change_p2s_movement@0.05` | 10109 | 1679 | 0.1661 | 2 |
| `arousal__future_change_p3s_movement@0.05` | 9985 | 2582 | 0.2586 | 1 |
| `arousal__future_change_p3s_movement@0.075` | 9985 | 1457 | 0.1459 | 5 |

## Blocked Full-Frame Primary Rows

| Feature | Target | Thr | Events | Real PR-AUC | AR | Shuf | Rand | Real F1 | Pass controls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 411 | 0.3101 | 0.3049 | 0.3106 | 0.3076 | 0.2393 | False |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 411 | 0.3040 | 0.3049 | 0.3023 | 0.2992 | 0.2162 | False |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 411 | 0.3007 | 0.3049 | 0.2684 | 0.2587 | 0.2689 | False |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 411 | 0.2996 | 0.3049 | 0.2897 | 0.2932 | 0.2019 | False |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 668 | 0.3448 | 0.3376 | 0.3388 | 0.3379 | 0.1337 | True |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 668 | 0.3458 | 0.3376 | 0.3293 | 0.3354 | 0.1373 | True |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 668 | 0.3762 | 0.3376 | 0.3368 | 0.3280 | 0.3548 | True |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 668 | 0.3858 | 0.3376 | 0.3446 | 0.3335 | 0.2017 | True |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 363 | 0.2763 | 0.2729 | 0.2707 | 0.2735 | 0.2916 | True |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 363 | 0.3008 | 0.2729 | 0.2679 | 0.2707 | 0.3309 | True |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 363 | 0.2503 | 0.2729 | 0.2410 | 0.2318 | 0.2744 | False |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 363 | 0.2853 | 0.2729 | 0.2719 | 0.2709 | 0.3002 | True |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 392 | 0.2052 | 0.1969 | 0.1980 | 0.2007 | 0.2338 | False |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 392 | 0.2196 | 0.1969 | 0.1988 | 0.1956 | 0.2362 | True |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 392 | 0.2536 | 0.1969 | 0.1840 | 0.1944 | 0.2677 | True |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 392 | 0.2455 | 0.1969 | 0.1981 | 0.1930 | 0.2510 | True |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 228 | 0.1330 | 0.1151 | 0.1148 | 0.1148 | 0.1465 | True |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 228 | 0.1500 | 0.1151 | 0.1167 | 0.1175 | 0.1548 | True |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 228 | 0.1614 | 0.1151 | 0.1190 | 0.1208 | 0.1682 | True |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 228 | 0.1575 | 0.1151 | 0.1141 | 0.1147 | 0.1563 | True |

## Balanced Event-vs-Stable Rows

Shown for 1:3 event/pre-event positives versus stable negatives. These are the cleanest event-conditioned discrimination rows because positive-only masks make PR-AUC undefined.

| Feature | Target | Thr | Real PR-AUC | vs AR | vs Shuf | vs Rand | F1 | BalAcc | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cortical_fast_default | `arousal__future_change_p2s_movement` | 0.0500 | 0.3952 | 0.0021 | 0.0036 | 0.0027 | 0.1770 | 0.5363 | 0.1064 |
| cortical_global_delta | `arousal__future_change_p2s_movement` | 0.0500 | 0.3897 | -0.0034 | -0.0012 | -0.0026 | 0.1472 | 0.5316 | 0.0842 |
| cortical_pca64_delta | `arousal__future_change_p2s_movement` | 0.0500 | 0.3807 | -0.0123 | 0.0149 | 0.0014 | 0.2083 | 0.5419 | 0.1322 |
| cortical_pca_64 | `arousal__future_change_p2s_movement` | 0.0500 | 0.3894 | -0.0037 | -0.0052 | 0.0149 | 0.1457 | 0.5295 | 0.0842 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0500 | 0.4512 | 0.0024 | 0.0025 | 0.0037 | 0.1068 | 0.5216 | 0.0579 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0.4507 | 0.0019 | 0.0050 | 0.0035 | 0.1027 | 0.5218 | 0.0553 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0500 | 0.4634 | 0.0146 | 0.0249 | 0.0290 | 0.3514 | 0.5562 | 0.2737 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0500 | 0.4797 | 0.0310 | 0.0318 | 0.0335 | 0.1669 | 0.5307 | 0.0967 |
| cortical_fast_default | `arousal__future_change_p3s_movement` | 0.0750 | 0.3609 | -0.0041 | -0.0026 | -0.0030 | 0.4141 | 0.5759 | 0.6569 |
| cortical_global_delta | `arousal__future_change_p3s_movement` | 0.0750 | 0.3794 | 0.0145 | 0.0183 | 0.0272 | 0.3584 | 0.5778 | 0.3372 |
| cortical_pca64_delta | `arousal__future_change_p3s_movement` | 0.0750 | 0.3549 | -0.0100 | 0.0168 | 0.0086 | 0.3721 | 0.5652 | 0.4428 |
| cortical_pca_64 | `arousal__future_change_p3s_movement` | 0.0750 | 0.3741 | 0.0092 | 0.0225 | 0.0181 | 0.3888 | 0.5739 | 0.4868 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0500 | 0.2850 | 0.0065 | 0.0066 | 0.0049 | 0.3975 | 0.4985 | 0.9778 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0500 | 0.2970 | 0.0185 | 0.0186 | 0.0189 | 0.3960 | 0.5001 | 0.9512 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0500 | 0.3394 | 0.0609 | 0.0631 | 0.0476 | 0.4129 | 0.5510 | 0.8225 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0500 | 0.3276 | 0.0491 | 0.0519 | 0.0506 | 0.4098 | 0.5302 | 0.9305 |
| cortical_fast_default | `arousal__future_spike_1_3s` | 0.0750 | 0.2990 | 0.0075 | 0.0064 | 0.0062 | 0.4017 | 0.5101 | 0.9603 |
| cortical_global_delta | `arousal__future_spike_1_3s` | 0.0750 | 0.3143 | 0.0228 | 0.0244 | 0.0210 | 0.4126 | 0.5357 | 0.9276 |
| cortical_pca64_delta | `arousal__future_spike_1_3s` | 0.0750 | 0.3460 | 0.0544 | 0.0630 | 0.0505 | 0.4035 | 0.5414 | 0.7850 |
| cortical_pca_64 | `arousal__future_spike_1_3s` | 0.0750 | 0.3331 | 0.0415 | 0.0455 | 0.0410 | 0.4064 | 0.5331 | 0.8692 |

## Grouped Video Generalization

The main benchmark used grouped video 5-fold validation for the video-generalized split. For arousal future spike at the fixed 0.05 event threshold, PCA modes improved aggregate grouped F1 over AR.

| Feature | Folds | AR F1 | Real F1 | Delta | Event rate |
|---|---:|---:|---:|---:|---:|
| cortical_global | 5 | 0.3899 | 0.3955 | 0.0056 | 0.1643 |
| cortical_global_delta | 5 | 0.3899 | 0.3940 | 0.0041 | 0.1643 |
| cortical_pca_64 | 5 | 0.3899 | 0.4155 | 0.0256 | 0.1643 |
| cortical_pca64_delta | 5 | 0.3899 | 0.4077 | 0.0177 | 0.1643 |

## Alignment / Shift Audit

Best-shift counts across focused blocked rows: negative=10, zero=3, positive=3. Non-zero best shifts are not automatically leakage, but the spike rows often prefer negative offsets, so alignment/annotation lag remains a real follow-up item before any strong timing claim.

## Continuous Targets

Continuous future-change MAE beat the zero-change baseline in 5/24 continuous checks. This is not enough for a continuous-value prediction claim. Continuous rows are useful as diagnostics and for sign/movement target construction, not as the headline.

## Output Artifacts

- Manifest validation: `benchmarks/veatic/veatic_manifest_124_validation_20260616.md`
- Main benchmark JSONs/summaries: `benchmarks/veatic/veatic_neuro_benchmark_124video_*_mpsgram_cpu.*`
- Event/spike retest: `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md`
- Event-conditioned retest: `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md`

## Bottom Line

The 124-video run supports a cautious Neuro Bridge claim for arousal event/spike ranking under blocked and grouped-video validation. It does not support overselling exact continuous future arousal-value prediction. The next scientific risk to resolve is temporal alignment: several spike rows improve at non-zero offsets, so the timing interpretation needs a dedicated alignment/annotation-lag pass.