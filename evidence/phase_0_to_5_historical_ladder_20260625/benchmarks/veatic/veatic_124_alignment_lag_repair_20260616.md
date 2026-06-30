# VEATIC 124 Alignment Lag Repair

After auditing the full 124-video VEATIC benchmark, the main timing issue appears to be video-specific or target-specific lag, not a safe global correction. The best supported correction is keep_current_0s_as_primary_plus_report_offset_diagnostics. This correction does improve interpretability but does not justify a test-derived global lag correction beyond AR, shuffled, random, and time/video controls. The strongest remaining claim is arousal spike/event ranking with balanced event-vs-stable and grouped-video validation. The claim still not supported is broad exact continuous future-value prediction. The next recommended step is a train-only lag-selection confirmatory run if a global offset is desired.

## Section 1: Executive Verdict

Reliable blocked best-offset counts: negative=12, zero=2, positive=6. Median=-1.5000, mean=-0.7750.

Classification: video-specific lag / target construction issue, with no safe global lag correction selected for final claims.

## Section 2: Manifest and Row Alignment

Videos audited: 124. Resampled/suspicious videos: 1.
- Video `83` raw rows 263 vs manifest rows 143; policy=linear_resampled_by_benchmark.

## Section 3: Offset Grid Results

See `veatic_124_alignment_offset_grid.csv` and `veatic_124_alignment_offset_grid_summary.md`.

## Section 4: Global vs Per-Video Lag

Per-video rows written: 1488. Enough-event rows should drive interpretation; low-event rows are flagged.

## Section 5: Causal Window / Smoothing Audit

Future-looking feature leakage found: 0.

## Section 6: Label Smoothing and Horizon Sweep

Label smoothing sensitivity rows: 48. Horizon sweep rows: 400.

## Section 7: Event Onset Definition Sweep

Onset definition rows: 48.

## Section 8: Balanced Event-vs-Stable Results

Balanced rebuild rows: 1920 using ratios 1:1, 1:2, 1:3, 1:5 and 50 seeds.

## Section 9: Cross-Correlation Lag Diagnostics

Cross-correlation lag rows: 238.

## Section 10: Interpolation and Resampling Sensitivity

Interpolation policy rows: 120. Subset sensitivity rows: 48.

## Section 11: AR Baseline Behavior

AR behavior rows: 20. Use predicted-positive-rate columns to identify AR over-firing.

## Section 12: Candidate Fixes

See `veatic_124_alignment_candidate_fixes.md`.

## Section 13: Confirmatory Rerun

Focused baseline-vs-best-offset rows written: 56.

## Section 14: Bootstrap Confidence

Video-level bootstrap CI rows written: 20.

## Section 15: Final Recommendation

Use current 0s alignment as the final non-leaky benchmark baseline, with offset-grid diagnostics reported transparently. Do not apply per-video or test-derived lag correction to final claims. Prefer event/ranking claims, balanced event-vs-stable rows, grouped-video validation, and bootstrap CIs. Avoid broad continuous-value prediction claims and avoid claiming pre-event early warning from recall alone.