# VEATIC 124 Manifest Validation

- Manifest: `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- Cache: `<external-assets-root>/benchmarks/veatic/tribe_cache`
- Videos: 124/124 complete
- Rows: 10357 total; per-video min/median/max = 11/78.5/179
- Rejected for cache: 0
- Nonfinite labels: 0; nonfinite predictions: 0
- Feature alignment: {'exact': 123, 'linear_resampled_by_benchmark': 1}

## Alignment Notes
- Video `83` raw predictions [263, 20484] vs manifest rows 143; policy: included; benchmark resamples rows linearly to manifest length.

## Primary Event Counts

| Target | Rows | Events | Positive rate | Zero-event videos |
|---|---:|---:|---:|---:|
| `arousal__future_spike_1_3s@0.05` | 10233 | 1681 | 0.1643 | 0 |
| `arousal__future_spike_1_3s@0.075` | 10233 | 992 | 0.0969 | 7 |
| `arousal__future_change_p2s_movement@0.05` | 10109 | 1679 | 0.1661 | 2 |
| `arousal__future_change_p3s_movement@0.05` | 9985 | 2582 | 0.2586 | 1 |
| `arousal__future_change_p3s_movement@0.075` | 9985 | 1457 | 0.1459 | 5 |

Zero-event videos are expected to skew thresholded and MAE-adjacent summaries; PR-AUC/top-k and balanced event-vs-stable diagnostics should carry more interpretive weight for event claims.