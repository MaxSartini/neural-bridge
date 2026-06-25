# AGAIN Real Scout Selector Validation

## Executive Verdict
- The real 10-video smoke completed, then the 50-video validation completed from saved AGAIN videos/logs.
- ViT-B scout is fast enough for scout use: 1508 windows over 50 videos, mean total scout time 9.23s/video, median 10.92s/video.
- Uncached ViT-B forward time averaged 8.23s/video; decode+preprocess averaged 1.97s/video.
- The current hybrid selector did not beat same-budget random at top-5%: hybrid spike recall 47.8%, random 53.0%.
- ViT-B novelty alone was weak at top-5%: 30.4% spike recall versus telemetry-change 53.4%.
- Oracle top-5% reached 69.3% spike recall at 28.5% coverage, so the dataset contains recoverable spike structure, but the deployable selector is not there yet.
- Recommendation: do not spend a 1000/2000-window ViT-G/TRIBE teacher budget yet. If a teacher run is required now, cap it at 500 windows and mark it exploratory; otherwise improve selector scoring first.

## Guardrails
- AGAIN only.
- MLX only.
- No CUDA.
- No dense ViT-G/TRIBE encoding.
- No training.
- Arousal labels were used only for evaluation and the oracle upper-bound selector, not for deployable selectors.

## Throughput
- Videos validated: 50
- Total ViT-B windows processed: 1508
- Cache hits: 10
- Mean total scout time/video: 9.228s
- Median total scout time/video: 10.918s
- Mean ViT-B forward time/video, all videos: 6.587s
- Mean ViT-B forward time/video, uncached only: 8.233s
- Mean decode+preprocess time/video, uncached only: 1.972s
- Audio available in cheap scan: 0/50 videos
- Peak RSS observed: 8.15 GB

## Top-5 Selector Comparison
| selector | selected % video | spike recall | pre T-2 recall | pre T-4 recall | precision event/pre8 |
|---|---:|---:|---:|---:|---:|
| telemetry_change | 49.8% | 53.4% | 52.5% | 53.7% | 46.3% |
| cheap_video_audio | 38.0% | 43.8% | 44.9% | 45.2% | 48.1% |
| vjepa_b_novelty | 25.0% | 30.4% | 31.3% | 32.1% | 48.0% |
| telemetry_plus_video_audio | 43.7% | 51.4% | 51.8% | 50.8% | 49.9% |
| hybrid_telemetry_video_audio_vjepa_b | 39.2% | 47.8% | 48.3% | 47.1% | 49.3% |
| random_same_budget | 51.3% | 53.0% | 54.2% | 53.1% | 44.2% |
| oracle_spike_upper_bound | 28.5% | 69.3% | 72.2% | 73.6% | 93.4% |

## Hybrid Budget Sweep
| budget | selected % video | spike recall | pre T-2 recall | pre T-8 recall |
|---|---:|---:|---:|---:|
| top2pct | 20.6% | 25.2% | 26.6% | 24.6% |
| top5pct | 39.2% | 47.8% | 48.3% | 44.2% |
| top10pct | 58.8% | 68.2% | 69.0% | 66.9% |
| max30 | 87.5% | 89.9% | 90.5% | 91.2% |
| max60 | 98.6% | 98.9% | 99.1% | 99.7% |
| max120 | 100.0% | 100.0% | 100.0% | 100.0% |

## Answers
- Is ViT-B scout fast enough on real AGAIN videos? Yes for a locator/scout pass; the uncached forward is about 8.2s/video on this 50-video subset, with about 10 to 11s total typical wall time per video.
- Does ViT-B novelty add useful candidate recall beyond telemetry? Not in this run. ViT-B novelty alone underperformed telemetry-change, and the hybrid did not improve top-5 recall over telemetry-change.
- Does hybrid selection beat same-budget random? No at top-5%. Random had higher spike recall, though with higher selected coverage. This blocks strong deployable selector claims.
- What coverage is needed to recover most spike/pre-spike windows? Hybrid top-10 recovered about 68% spike recall at 58.8% coverage. Hybrid max-30 recovered about 90% spike recall but covered 87.5% of video, which is too dense for the intended sparse path.
- Is ViT-L worth testing next? Only as a narrow follow-up on the same subset if it can be cached cheaply and judged against random/telemetry. The current evidence says selector design needs improvement more than model size escalation.
- What ViT-G/TRIBE teacher budget should be used first? 500 windows maximum, exploratory only, after selector fixes or as a small diagnostic. Do not jump to 1000/2000 windows yet.

## Files
- Output root: `outputs/again_real_scout_selector_validation_20260621_224953_n50`
- External cache root: `/Volumes/onn. Drive/Neural Bridge/benchmarks/again/real_scout_selector_validation_20260621_224618`
- Feature rows: `again_real_scout_feature_rows.csv`
- Selector metrics: `again_real_scout_selector_metrics.csv`
- Per-video selector metrics: `again_real_scout_selector_video_metrics.csv`
- Throughput: `again_real_scout_throughput.csv`
