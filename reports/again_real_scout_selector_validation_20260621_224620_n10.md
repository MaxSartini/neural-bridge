# AGAIN Real Scout Selector Validation

## Executive Summary
- Videos validated: 10
- ViT-B scout windows processed: 301
- Mean ViT-B forward time/video: 8.217s
- Mean total scout time/video: 12.287s
- Hybrid top-5% spike recall: 52.8%
- Random top-5% spike recall: 53.4%
- Oracle top-5% spike recall: 14.8%

## Guardrails
- AGAIN only.
- MLX only.
- No CUDA.
- No dense ViT-G/TRIBE encoding.
- No training.
- Arousal labels are used only for evaluation and oracle upper bound selectors.

## Selector Takeaways
- Telemetry-change top-5% spike recall: 57.4%
- V-JEPA-B novelty top-5% spike recall: 27.3%
- Hybrid top-5% pre-spike 2s/4s/6s/8s recall: 50.6% / 44.7% / 40.0% / 36.1%
- Hybrid top-5% selected video coverage: 37.5%

## Interpretation
- Hybrid selection did not beat same-budget random on spike recall in this subset.
- V-JEPA-B novelty did not add spike recall beyond telemetry-change at top-5%.

## Recommended Next Budget
- Start sparse ViT-G/TRIBE with 500 windows only if hybrid beats random at a useful coverage level.
- Use 1000 or 2000 windows only after the 500-window teacher subset shows lift over telemetry and V-JEPA-B controls.

## Files
- Output root: `outputs/again_real_scout_selector_validation_20260621_224618_n10`
- External cache root: `/Volumes/onn. Drive/Neural Bridge/benchmarks/again/real_scout_selector_validation_20260621_224618`
