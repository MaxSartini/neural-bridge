# AGAIN Real Scout Selector Validation

## Executive Summary
- Videos validated: 50
- ViT-B scout windows processed: 1508
- ViT-B cache hits: 50/50
- Mean uncached ViT-B forward time/video: n/a
- Mean total scout time/video: 1.035s
- Hybrid top-5% spike recall: 47.8%
- Same-timestamp-budget random top-5% spike recall: 53.0%
- Coverage-matched random top-5% spike recall: 38.8%
- Oracle top-5% spike recall: 69.3%
- Hybrid top-10% spike recall: 68.2%
- Hybrid max-30 spike recall: 89.9%

## Guardrails
- AGAIN only.
- MLX only.
- No CUDA.
- No dense ViT-G/TRIBE encoding.
- No training.
- Arousal labels are used only for evaluation and oracle upper bound selectors.

## Selector Takeaways
- Telemetry-change top-5% spike recall: 53.4%
- V-JEPA-B novelty top-5% spike recall: 30.4%
- Hybrid top-5% pre-spike 2s/4s/6s/8s recall: 48.3% / 47.1% / 45.9% / 44.2%
- Hybrid top-5% selected video coverage: 39.2%
- Coverage-matched random top-5% selected video coverage: 39.0%
- Hybrid top-10% selected video coverage: 58.8%
- Coverage-matched random top-10% selected video coverage: 58.5%
- Hybrid max-30 selected video coverage: 87.5%
- Coverage-matched random max-30 selected video coverage: 87.1%

## Interpretation
- Hybrid selection beat coverage-matched random on spike recall in this subset.
- The earlier same-timestamp random control covered more video after region expansion, so it is retained as a stress test but not the fair primary random control.
- V-JEPA-B novelty beat its own coverage-matched random control at top-5%.

## Recommended Next Budget
- Start sparse ViT-G/TRIBE with 500 windows only if hybrid beats random at a useful coverage level.
- Use 1000 or 2000 windows only after the 500-window teacher subset shows lift over telemetry and V-JEPA-B controls.

## Files
- Output root: `outputs/again_real_scout_selector_validation_20260621_230938_n50_covmatched`
- External cache root: `${NEURAL_BRIDGE_EXTERNAL_ROOT}/benchmarks/again/real_scout_selector_validation_20260621_224618`
