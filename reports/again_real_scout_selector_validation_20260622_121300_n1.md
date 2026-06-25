# AGAIN Real Scout Selector Validation

## Executive Summary
- Videos validated: 1
- Scout model: `vjepa21_vitl_dgrauet_mlx_scout`
- V-JEPA-L scout windows processed: 120
- V-JEPA-L cache hits: 0/1
- Mean uncached V-JEPA-L forward time/video: 31.756s
- Mean total scout time/video: 35.664s
- Hybrid top-5% spike recall: 13.6%
- Same-timestamp-budget random top-5% spike recall: 50.0%
- Coverage-matched random top-5% spike recall: 36.4%
- Oracle top-5% spike recall: 86.4%
- Hybrid top-10% spike recall: 59.1%
- Hybrid max-30 spike recall: 90.9%

## Guardrails
- AGAIN only.
- MLX only.
- No CUDA.
- No dense ViT-G/TRIBE encoding.
- No training.
- Arousal labels are used only for evaluation and oracle upper bound selectors.

## Selector Takeaways
- Telemetry-change top-5% spike recall: 63.6%
- V-JEPA-B novelty top-5% spike recall: 40.9%
- Hybrid top-5% pre-spike 2s/4s/6s/8s recall: 13.6% / 18.2% / 22.7% / 22.7%
- Hybrid top-5% selected video coverage: 32.5%
- Coverage-matched random top-5% selected video coverage: 32.5%
- Hybrid top-10% selected video coverage: 63.4%
- Coverage-matched random top-10% selected video coverage: 59.3%
- Hybrid max-30 selected video coverage: 91.1%
- Coverage-matched random max-30 selected video coverage: 90.2%

## Interpretation
- Hybrid selection did not beat coverage-matched random on spike recall in this subset.
- The earlier same-timestamp random control covered more video after region expansion, so it is retained as a stress test but not the fair primary random control.
- V-JEPA-B novelty did not beat its own coverage-matched random control at top-5%.

## Recommended Next Budget
- Start sparse ViT-G/TRIBE with 500 windows only if hybrid beats random at a useful coverage level.
- Use 1000 or 2000 windows only after the 500-window teacher subset shows lift over telemetry and V-JEPA-B controls.

## Files
- Output root: `/Users/maxsartini/neural_bridge_scratch/outputs/again_real_scout_selector_validation_vitl_bfloat16_smoke_n1`
- External cache root: `/Users/maxsartini/neural_bridge_scratch/external_root/benchmarks/again/real_scout_selector_validation_vitl_bfloat16_smoke_n1`
