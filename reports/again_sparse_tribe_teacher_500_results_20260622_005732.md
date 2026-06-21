# AGAIN Sparse TRIBE Teacher 500 Results

## Scope

- This sparse teacher pilot uses `50` selected AGAIN videos out of `995` total videos.
- That is `5.0%` of the dataset.
- AR-only in this report is computed only on the same sparse pilot center rows as the sparse TRIBE/PCA rows.
- It is not the full-AGAIN AR baseline and must not be read as a 995-video comparison.
- AR + sparse PCA128 cannot be tested against all 995 videos until matching sparse PCA rows exist for that full scope.

## Executive Verdict
- Completed sparse ViT-G/TRIBE windows: `480`
- Hybrid AR-only PR-AUC: `40.19%`
- Hybrid AR + sparse PCA128 causal PR-AUC: `24.90%`
- Hybrid AR + telemetry + V-JEPA-B + sparse PCA128 causal PR-AUC: `24.90%`
- Coverage-random AR + sparse PCA128 causal PR-AUC: `52.98%`
- Oracle+background AR + sparse PCA128 causal PR-AUC: `57.45%`

## Gate Summary
- sparse PCA128 vs AR-only: fail (delta -15.29 pp)
- sparse PCA128 vs AR + telemetry + V-JEPA-B: fail (delta -15.29 pp)
- sparse PCA128 vs raw sparse current: fail (delta -13.90 pp)
- hybrid sparse vs coverage-random sparse: fail (delta -28.08 pp)

## Decision Rule
- This is a sparse teacher pilot only, not final AGAIN proof.
- Approve 1000 windows only if the sparse PCA128 causal lane beats AR + telemetry + V-JEPA-B and coverage-matched random.
- Keep 2000 windows premature until a 1000-window follow-up confirms the effect.
