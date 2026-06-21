# AGAIN Full AR Context Baseline

## Scope

- This is a full-AGAIN annotation-history AR context baseline.
- It uses the boundary-audited 1Hz manifest only.
- It does not run TRIBE, V-JEPA, or sparse PCA128.
- It must not be treated as a direct row-matched comparison against the 50-video sparse teacher pilot.

## Results

- `future_spike_1_3s_ge_0.05`: videos `995`, rows `118990`, events `34068`, PR-AUC `35.48%`, ROC-AUC `59.07%`
- `future_spike_1_3s_ge_0.075`: videos `995`, rows `118990`, events `25954`, PR-AUC `28.11%`, ROC-AUC `58.79%`

## Fix Applied

The sparse 500-window pilot remains a 50-video sparse-row pilot. This full AR context file is the correct
full-dataset denominator for AR-only, but it is not a replacement for a row-matched AR + sparse PCA128 test.
To test AR + PCA128 on all AGAIN videos, sparse PCA128 features must first be generated for a full-scope queue
covering the 995 videos under the same row contract.

## Guardrails

- tribe_encoding_run=`False`
- models_trained=`True`
- veatic_outputs_modified=`False`
- direct_sparse_pca128_comparison_made=`False`
