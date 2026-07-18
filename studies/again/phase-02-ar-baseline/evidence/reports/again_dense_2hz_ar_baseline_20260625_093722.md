# AGAIN Dense 2Hz AR Baseline

## Scope

- This benchmark uses the dense H100 AGAIN cache with true 0.5s row-level targets.
- Saved `time_seconds` values are used directly; no 1Hz fallback was used.
- No V-JEPA/TRIBE re-encoding was performed.
- Binary event thresholds are selected inside each train split from continuous future-label movement.
- This is not PCA bridge training.

## Coverage

- rows: `243575`
- labeled rows: `243441`
- targets: `['arousal_spike_rows_2_6_train_q90', 'arousal_delta_p2rows_train_q90', 'arousal_abs_delta_p4rows_train_q90']`
- validation protocols: `['blocked_temporal_70_30', 'grouped_video']`
- ridge alpha selection: `train_only_inner_validation`
- ridge alpha grid: `[0.1, 1.0, 10.0, 100.0]`
- ridge backend: per-fold `mlx_primal_conjugate_gradient`

## Lane Summary

- `arousal_abs_delta_p4rows_train_q90` / `blocked_temporal_70_30` / `AR_only`: folds `1`, PR-AUC `11.60%`, ROC-AUC `45.90%`, F1 `21.13%`
- `arousal_abs_delta_p4rows_train_q90` / `grouped_video` / `AR_only`: folds `5`, PR-AUC `11.82%`, ROC-AUC `53.98%`, F1 `19.55%`
- `arousal_delta_p2rows_train_q90` / `blocked_temporal_70_30` / `AR_only`: folds `1`, PR-AUC `26.19%`, ROC-AUC `70.70%`, F1 `34.64%`
- `arousal_delta_p2rows_train_q90` / `grouped_video` / `AR_only`: folds `5`, PR-AUC `20.84%`, ROC-AUC `65.66%`, F1 `30.10%`
- `arousal_spike_rows_2_6_train_q90` / `blocked_temporal_70_30` / `AR_only`: folds `1`, PR-AUC `20.36%`, ROC-AUC `67.04%`, F1 `19.60%`
- `arousal_spike_rows_2_6_train_q90` / `grouped_video` / `AR_only`: folds `5`, PR-AUC `14.73%`, ROC-AUC `58.44%`, F1 `20.56%`

## Promotion Gates

- `arousal_abs_delta_p4rows_train_q90`: raw beats AR `False`, AR+raw beats AR `False`, strict raw pass `False`, strict AR+raw pass `False`
- `arousal_delta_p2rows_train_q90`: raw beats AR `False`, AR+raw beats AR `False`, strict raw pass `False`, strict AR+raw pass `False`
- `arousal_spike_rows_2_6_train_q90`: raw beats AR `False`, AR+raw beats AR `False`, strict raw pass `False`, strict AR+raw pass `False`

## Limitations

- Raw cortical lanes use fixed label-free cortical block mean/std summaries for computational safety when modelling the 20,484 vertex output.
- Temporal diagnostics are non-PCA causal cache diagnostics, not learned bridge features.
- Promotion requires grouped-video wins over AR and nuisance controls; raw cortical losing is a valid outcome.

## Guardrails

- vjepa_encoding_run=`False`
- tribe_encoding_run=`False`
- pca_run=`False`
- models_trained=`True`
