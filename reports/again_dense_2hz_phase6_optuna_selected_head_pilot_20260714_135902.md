# Phase 6 Optuna Selected-Head Pilot

This is an exploratory one-seed calibration pilot around the already-proven
AGAIN selected head. Optuna saw only the inner train/validation partition; the
blocked held-out test was scored once after the winner was locked. The
canonical 420-row result is unchanged.

## Scope

- seed: `20260625`
- trials: `16`
- target: `future_arousal_max_delta_rows_4_10_train_q90`
- head: `short_temporal_conv_residual`
- feature: `temporal_mean_2s_then_pca256`
- MLX device: `Device(gpu, 0)`
- V-JEPA/TRIBE/PCA rerun: `false`

## Result

- canonical stored original PR-AUC: `0.2697372519`
- fresh original reproduction PR-AUC: `0.2697372519`
- Optuna-tuned PR-AUC: `0.2718557352`
- frozen AR PR-AUC: `0.2636910907`
- best tuned matched control: `label_permutation_residual` / `0.2636910907`
- tuned minus fresh original: `+0.0021184833`
- tuned minus frozen AR: `+0.0081646445`
- tuned minus best control: `+0.0081646445`
- original minus frozen AR: `+0.0060461612`
- tuned bridge-specific lift vs original bridge-specific lift: `+35.04%`
- tuned relative PR-AUC gain vs original: `+0.7854%`
- reproduction absolute difference: `0.0000000000`
- promising enough to justify a multi-seed Optuna follow-up: `True`

## Locked Winner

```json
{
  "alpha_cap": 0.16,
  "alpha_initial_logit": -3.0,
  "gate_bias": 5.0,
  "hidden": 64,
  "lambda_binary": 0.8,
  "learning_rate": 0.00010528366155183298,
  "weight_decay": 0.00020452569809101856
}
```

The two best trials converged on the same categorical settings and nearly the
same learning rate/weight decay. Trial 14 reached inner-validation delta
`0.0173822388`; trial 15 reached `0.0173836433`. The enqueued original trial
reached `0.0148640280`.

This single seed cannot promote a new claim. It only measures whether Optuna
adds enough value to justify a bounded multi-seed follow-up.
