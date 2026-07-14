# Phase 6 Optuna Selected-Head Pilot Evidence

This compact snapshot records the completed one-seed calibration pilot around the already-proven AGAIN target/head. The full runtime artifacts remain in the ignored output root `outputs/again_dense_2hz_phase6_optuna_selected_head_pilot_20260714_135902/`.

## Result

- seed: `20260625`
- trials: `16`, seeded TPE; exact original enqueued as trial 0
- accelerator: MLX `Device(gpu, 0)`
- study objective: inner-validation PR-AUC delta versus seed-specific frozen AR
- held-out access during study: `false`
- original canonical/fresh PR-AUC: `0.2697372518888213` / `0.2697372518888213`
- tuned PR-AUC: `0.2718557351911264`
- frozen AR / best matched control PR-AUC: `0.26369109067321667` / `0.26369109067321667`
- tuned minus original: `+0.002118483302305074`
- tuned minus AR/best control: `+0.008164644517909714`
- result: promising bounded follow-up, not a promoted claim

The winner was locked before held-out scoring. The two best trials converged on identical categorical parameters and nearly identical continuous parameters. See `reports/again_dense_2hz_phase6_optuna_selected_head_pilot_20260714_135902.md` for the human-readable report.

## Runtime Artifact Checksums

```text
0429bc7e6bf6a8be55597287e56643c2a279180cfdae33d545cdea561ef939d9  metrics/pilot_result.json
c25c759f82be4410632060e97188d2849f75a996a52db26da00953154470ada5  metrics/pilot_heldout_metrics.csv
33e7d52faaf87259d14925a3ea6f7c5b1238f4174f6eb6fc39e8e549562afb97  metrics/optuna_trials.csv
cf1f783105f5fc80df0639cbfd66d2487e2ba11f679e325cf6f9684099f98d0a  manifests/locked_optuna_winner.json
cb52cb3ee7b47b1f8741352c8680531507a2f422ef5e05af25ee4003495e1ac5  manifests/run_manifest.json
```
