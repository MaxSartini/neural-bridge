# Phase 6 Optuna Locked-Winner 10-Seed Confirmation Evidence

This compact snapshot records the preregistered locked-winner confirmation. Full runtime artifacts remain under `outputs/again_dense_2hz_phase6_optuna_locked_10seed_confirm_20260714_141457/`.

## Verdict

- MLX runtime: `515.80 s`, `Device(gpu, 0)`
- tuned / canonical original PR-AUC: `0.2659654274` / `0.2670735630`
- all-seed tuned-minus-original: `-0.0011081356`
- nine-follow-up-seed tuned-minus-original: `-0.0014666488`
- positive vs original: `7/10`; positive on follow-up seeds: `6/9`
- tuned-minus-frozen-AR / best-control: `+0.0057318043` / `+0.0067636509`
- positive vs AR / best control: `8/10` / `8/10`
- locked improvement pass: `false`
- failed gates: `followup_mean_delta_at_least_0_001`, `full_mean_delta_positive`

Seed `20260627` contributed `-0.0178629568` tuned-minus-original. Its canonical original training curve had an unusually favorable peak. A separate post-hoc 80-epoch convergence diagnostic reproduced the tuned result and did not change the verdict.

## Runtime Artifact Checksums

```text
66fee1160cfe18d7411b5c245bc49fcf39affc5b14a5e23f2be6ed5162ae704c  metrics/locked_10seed_result.json
0495fb3a921f2071f08db1c2caed8f5d3e6d42e4dea88a72cafa142f8e3c58f1  metrics/locked_10seed_seed_deltas.csv
5f3e240c20361be246d3bbd689a46fe3930326951d26258eb1b2829ad89c360d  metrics/locked_10seed_summary.csv
1ba35cdbc2cf82908a3c1c0650015da9e00b2c68d0648bb197ef1dd7f519a781  manifests/run_manifest.json
ff6f3590018427f2a241c7238eeb603be6cc3e8e56d1664308f4373f368282fa  manifests/locked_winner_provenance.json
81b5be1bb6dbfc9c5e039f20abba5b77fb80c367704f0c4121163e1e69a3ab1a  posthoc seed-20260627 convergence result.json
eb1051e4ab8666abc82041dc720fb0555f491e1fba702576b73936102cdc96c2  posthoc seed-20260627 extended_training_curve.csv
```
