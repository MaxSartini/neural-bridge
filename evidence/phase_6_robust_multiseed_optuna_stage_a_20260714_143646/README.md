# Phase 6 Robust Multi-Seed Optuna Stage A Evidence

This compact snapshot records the fail-closed Stage A result. Full runtime artifacts remain under `outputs/again_dense_2hz_phase6_robust_multiseed_optuna_20260714_143646/`.

## Result

- runtime: `532.38 s` on MLX `Device(gpu, 0)`
- trials: `24`, original configuration enqueued as trial zero
- development seeds: `20260625`–`20260629`
- reserved inner-validation seeds: `20260630`–`20260634`
- blocked held-out scores read: `false`
- grouped scores read: `false`
- best trial: `22`
- reserved paired wins: `4/5`
- candidate/original mean delta vs AR: `0.0138709247` / `0.0138627123`
- mean improvement: `+0.0000082124`
- candidate/original robust objective: `0.0115854026` / `0.0120504091`
- robust-objective gain: `-0.0004650065`
- Stage A pass: `false`
- Stage B/C and 720 rows run: `false`

## Runtime Artifact Checksums

```text
7fe803db7fae652541ecfc0c036ea4c5ad161bbbfaa7106df22edc45230cd5dc  metrics/result.json
1585281b39273294ae5bdcf23f06e5b83f5018c6804982d88ec83f3def804f26  metrics/optuna_trials.csv
aa5611b49247214268eb9bed11169e506d94dfbf88ae75beabeecb033d1eafc8  metrics/trial_seed_inner_validation.csv
e186c47df848e3e66289ef09369b8f9f99ed0c56ecc8e6589e263e500b3d237c  manifests/locked_winner.json
```
