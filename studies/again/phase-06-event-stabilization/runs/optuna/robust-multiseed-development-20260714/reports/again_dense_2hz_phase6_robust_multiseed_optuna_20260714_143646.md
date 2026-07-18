# Phase 6 Robust Multi-Seed Optuna — Stage A

Output root: `outputs/again_dense_2hz_phase6_robust_multiseed_optuna_20260714_143646`

This is inner-train/validation-only development. No blocked held-out or grouped
test score was read or produced.

## Study

- trials: `24`
- development seeds: `[20260625, 20260626, 20260627, 20260628, 20260629]`
- reserved validation seeds: `[20260630, 20260631, 20260632, 20260633, 20260634]`
- best trial: `22`
- best development robust objective: `0.0166151902`

## Reserved Inner-Validation Gate

- candidate robust objective: `0.0115854026`
- original robust objective: `0.0120504091`
- gain: `-0.0004650065`
- paired wins: `4/5`
- Stage A pass: `False`
- failed gates: `['robust_objective_gain_at_least_0_001']`

A pass authorizes only the preregistered 15-seed blocked Stage B. A failure
stops the 720-row campaign before held-out scoring.
