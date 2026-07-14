# Phase 6 Trial 4 Blocked 15-Seed Confirmation — Stage B

Output root: `outputs/again_dense_2hz_phase6_trial4_blocked_15seed_20260714_145637`

- rows: `120/120`
- candidate/original PR-AUC: `0.2673383277` / `0.2672870537`
- mean / median candidate-minus-original: `+0.0000512741` / `+0.0002953952`
- paired wins vs original: `10/15`
- stable 14-seed mean / median / wins: `+0.0003639768` / `+0.0003102598` / `10/14`
- fresh 5-seed mean / median / wins: `-0.0013756950` / `+0.0003251243` / `3/5`
- candidate-minus-AR / best-control: `+0.0074160357` / `+0.0085749406`
- positives vs AR / best control: `15/15` / `15/15`
- seed 20260627 candidate-minus-original: `-0.0043265637`
- Stage B pass: `False`
- failed gates: `['fresh_mean_candidate_beats_original', 'fresh_paired_wins_at_least_4_of_5', 'single_seed_contribution_at_most_0_40']`

Stage C grouped evaluation is authorized only on a Stage B pass.

## Stability Interpretation

- all-15 candidate-minus-original mean: `+0.0000512741`
- all-15 candidate-minus-original median: `+0.0002953952`
- stable-14 mean / median / wins: `+0.0003639768` / `+0.0003102598` / `10/14`
- fresh-five mean / median / wins: `-0.0013756950` / `+0.0003251243` / `3/5`
- candidate/original seed-level PR-AUC standard deviation: `0.0047757770` / `0.0059260615`
- candidate seed-level standard-deviation reduction: `19.41%`

The candidate is a valid controlled bridge model and is modestly more stable,
but it is not a reliably superior replacement. Fresh seed `20260636` produced
another favorable original outcome (`0.2729798`) and a candidate-minus-original
delta of `-0.0090493`; candidate still beat its AR and matched control on that
seed. This supports studying a locked ensemble/checkpoint-stabilization method,
not changing the failed Stage B verdict.
