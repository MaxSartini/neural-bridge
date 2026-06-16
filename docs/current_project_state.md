# Current Project State - 2026-06-16

This is the short operating snapshot for the cleaned Neural Bridge repo after the VEATIC-124 v2 evidence pass.

## Repo

- Active repo: this Git checkout.
- External asset root: configured locally through `.env` as `NEURAL_BRIDGE_EXTERNAL_ROOT`.

The repo should stay lightweight. Heavy research assets belong on the external drive, not in git.

## Active Scientific Direction

Neural Bridge is testing where predicted neural response trajectories improve human-response forecasts under controlled baselines.

Current evidence is strongest for VEATIC-124 cortical/TRIBE arousal event and spike ranking. v2 has validated specific hypotheses around event/spike ranking and causal temporal context. It still should not claim exact continuous arousal-value forecasting.

## Current Benchmark Assets

- Complete VEATIC manifest: `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- Manifest rows: 10,357 at 1 Hz
- Complete cortical cache: `<external-assets-root>/benchmarks/veatic/tribe_cache`
- Cache shape contract: per-video `tribe_raw_output.npz` with required key `predictions`
- Main targets: `valence`, `arousal`

Current feature families:

- `cortical_global`
- `cortical_global_delta`
- `cortical_pca_64`
- `cortical_pca64_delta`
- raw cortical trajectories for future loader work

Current v2 evidence reports now tracked in this repo:

- `benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md`
- `benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md`
- `benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md`
- `outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md`
- `outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md`

Current default benchmark entrypoint:

```bash
python3 backend/scripts/run_veatic_strict_benchmark.py --primary-only
```

Use `--dry-run` to print the strict contract and control ledger without loading the external cache.

## Validated v2 Findings

- Cortical/TRIBE features improve arousal future-spike/event ranking under blocked validation.
- `cortical_pca64_delta` is the strongest blocked full-frame spike row at threshold `0.05`: PR-AUC `0.2536` versus AR `0.1969`, shuffled `0.1840`, and random `0.1944`.
- Official split event/spike rows pass controls across the current feature families.
- Grouped-video validation improves aggregate spike F1 over AR for PCA modes.
- Balanced event-vs-stable sampling confirms event-conditioned discrimination for the strongest spike rows.
- Temporal context v2 shows short causal windows improve selected future arousal spike ranking over current-only evaluation.
- Alignment policy is resolved: current 0s alignment remains the primary non-leaky benchmark, while offset-grid results are diagnostics.

## Benchmark Rules

- Full-frame VEATIC rows remain the main baseline.
- Event-conditioned rows are diagnostics unless balanced against stable controls.
- Positive-only pre-event and event masks should report recall/top-k style diagnostics, not PR-AUC as the main claim.
- Thresholds must be fit on train data only.
- PCA and other transforms must be fit on train data only.
- Controls include AR, shuffled cortical rows, split-local shuffles, Gaussian features, label shuffles, feature shuffles, timestamp-only, video/time-only, majority, fixed-split holdouts, grouped-video holdouts, zero-change diagnostics, and one backend policy per final run.
- CPU/MPS device consistency should be checked before mixing thresholded results.

## Remaining Work

1. Freeze the current 124-video v2 baseline into a protected external snapshot.
2. Run and freeze the consolidated strict suite outputs as the canonical v2 artifact set.
3. Freeze the v2 training tensor contract for future model heads.
4. Carry the resolved alignment policy into the tensor contract and benchmark dashboard.
5. Build next model heads against the frozen v2 baseline.
6. Delete or archive old pre-124 runs after preserving the useful evidence.

## Next Safe Move

Freeze the current 124-video v2 evidence bundle, expose the benchmark contract that the suite already enforces, then freeze the tensor interface for new heads.
