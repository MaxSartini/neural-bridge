# Neural Bridge

Neural Bridge is a local neuroscience-to-behavior research system. Its current proven core is VEATIC-124 v2: TRIBE-predicted cortical response features improve arousal event/spike ranking under leakage-controlled evaluation.

## Current Status

The VEATIC-124 v2 evidence bundle validates these hypotheses:

1. Real cortical/TRIBE features contain stimulus-specific signal for future arousal spike ranking.
2. PCA cortical feature modes beat autoregressive, shuffled, random, timestamp, and video/time controls on the strongest event/spike rows.
3. Balanced event-vs-stable evaluation exposes signal that full-frame continuous MAE hides.
4. Short causal temporal context improves selected future spike-ranking rows over current-only evaluation.
5. A single 0s feature snapshot can underfeed the bridge head for spike/event tasks.
6. The alignment audit resolved the benchmark policy: keep current 0s alignment as the primary non-leaky benchmark and report offset grids as diagnostics, not as a test-derived correction.

The current claim is intentionally precise: Neural Bridge has evidence for arousal event/spike ranking and temporal-context sufficiency, not exact continuous arousal-value prediction or a finished downstream product model.

## Key VEATIC-124 v2 Results

- Manifest: 124 videos, 10,357 1 Hz target rows.
- Cache: complete cortical/TRIBE cache under the configured external assets root, at `benchmarks/veatic/tribe_cache`.
- Strongest blocked spike row: `cortical_pca64_delta`, `arousal__future_spike_1_3s`, threshold `0.05`.
- PR-AUC for that row: real `0.2536`, AR `0.1969`, shuffled `0.1840`, random `0.1944`.
- Official split spike rows pass controls across current feature families.
- Grouped-video aggregate spike F1 improves over AR for PCA modes: `cortical_pca_64` `+0.0256`, `cortical_pca64_delta` `+0.0177`.
- Balanced event-vs-stable spike row at threshold `0.05`: `cortical_pca64_delta` PR-AUC `0.3394`, `+0.0609` over AR, `+0.0631` over shuffled, `+0.0476` over random.
- Temporal context v2: 4/4 focused feature-target rows improve over current-only by more than `0.005` PR-AUC; best focused windows are `causal_past_2s`.
- Alignment repair: no safe global lag correction was selected; the final non-leaky policy is `keep_current_0s_as_primary_plus_report_offset_diagnostics`.

Source summaries:

- [docs/veatic_v2_evidence_summary.md](docs/veatic_v2_evidence_summary.md)
- [benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md](benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md)
- [benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md](benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md)
- [benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md](benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md)
- [outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md](outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md)
- [outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md](outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md)
- [benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md](benchmarks/veatic/veatic_124_alignment_lag_repair_20260616.md)

## What Remains Unproven

- Exact continuous future arousal-value forecasting. Continuous MAE remains diagnostic because zero-change baselines still win most continuous checks.
- Strong universal early-warning claims. The alignment pass supports 0s primary scoring plus transparent offset diagnostics, not a global lag correction.
- New model heads still need to improve over the v2 baseline without weakening the controls.

## System Shape

```text
backend/app/
  Flask neuro-viewer API plus TRIBE/MLX service adapters.

backend/scripts/
  VEATIC/TRIBE extraction plus the consolidated strict VEATIC benchmark suite.

frontend/
  Vue/Vite cortical cache and stimulus viewer.

docs/, reports/, benchmarks/, outputs/
  Current evidence summaries, benchmark reports, and small tracked result artifacts.
```

Active cortical extraction path:

```text
backend/scripts/run_veatic_tribe_cache.py
  -> app.services.tribe_adapter.TribeAdapter
  -> tribe_mlx runtime
  -> MLX/MPS TRIBE path
  -> per-video tribe_raw_output.npz
```

## External Assets

Heavy assets stay outside git. Configure the external assets root in `.env` for each workstation:

```bash
NEURAL_BRIDGE_EXTERNAL_ROOT=/path/to/neural-bridge-assets
```

This external root contains model weights, Hugging Face caches, raw/processed datasets, TRIBE caches, benchmark caches, and large generated outputs.

See [docs/external_assets_manifest.md](docs/external_assets_manifest.md).

## Local Setup

Backend:

```bash
cd <repo-root>/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Frontend:

```bash
cd <repo-root>/frontend
npm install
npm run dev
```

Repo-root helper:

```bash
cd <repo-root>
npm install
npm run dev
```

The local `.env` file is ignored by git. Full benchmark workflows require the configured external assets root to be mounted or otherwise available.

## Strict Benchmark

Use the unified VEATIC-124 suite as the default benchmark entrypoint:

```bash
python3 backend/scripts/run_veatic_strict_benchmark.py --primary-only
```

It consolidates the formerly separate event/spike, event-conditioned, alignment, and temporal-context checks into one coordinated run. The suite includes AR, shuffled cortical rows, split-local shuffles, Gaussian feature controls, label shuffles, feature shuffles, timestamp-only, video/time-only, majority, fixed-split holdouts, grouped-video holdouts, zero-change diagnostics, and a single-backend policy.

## Guardrails

- Keep train-only thresholding, train-only PCA, grouped-video validation, and blocked validation intact.
- Do not promote positive-only event/pre-event masks as PR-AUC discrimination tests; use recall/top-k or balanced event-vs-stable rows.
- Do not tune thresholds on filtered test subsets.
- Do not treat continuous MAE diagnostics as the headline result.
- Do not add CUDA/Triton/FlashAttention stacks to the Mac/MPS runtime without an isolated adapter plan.
- Do not commit heavyweight caches or model weights.

## Roadmap

The active post-v2 roadmap is in [ROADMAP.md](ROADMAP.md). The short version is:

1. Freeze and package the v2 evidence bundle.
2. Surface the already-strict v2 benchmark rules as a named contract and verifier.
3. Freeze the v2 training tensor contract for future model heads.
4. Build next model heads on the proven cortical/TRIBE signal.
5. Productize the evidence verifier and dashboard workflow.

## License

AGPL-3.0. Check upstream licenses for bundled or referenced research projects before redistribution.
