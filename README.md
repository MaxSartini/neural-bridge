# Neural Bridge

Local neural-response benchmarking, temporal-context, and simulation infrastructure.

Neural Bridge is a research workbench for proving where predicted brain-response features improve human-response forecasting and where they do not. The current project direction is not a generic chatbot benchmark, a finance predictor, or an investor-grade forecasting product. It is a local-first system for building, caching, validating, and eventually injecting neuro-derived features into agent simulations under strict baseline and ablation tests.

The project grew out of the old MiroFish checkout, but this repo is the cleaned Neural Bridge extraction. Heavy models, datasets, caches, and generated benchmark outputs live outside git on the external drive.

## Current Direction

The core research question is:

> Can stimulus-derived neural response predictions add measurable, leakage-controlled signal beyond ordinary behavioral and temporal baselines?

The active v2 pipeline is:

1. Stimuli are processed through TRIBE/V-JEPA style encoders.
2. TRIBE predicts cortical response trajectories.
3. Benchmark scripts turn those trajectories into validated feature sets.
4. Features are tested against human-response datasets such as VEATIC.
5. Validated signals become candidates for simulation conditioning through explicit ablations.

The v2 results now support specific validated hypotheses for VEATIC arousal event/spike ranking: cortical/TRIBE PCA features improve future arousal spike/event ranking under blocked and grouped-video validation, short causal temporal context improves selected spike-ranking rows, and single-frame 0s evaluation can underfeed the bridge head. The evidence does not support exact continuous future arousal-value prediction as the headline claim.

## What Works Now

- Fresh local git repo extracted from the older MiroFish clone.
- Flask backend, Vue frontend, Neo4j-backed graph services, and local OpenAI-compatible LLM wiring.
- Apple Silicon oriented TRIBE path using MLX/MPS where it preserves benchmark contracts.
- Complete VEATIC 124-video cortical cache on the external drive.
- Current VEATIC manifest with 10,357 1 Hz rows.
- Confirmatory 124-video v2 reports for event/spike ranking, event-conditioned retests, temporal fairness, and temporal context.
- Benchmark scripts for cortical global features, delta features, PCA features, temporal fairness checks, event-conditioned diagnostics, and device-consistency audits.
- External asset boundary documented in [docs/external_assets_manifest.md](docs/external_assets_manifest.md).

## What Is Not Claimed

- Neural Bridge has not yet proven full end-to-end simulation accuracy.
- TRIBE features are predictors, not labels.
- Prompt injection from neural features is not considered validated until it beats shuffled, neutral, inverted, and oracle-style controls.
- Current v2 evidence supports arousal event/spike ranking, not exact continuous-value arousal forecasting.
- Financial forecasting, quant-desk language, and old "future predictor" claims are legacy copy and should not guide current work.
- CUDA-only research repos are not part of the main Mac/MPS runtime unless adapted safely.

## Architecture

```text
frontend/
  Vue/Vite interface for graph, simulation, and report workflows.

backend/app/
  Flask API, graph services, simulation services, local LLM clients, storage, and utilities.

backend/neuro_core/
  Neural Bridge core data contracts and neuro feature utilities.

backend/scripts/
  Benchmarking, cache extraction, audit, and simulation scripts.

external_models/
  Lightweight source checkouts and adapters. Heavy weights stay outside git.

docs/ and reports/
  Current project memory, benchmark contracts, readiness notes, handoffs, and audits.
```

The active cortical extraction path is:

```text
backend/scripts/run_veatic_tribe_cache.py
  -> app.services.tribe_adapter.TribeAdapter
  -> backend tribe_mlx path
  -> MLX/MPS TRIBE runtime
  -> per-video tribe_raw_output.npz
```

The main runtime contract is cortical-first. Subcortical extraction exists as a research path, but the main 124-video VEATIC cache is cortical unless a separate frozen subcortical cache is explicitly produced.

## External Assets

Primary external root:

```bash
/Volumes/onn. Drive/Neural Bridge
```

Large assets are intentionally not committed:

- model weights
- Hugging Face caches
- raw and processed datasets
- VEATIC/TRIBE benchmark caches
- generated outputs
- temporary extraction files

Historical commands that still reference the old external name are supported by a compatibility symlink:

```bash
/Volumes/onn. Drive/MiroFish -> /Volumes/onn. Drive/Neural Bridge
```

## Current VEATIC v2 Evidence

The current benchmark work centers on VEATIC-124:

- Complete 124-video manifest: `benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl`
- Complete 124-video cortical cache: `/Volumes/onn. Drive/Neural Bridge/benchmarks/veatic/tribe_cache`
- Required cache key: `predictions`
- Main targets: `valence`, `arousal`
- Important splits: official split, blocked temporal gap, leave-video/grouped holdout
- Leakage controls: train-only transforms, blocked temporal gaps, within-video dynamics, train-prevalence thresholds, causal-window checks

Candidate feature sets include:

- `cortical_global`
- `cortical_global_delta`
- `cortical_pca_64`
- `cortical_pca64_delta`
- raw cortical trajectories for future loader work

Validated v2 findings:

- Blocked full-frame arousal future-spike ranking improves with cortical/TRIBE features. The strongest blocked spike row is `cortical_pca64_delta` at threshold `0.05`: PR-AUC `0.2536` versus AR `0.1969`, shuffled `0.1840`, and random `0.1944`.
- Official split spike ranking passes controls across all four current feature families in the event/spike retest.
- Grouped-video validation improves aggregate spike F1 over AR for PCA modes: `cortical_pca_64` improves F1 by `+0.0256`, and `cortical_pca64_delta` by `+0.0177`.
- Balanced event-vs-stable sampling confirms usable event-conditioned discrimination. For `arousal__future_spike_1_3s@0.05`, `cortical_pca64_delta` reaches PR-AUC `0.3394`, `+0.0609` over AR, `+0.0631` over shuffled, and `+0.0476` over random.
- Temporal context v2 shows short causal context helps selected spike-ranking rows: all 4 focused feature-target rows improved over current-only by more than `0.005` PR-AUC, with best causal windows at `causal_past_2s`.
- Temporal fairness classifies the single-timestep 0s interface as context-starved in the strongest spike rows, meaning 0s-only evaluation can underestimate the representation.

Still unresolved:

- Continuous future-change MAE is diagnostic only. Zero-change baselines still beat real cortical features on most continuous checks.
- Temporal alignment remains a follow-up risk because several spike rows improve at non-zero offsets.
- Video `83` still uses the documented linear-resample policy because its cached prediction length differs from manifest rows.
- Full simulation injection is not validated until the neural prior beats the required ablation suite.

Open issues before new model training:

- Freeze the current 124-video benchmark artifacts into an immutable baseline snapshot.
- Formalize the production training tensor loader contract.
- Decide and document the policy for video `83`, whose prediction length differs from its manifest rows and is currently resampled.
- Keep CPU/MPS consistency checks separate from headline claims when thresholded metrics can drift.

Authoritative current evidence:

- [benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md](benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md)
- [benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md](benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md)
- [benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md](benchmarks/veatic/veatic_124_event_conditioned_retest_20260616.md)
- [outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md](outputs/veatic_124_temporal_context_v2_20260616_1557/veatic_124_temporal_context_v2_report.md)
- [outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md](outputs/veatic_124_temporal_fairness_20260616_1509/veatic_124_temporal_fairness_report.md)
- [docs/current_project_state.md](docs/current_project_state.md)

## Local Setup

Prerequisites:

- macOS with Apple Silicon is the main development target.
- Python 3.11+.
- Node.js 18+.
- Neo4j 5.x for graph workflows.
- LM Studio or another OpenAI-compatible local LLM server for simulation/report workflows.
- External drive mounted at `/Volumes/onn. Drive/Neural Bridge` for full benchmark workflows.

Backend:

```bash
cd "/Users/maxsartini/Neural Bridge/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Frontend:

```bash
cd "/Users/maxsartini/Neural Bridge/frontend"
npm install
npm run dev
```

Repo-root development helper:

```bash
cd "/Users/maxsartini/Neural Bridge"
npm install
npm run dev
```

The local `.env` file is ignored by git. Use `.env.example` and the docs above as the template for local paths and service URLs.

## Benchmark Safety Rules

- Reuse frozen cache outputs unless the benchmark policy changes.
- Do not tune thresholds on filtered test subsets.
- Keep full-frame results as the main baseline, with event-conditioned diagnostics reported separately.
- Treat positive-only event or pre-event masks as recall/top-k diagnostics, not full discrimination tests.
- Compare against shuffled, neutral, inverted, and oracle controls before claiming neuro-specific signal.
- Keep recursive-model experiments out of the main runtime until baseline freezing and loader contracts are complete.

## Roadmap

The active roadmap is maintained in [ROADMAP.md](ROADMAP.md).

Short version:

1. Preserve the v2 evidence bundle as the current baseline.
2. Resolve temporal alignment and video `83` policy without weakening controls.
3. Lock the training tensor contract for v2 feature families.
4. Extend from validated event/spike ranking into model heads, subcortical features, and OpenLAV only after the baseline remains reproducible.
5. Add neural features to simulation only through ablated, measurable experiments.

## License

This repo is AGPL-3.0 licensed. Neural Bridge includes modified local-first simulation work and adapters around external research projects; check each upstream dependency or source checkout for its own license before redistribution.
