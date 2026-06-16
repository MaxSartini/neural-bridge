# Benchmarking Accuracy

See `docs/tribe_neural_bridge_architecture_review.md` for the target architecture and
the distinction between shared TRIBE response, subgroup response, social
propagation, and final domain outcomes.

Neural Bridge should be benchmarked against real outcomes, not generic chatbot benchmarks.

The core comparison is:

1. baseline simulation without neuro-prior
2. neuro-prior-conditioned simulation
3. simple non-agent baselines
4. observed real-world outcome

## Accuracy Attribution

TRIBE v2's reported fMRI encoding performance and Neural Bridge's downstream
prediction accuracy are different measurements. Strong TRIBE reconstruction
performance means a weak downstream result should not automatically be
attributed to the brain encoder. The system must isolate these layers:

1. **TRIBE representation:** stimulus features to predicted cortical BOLD.
2. **Behaviour mapping:** predicted BOLD to behavioural axes and modifiers.
3. **Injection:** modifiers to agent prompts, probabilities, and actions.
4. **Simulation:** agent interactions to aggregate outcome prediction.

The current highest-risk layer is the unvalidated bridge between steps 1 and 3:
Destrieux parcel percentiles are mapped through fixed heuristic formulas and
then injected mainly through natural-language prompts.

Benchmark runs labeled `neuro_conditioned` must record a real TRIBE backend
provenance (`apple_silicon_tribe`, `official_tribe`, or `tribe_mlx`). Proxy
priors are useful for plumbing tests only and are rejected by the paired
StockNet benchmark by default.

### Exact Subcortical Contract

The released `loganf26/tribev2-subcortical` checkpoint is supported through an
exact Harvard-Oxford mapping:

- output width is exactly 8,808 voxels
- ordering is the positive voxels of `sub-maxprob-thr50-2mm` in NumPy C-order
- the 16 named ROI trajectories sum to all 8,808 source voxels
- ventricles remain in the source representation to preserve ordering but are
  excluded from behavioral interpretation
- the ten measured participant heads are ensembled and their disagreement is
  retained as model uncertainty

The subcortical checkpoint has its own upstream feature contract. Do not feed it
features from the cortical checkpoint merely because both have width 1,024.
Subcortical inference is valid only with provenance matching:

- text: `Qwen/Qwen3-0.6B`
- audio: `facebook/w2v-bert-2.0`
- video: `facebook/vjepa2-vitl-fpc64-256`

The canonical `neuro_response_ir_v2` artifact stores exact subcortical ROI
trajectories and a frozen `neuro_calibration_features_v1` vector. These are
label-free predictors for supervised calibration; they are not emotion labels.

### Translation Fidelity Gate

Before scoring any downstream encoder, record:

- source and target spatial vocabularies
- temporal sampling and any resampling
- available ROI and time coverage
- spatial reconstruction explained variance
- whether the target encoder consumes availability masks
- whether the run is production-eligible or only a research ablation

Do not compare a padded Brain-JEPA embedding against a mask-aware MOMENT
embedding as if both received equivalent evidence. A model is ineligible when
its required representation cannot be translated without fabricating values.
Use `backend/scripts/select_neuro_encoder.py` to apply the current geometry
gate. Accuracy ranking happens only among eligible models.

The first local runtime smoke comparison is recorded at
`benchmarks/results/neuro_encoder_runtime_smoke_20260605.json`. It is not an
accuracy result. On the short TRIBE text smoke input, MOMENT-small used
substantially less time and memory than MOMENT-large. Brain-JEPA Metal inference
worked, but that run is excluded from scientific comparison because the input
was incomplete for Brain-JEPA's unmasked 450-by-160 geometry.

### Required Injection Ablations

Every serious benchmark run should use the same stimuli, model, seeds,
temperature, personas, and chronology across these conditions:

| condition | purpose |
| --- | --- |
| `llm_only` | establishes the simulator baseline |
| `true_neuro_current_mapping` | measures the complete current integration |
| `shuffled_neuro_prior` | detects gains caused merely by adding extra context |
| `neutral_neuro_prior` | detects prompt-format effects |
| `inverted_neuro_prior` | tests whether outputs respond in the expected direction |
| `oracle_behaviour_prior` | estimates the injection layer's achievable upper bound |
| `true_neuro_no_prompt_injection` | tests direct numerical modifiers separately |

Also run:

- **Dose response:** inject the true prior at `0.25`, `0.5`, `1.0`, and `1.5`
  strength. Useful conditioning should usually produce a coherent response,
  rather than arbitrary jumps.
- **Leave-one-axis-out:** remove threat, reward, salience, uncertainty, and
  other axes individually to identify which mappings help or hurt.
- **Permutation test:** shuffle true neuro priors across stimuli. Correctly
  matched priors should outperform shuffled priors.
- **Seed replication:** repeat every condition across multiple deterministic
  seeds. A one-seed gain is not evidence.

Interpretation:

- If `oracle_behaviour_prior` fails, the injection/simulation layer is broken.
- If oracle works but `true_neuro_current_mapping` fails, the BOLD-to-behaviour
  mapping is the likely bottleneck.
- If true neuro beats shuffled neuro, the TRIBE signal is stimulus-specific and
  useful downstream.
- If true and shuffled neuro perform similarly, added prompt context or bias is
  probably driving the apparent gain.
- If direct numerical modifiers outperform prompt injection, move behavioural
  conditioning out of prose and into explicit simulation state transitions.

Use `benchmarks/benchmark_registry.json` to choose a benchmark shape, then create JSONL validation records:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/create_benchmark_template.py \
  --benchmark financial_news_stock_reaction \
  --stimulus-id case_001 \
  --out backend/uploads/validation/case_001.jsonl
```

Then score:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/run_neuro_validation_harness.py \
  backend/uploads/validation/case_001.jsonl
```

Train the calibration-ready CatBoost baseline only after collecting at least 20
paired rows across multiple participant or experiment groups. Each manifest
JSONL row must contain `feature_path`, `group`, and normalized `targets`:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/train_neuro_calibrator.py \
  backend/uploads/validation/calibration_manifest.jsonl \
  models/neuro_calibrators/demo_v1
```

The trainer performs a grouped holdout and writes one model per response axis
plus `calibration_report.json`. Do not promote it into live simulation unless
repeated grouped holdouts beat semantic-only and persona/history-only
baselines.

Run the local paired StockNet component benchmark:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/run_stocknet_paired_benchmark.py \
  --cases 25 --run-id stocknet_holdout_v1
```

This compares identical Qwen forecasts with and without the TRIBE-derived prior,
using a deterministic recent chronological StockNet holdout. It is a
component-level conditioning benchmark, not evidence that the full multi-agent
simulation is accurate. Results and complete audit traces are written under
`benchmarks/results/`.

The runner uses the final 20% chronological holdout by default, sends every
condition as an independent LLM request, and checkpoints after every case.
Reuse `--run-id stocknet_holdout_v1` to resume without repeating completed
cases. Small runs use deterministic evenly spaced sampling across the holdout
to avoid measuring many stocks from one market-wide day. Use `--cases 0` only
when intentionally running the entire holdout.

## Initial Human-Response Dataset Batch

Download or resume the public initial batch directly onto external storage:

```bash
backend/.venv/bin/python backend/scripts/download_initial_benchmark_batch.py all \
  --root "/Volumes/onn. Drive/Neural Bridge/datasets"
```

Components are independently resumable:

- `openlav_videos`: the exact 188 rated OpenLAV clips plus published metadata
  from PsychArchives, downloaded through resumable `.part` files.
- `openlav_tools`: OpenLAV participant ratings and analysis code.
- `openfmri_affective_videos`: OpenNeuro `ds000205`.
- `emofilm_annotations`: OpenNeuro `ds004872` annotations.
- `pvp`: public PVP code. The full `holi-lab/PVP` Hugging Face dataset is
  manually gated and cannot be downloaded until access is granted.

`initial_batch_status.json` records current file counts and sizes. The earlier
108 GB figure was a conservative storage/workspace budget, not the size of the
currently accessible public files.

Refresh status without starting a transfer:

```bash
backend/.venv/bin/python backend/scripts/download_initial_benchmark_batch.py status \
  --root "/Volumes/onn. Drive/Neural Bridge/datasets"
```

For calibrated simulation runs, keep `OASIS_NEURO_PRIOR_IN_ROUND_PROMPTS=false`
so the same prior is not repeatedly amplified in prose. Keep uncalibrated
heuristic modifiers recorded-only. The local defaults also use `OASIS_LLM_TEMPERATURE=0.6`,
`OASIS_LLM_TOP_P=0.9`, and `OASIS_RANDOM_SEED=33`; vary seeds explicitly during
replicated evaluations.

Chronological CSV-backed simulations enable a chronology guard in generated
config. The full extracted historical document is excluded from config
generation and round prompts; each round receives only its proportional CSV
date window. Saved upload hashes are mapped back to original filenames before
being shown to agents. Graph/persona construction from historical projects
still requires a separately frozen training window for a fully leakage-safe
evaluation.

Generated simulation configs include `neuro_integration_contract`. Only fields
listed under `active_runner_modifiers` may be treated as live conditioning in
an evaluation. `recorded_only_modifiers` are audit artifacts until a calibrated
agent-state/action model or supported OASIS hook consumes them.

Hand-authored `NeuroPriorMapper` outputs are now recorded-only by default.
`NEURO_HEURISTIC_MODIFIERS_ACTIVE=true` is an explicit experimental ablation,
not a production setting or a validated scientific claim.

## OpenLAV Calibration

OpenLAV is complete locally at 188/188 labeled videos. Audit it before inference:

```bash
python3 backend/scripts/audit_openlav_dataset.py
```

Cache expensive TRIBE output sequentially and resumably:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/run_openlav_tribe_cache.py
```

This runner requires the exact cortical `facebook/vjepa2-vitg-fpc64-256`
extractor and keeps it separate from the experimental subcortical branch's
ViT-L extractor. After caching, build the grouped official-label manifest:

```bash
python3 backend/scripts/build_openlav_calibration_manifest.py --require-complete
```

Then run the source-family grouped feature ablations:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/run_openlav_benchmark.py \
  benchmarks/openlav/calibration_manifest.jsonl
```

The runner explicitly marks semantic, LLM-only, persona-only, and
neuro-plus-persona conditions unavailable until separately frozen predictors
or validated participant interaction models are supplied. It does not silently
replace those conditions with neuro features.

OpenLAV is sufficient for a first held-out short-video valence/arousal
calibration benchmark. It is not sufficient evidence of text, audio-only,
political, financial, subgroup, or domain-general performance. Participant
ratings must not be treated as independent neuro samples because each stimulus
shares one population-average TRIBE feature vector.

The first OpenLAV model has only 188 independent stimulus rows and hundreds of
IR features. Treat CatBoost as an initial benchmark, not the default winner.
Require it to beat the train-fold mean baseline and simpler regularized models
under repeated source-family grouped holdouts before promotion.

## Priority Benchmarks

### Financial News And Market Reaction

Best near-term fit for investor demos.

Candidate datasets:

- FNSPID: financial news plus stock-price time series.
- StockNet ACL18: tweets, news/text, and historical prices.
- BigData22 / CIKM18-style financial movement datasets.

Targets:

- next-day or multi-day return direction
- abnormal return magnitude
- sentiment trajectory
- volatility or attention proxy

Required baselines:

- price momentum
- naive sentiment
- LLM-only Neural Bridge
- neuro-conditioned Neural Bridge

### Event-Centric Social Sentiment

Best fit for crisis comms, policy, advertising, and public discourse.

Candidate datasets:

- SURGE event-centric social-media sentiment time-series benchmark.
- GDELT tone/event time series.
- EventRegistry exports where licensed.

Targets:

- sentiment curve
- attention/volume curve
- polarisation or reply-density proxy
- direction of sentiment shift after event release

Required baseline:

- persistence model. Sentiment time series often have strong local persistence, so beating this matters.

### Geopolitical / Regulatory Event Panels

Best fit for oil disruption, court ruling, geopolitical risk, and regulatory-shift demos.

Candidate datasets:

- GDELT Events.
- ICEWS-style event forecasting datasets.
- ICBe crisis events.

Targets:

- escalation/de-escalation direction
- event-count change
- attention/risk proxy
- actor-region reaction type

## Normalized Axes

The current validation harness expects normalized `0..1` axes:

- `sentiment`
- `virality`
- `polarisation`
- `trust`
- `risk_aversion`
- `attention`

Raw outcomes must be converted into these axes using explicit, saved transforms. Example:

- `sentiment`: min-max or z-score transformed sentiment model output
- `virality`: normalized post/news volume
- `risk_aversion`: volatility, drawdown, or defensive-language proxy
- `attention`: article count, comment count, search interest, or volume

## Scientific Standard

Do not claim accuracy from a single demo. Claim improvement only when neuro-conditioned runs outperform baselines on chronologically held-out cases.

Do not describe downstream failure as a TRIBE failure unless TRIBE itself is
evaluated against held-out fMRI targets. For product benchmarks, report the
result as end-to-end, mapping-layer, injection-layer, or simulation-layer
performance according to the ablations above.

Minimum credible table:

| condition | MAE | RMSE | directional accuracy | calibration error |
| --- | ---: | ---: | ---: | ---: |
| persistence/momentum | | | | |
| naive sentiment | | | | |
| LLM-only Neural Bridge | | | | |
| neuro-conditioned Neural Bridge | | | | |

For probability-like targets also report Brier score, log loss, expected
calibration error, paired candidate win rate, and a bootstrap confidence
interval over matched stimuli. Use grouped holdouts so participant or
experiment leakage cannot create artificial gains.

### Human-Choice Policy Benchmark

Use Psych-101 to compare Qwen against Minitaur before assigning either model the
agent-decision role. The local dataset contains 60,092 participant transcripts
and 10,681,650 recorded choices across 160 experiments.

Required splits and metrics:

- Hold out entire participants to measure within-task human-choice
  generalization.
- Hold out entire experiments to measure cross-task generalization.
- Score only the probability assigned to the recorded human choice at each
  trial; do not score prose quality.
- Report choice negative log-likelihood, top-1 accuracy, Brier score,
  calibration error, latency, and peak memory.
- Compare Qwen, Minitaur, simple empirical-frequency baselines, and a shuffled
  history control.

The current LM Studio `/v1/completions` endpoint returned `logprobs: null` for
the loaded Qwen model. Use constrained choice output there for top-1 accuracy,
but do not claim Qwen choice NLL or calibration from that endpoint. Minitaur can
be scored from direct local logits. A probability-level comparison requires a
Qwen runtime/API that exposes candidate token log-probabilities.

Psych-101 can validate which LLM better predicts generic human choices. It
cannot establish that TRIBE improves stimulus-response prediction because it
does not contain matched TRIBE-compatible stimuli and response trajectories.
