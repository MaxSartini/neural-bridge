# Local-First Prediction Pipeline

Neural Bridge is a two-stage offline prediction pipeline for testing population-level responses to unreleased stimuli.

The product path is not a Twitter or Reddit simulator. Those names are legacy OASIS adapter details. The local demo path is a single Neural Bridge simulation channel that runs entirely on local hardware.

## Stage 1: Neural Prior

TRIBE v2 runs first. It predicts population-average BOLD fMRI response patterns for a stimulus across text, audio, or video.

The output is treated as a conservative behavioural prior, not as emotion detection, thought reading, or individual diagnosis. The released TRIBE checkpoint represents an average subject. It is useful as a salience/threat/reward/uncertainty calibration signal, but it does not directly predict behaviour.

Key caveats:

- BOLD is a hemodynamic proxy with temporal lag and coarse spatial resolution.
- TRIBE is correlative, not causal.
- The released weights are population-average, not demographic or individual-specific.
- Generalisation to investor relations, political speech, advertising, and crisis communications must be validated empirically.
- Subcortical predictions are especially noisy and should be treated as weak priors unless validated on the target domain.

## Stage 2: Social Simulation

Neural Bridge/Qwen then runs the agent-based simulation. Agents are synthetic personas built from the user's data, knowledge graph, commentary, and scenario requirements.

The simulation should:

- Walk agents through the evidence chronologically.
- Reveal only what was knowable at each simulated time step.
- Preserve distinct agent perspectives and priors.
- Encourage disagreement, branching, and new hypotheses rather than convergence.
- Produce a directional prediction with a reasoning trail, not a black-box score.

Agents are LLM-driven simulations, not humans. The credible claim is population-level directional prediction that can improve with calibration against real outcomes.

## Data Flow

1. User supplies stimulus and historical/contextual evidence.
2. TRIBE computes neural-response features and stops.
3. Neural Bridge maps those features into behavioural modifiers.
4. Qwen-powered persona/config generation incorporates the modifiers.
5. The simulator runs agents serially through the evidence timeline.
6. Outputs are action logs, reasoning trails, reports, and calibration artifacts.

## Neuro-Behaviour Adapter

The current adapter is `destrieux_surface_percentile_adapter_v1`.

It takes TRIBE cortical predictions with 20,484 fsaverage5 vertices, splits them into left/right 10,242-vertex hemispheres, applies Nilearn's Destrieux surface atlas, and computes parcel-level activation summaries.

Those parcel summaries are grouped into transparent behavioural proxy axes:

- salience and attention
- threat and avoidance
- reward and approach
- memory and context
- uncertainty and control
- social-semantic interpretation

These axes are then mapped into Neural Bridge modifiers. This is an auditable heuristic adapter, not a validated neuroscience model. It is designed so advisors can critique the mapping directly.

Optional downloaded artifacts:

- `models/neuro_atlases/` for Nilearn surface atlases
- `models/tribe/loganf26-tribev2-subcortical/` for the optional TRIBE subcortical head
- `models/tribe-mlx/zimengxiong-tribev2-mlx/` for third-party MLX TRIBE artifacts

## Local Execution Rule

On local 27B inference, only one LLM call should execute at a time.

Defaults:

- `OASIS_ENV_SEMAPHORE=1`
- `OASIS_SERIAL_PLATFORMS=true`
- `OASIS_MAX_ACTIVE_AGENTS_PER_ROUND=8`
- Product-level platform: `neural_bridge`

Legacy internal adapters may still use OASIS names like `reddit` or `twitter`, but user-facing language should avoid presenting the system as a social-media clone.

## Validation Thesis

The moat is not TRIBE alone and not LLM agents alone. The moat is a paired dataset:

- stimulus features
- predicted neural priors
- simulated social trajectories
- real-world outcomes

Validation should measure whether the neuro-prior-conditioned simulation outperforms:

- LLM-only simulation
- naive sentiment analysis
- historical baseline models
- domain expert forecasts

The scientific advisor question is whether the mapping from predicted BOLD patterns to behavioural modifiers is credible enough to survive peer review and institutional diligence.

Use `backend/scripts/run_neuro_validation_harness.py` to score local calibration records.

See `docs/benchmarking_accuracy.md` for benchmark dataset options and a repeatable backtesting workflow.

JSONL record shape:

```json
{"stimulus_id":"case_001","condition":"baseline","predicted":{"sentiment":0.42,"virality":0.31},"observed":{"sentiment":0.66,"virality":0.58}}
{"stimulus_id":"case_001","condition":"neuro","predicted":{"sentiment":0.61,"virality":0.54},"observed":{"sentiment":0.66,"virality":0.58}}
```

The harness reports MAE, RMSE, bias, directional accuracy, and improvement over `baseline`.
