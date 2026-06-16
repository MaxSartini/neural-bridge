# TRIBE v2 + Neural Bridge Architecture Review

## Verdict

The combination is technically coherent as a research hypothesis, but the
current implementation overextends what TRIBE supplies.

TRIBE v2 predicts an average-subject cortical BOLD-response trajectory for a
naturalistic text, audio, or video stimulus. Neural Bridge/OASIS simulates agents
whose actions are driven by profiles, memories, recommended content, platform
state, and an LLM. The useful bridge is therefore:

`stimulus -> shared neuro-response features -> calibrated subgroup response -> agent state -> social propagation`

The current bridge is closer to:

`stimulus -> shared neuro-response features -> heuristic feelings -> repeated prompt text`

That loses temporal information, gives every persona substantially the same
prior, and asks the LLM to invent the missing subgroup-specific relationship.

See `docs/neuro_behavior_bridge_model_search.md` for released model and dataset
candidates, including Centaur/Minitaur and Psych-101.

## What Is Correct

- TRIBE runs on the original stimulus before the social simulation.
- Text, audio, and video are treated as stimulus modalities.
- The raw cortical output and calibration trace are retained for audit.
- TRIBE is used as a prior rather than described as direct mind reading.
- Neural Bridge/OASIS remains responsible for personas, memory, network exposure,
  recommendation, time progression, and emergent social interactions.
- Local inference is serial and reproducible.

## What Is Incorrect Or Incomplete

### TRIBE Is Population-Average

The released TRIBE checkpoint predicts the average subject on the fsaverage5
cortical mesh. It does not directly predict how a Democrat, Republican,
investor, customer, or individual voter will respond differently.

Subgroup differences require a separately validated conditional model:

`group response = shared TRIBE response + subgroup baseline + subgroup-by-stimulus interaction`

The interaction must be estimated from real subgroup outcomes. Asking Qwen to
infer it from a persona is useful as a provisional hypothesis, not scientific
ground truth.

### The BOLD-To-Behaviour Decoder Is Heuristic

`NeuroRoiCalibrator` currently maps surface-parcel percentiles into threat,
reward, salience, uncertainty, and related axes using hand-authored rules.
`NeuroPriorMapper` then maps those axes into more hand-authored simulation
modifiers. These are the highest-risk scientific components.

### Prompt Injection Is Too Indirect

The same population-level prior is inserted into persona generation,
configuration generation, and per-round prompts. This can:

- homogenize agents
- cause anchoring and repetition
- amplify the prior multiple times
- make results depend on Qwen prompt interpretation rather than simulation
  mechanics

The prior should initialize explicit numerical agent state once, then decay and
update through evidence and interaction.

### Temporal TRIBE Output Is Collapsed

TRIBE produces a response trajectory across time and cortical vertices. The
current integration collapses most of that trajectory into one stimulus-level
profile. For campaign videos, speeches, and earnings calls, the sequence of
peaks, reversals, and sustained responses may be more valuable than a mean.

### Neural Bridge Is A Social Simulator, Not A Price Model

OASIS is designed around profiles, posts, recommendation, action selection,
memory, networks, and time. It can estimate human reaction and propagation
features. It should not be solely responsible for predicting a stock price.

For earnings reports:

`price prediction = market/time-series baseline + event fundamentals + simulated human-reaction features`

For campaign ads:

`vote/support prediction = prior support + exposure + subgroup response + social propagation`

## Target Architecture

### 1. Preserve Stimulus Fidelity

- Feed the original campaign video, speech audio, report, or announcement into
  TRIBE.
- Do not substitute downstream social commentary for the original stimulus.
- Process commentary separately as evidence about historical participant
  reactions.

### 2. Retain Neuro-Response Trajectories

Produce auditable features including:

- parcel and network trajectories
- peak magnitude and timing
- sustained response
- change points
- cross-network contrasts
- uncertainty and out-of-distribution indicators

Do not label these features as emotions until a validated decoder supports that
label.

Use the canonical `neuro_response_ir_v2` artifact as the translation boundary.
It treats TRIBE's output as a source language: time-indexed signed values on
20,484 fsaverage5 cortical vertices. Each downstream adapter must declare its
target spatial/temporal vocabulary, preserve provenance and sampling metadata,
and report translation reconstruction loss. This permits loss-aware adapters
for ROI models such as Brain-JEPA/Brain-DiT and temporal models such as MOMENT
without silently converting cortical values into unsupported emotion labels.
Version 2 additionally carries exact Harvard-Oxford subcortical ROI
trajectories when the released subcortical head was run with its own compatible
Qwen3-0.6B, Wav2Vec-BERT 2.0, and V-JEPA2 ViT-L feature extractors. Feature
provenance is a hard validity gate; matching tensor dimensions are not enough.
For normalized temporal encoders, retain explicit source-channel magnitude
features alongside the normalized embedding. Normalization may improve model
compatibility, but it must not erase response amplitude from the downstream
decoder's available evidence.

The translation contract is:

1. Preserve the raw TRIBE trajectory as the authoritative source artifact.
2. Preserve sign, time order, sampling rate, and spatial registration.
3. Declare every aggregation, normalization, resampling, projection, and mask.
4. Measure spatial reconstruction loss and available-channel/time coverage.
5. Never fill unavailable channels or frames and then present them as observed.
6. Never translate activation directly into an emotion or action label without
   a separately validated decoder.
7. Reject a target model when it cannot consume the availability masks required
   by the translation.

This is closer to translating a structured scientific measurement than
translating prose: semantic equivalence is an empirical property measured by
held-out downstream utility and reconstruction diagnostics, not an assumption.

### 3. Train A Behaviour Decoder

Fit a small, interpretable model from TRIBE features to observed human outcomes:

- favorability rating
- recall
- sharing intent
- trust
- vote/support shift
- risk preference
- click, purchase, or engagement choice

Compare against stimulus-only and LLM-only baselines. TRIBE is useful only if it
adds held-out predictive value.

### 4. Learn Subgroup Interactions

For each subgroup or persona type, estimate:

- baseline position
- susceptibility to each decoded response dimension
- confidence/uncertainty
- historical reaction patterns
- exposure and activity behavior

Never assume that a shared TRIBE signal implies the same behavioral change for
all groups. For example, high salience can increase support in one group and
opposition in another.

### 5. Initialize Agent State Numerically

Add explicit per-agent state such as:

- attention
- arousal proxy
- valence/favorability
- trust
- threat sensitivity
- uncertainty
- action propensity

Initialize it once using the calibrated subgroup interaction. Update it through
new evidence and social interactions. Use it to alter action probabilities,
not merely prompt prose.

Qwen should not invent this state. Pretrained and fitted components should
compute the state before the LLM call. Qwen's role should be constrained to:

- reason over supplied evidence and the computed persona state
- select among allowed actions using supplied probabilities or bounds
- render an action in the persona's established voice
- explain uncertainty without fabricating missing facts

Persona identity, stimulus response, and action propensity must remain separate
objects. This prevents the shared population neuro-prior from contaminating
identity generation and prevents repeated prompt injection from amplifying the
same signal every round.

The local default disables neuro-prior prose in persona and configuration
generation. Hand-authored numerical modifiers are also recorded-only by
default because they have not been calibrated against behavior. Their active
use is available only as the explicit research ablation
`NEURO_HEURISTIC_MODIFIERS_ACTIVE=true`. Prompt injection remains available
only as a separate explicit research ablation through
`NEURO_PRIOR_IN_PERSONA_PROMPTS=true` or
`NEURO_PRIOR_IN_CONFIG_PROMPTS=true`.

The shared population prior also cannot overwrite an agent's stance by default.
That behavior is available only as the explicit
`NEURO_PRIOR_CAN_OVERRIDE_STANCE=true` ablation. A shared neural response can
change attention or action propensity, but subgroup-specific directional
stance changes require observed subgroup calibration data.

Round-level neuro-prior prose is also disabled by default through
`OASIS_NEURO_PRIOR_IN_ROUND_PROMPTS=false`. Repeating the shared prior every
round creates anchoring and convergence rather than new evidence. Both prompt
injection and uncalibrated numerical modifiers are retained only as explicit
ablations.

The shared population prior does not shift every agent's sentiment by default
(`NEURO_PRIOR_SHARED_SENTIMENT_SHIFT=false`). Shared salience can alter
attention and propagation, but directional favorability must come from persona
priors plus an empirically fitted subgroup interaction. Uniform sentiment
shifts are retained only as an explicit ablation.

No hand-authored modifier is claimed as an active OASIS control by default.
Operational modifiers can be consumed by the runner's bounded activation
probability only during the explicit heuristic ablation. Production activation
requires a held-out calibrated numerical agent-state/action model.

### Sequential Local Inference

Run heavy local models as isolated sequential stages:

```text
TRIBE -> temporal encoder -> persona interaction/choice model -> Qwen/OASIS
```

Each stage must write a versioned artifact and exit before the next stage
starts. Process termination is the memory-release boundary; do not depend on
Python garbage collection to release Metal, MLX, or PyTorch allocations. This
also permits separate patched runtimes for models with incompatible upstream
dependency pins.

Audio and video simulations must provide `stimulus_media_path` when creating or
preparing the simulation. The path is passed through `NeuroPriorService` to the
real TRIBE adapter. Non-text simulations no longer substitute the simulation
requirement as fake media input.

Use `backend/scripts/run_sequential_model_pipeline.py` for JSON-defined staged
execution. The runner validates required outputs and records stage logs/status
before permitting the next model to load.

For the standard TRIBE-output handoff, use
`backend/scripts/run_neuro_bridge_pipeline.py`. It builds the canonical IR and
translation report, applies the compatibility router, then launches exactly
one selected heavy encoder in an isolated subprocess.

Treat model compatibility as part of translation fidelity. MOMENT consumes
native-time trajectories with an availability mask and is therefore the valid
default for short stimuli. The released Brain-JEPA encoder consumes exactly
450 ROIs by 160 frames and does not consume availability masks during ordinary
inference. Use it only when all Schaefer-400 cortical channels, Tian-50
subcortical channels, and complete 160-frame windows are available. Any run on
zero-padded or missing-channel input is an explicitly labeled research
ablation, not valid neuro-conditioning. Use
`backend/scripts/select_neuro_encoder.py` to apply this geometry gate before
accuracy-based model selection.

### 6. Use OASIS For Propagation

Use OASIS components for what they model:

- historically grounded profiles
- relationships and network structure
- recommendation/exposure
- hourly activity patterns
- memory
- actions such as posting, liking, commenting, and sharing

The social simulation should predict reaction distributions and trajectories,
not manufacture the upstream stimulus response.

### 7. Use A Separate Outcome Model

Translate simulation aggregates into domain outcomes with a calibrated model:

- polls or vote likelihood for campaigns
- abnormal return and volatility for earnings
- purchase/conversion for advertising
- sentiment and attention curves for communications

This prevents the LLM from being asked to directly guess a stock price from
brain-response prose.

## Required Validation Ladder

1. TRIBE feature extraction parity and stability.
2. TRIBE features versus measured human response labels.
3. Behaviour decoder versus text/video-only and LLM-only baselines.
4. Subgroup interaction predictions versus held-out subgroup outcomes.
5. Agent action distributions versus observed participant actions.
6. Social propagation trajectories versus real propagation.
7. Domain outcome model versus naive and professional baselines.
8. Full end-to-end ablations: true, shuffled, neutral, inverted, and oracle
   neuro priors.

## Commercially Defensible Claim

The defensible near-term claim is:

> The system tests whether predicted population-level neural-response features
> improve calibrated simulations of subgroup reaction and social propagation
> beyond semantic LLM-only baselines.

Do not claim that TRIBE directly measures emotion, predicts an individual's
vote, or determines a stock price.
