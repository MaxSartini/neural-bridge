# Neuro-Behavior Bridge Model Search

## Finding

No released model was found that directly performs:

`TRIBE v2 cortical response + persona -> validated individual action probabilities`

Existing models cover pieces of the pipeline, but none replace the missing
calibrated interaction layer.

## Recommended Repurposing Stack

The useful strategy is not to find one replacement model. It is to reuse
specialized pretrained representations around a small custom interaction
model:

```text
original stimulus
  -> TRIBE cortical trajectory
  -> temporal representation model
  -> persona x stimulus interaction decoder
  -> calibrated action probabilities and uncertainty
  -> OASIS social propagation
  -> separate domain outcome model
```

These pretrained additions are intended to reduce Qwen's freedom to improvise.
They should compute representations, priors, and probabilities before Qwen is
called. Qwen should operate as a constrained policy reasoner and persona-voice
renderer over those values, not infer them independently from prose.

### 1. MOMENT For TRIBE Trajectory Representation

- Model: `AutonLab/MOMENT-1-small`
- Purpose: encode temporal patterns in derived TRIBE parcel/network
  trajectories, including peaks, sustained responses, reversals, and change
  points.
- Practicality: the small checkpoint is approximately 152 MB and is the
  lowest-risk model to prototype locally.
- License: MIT.

MOMENT is a general time-series foundation model, not a neuroscience decoder.
It should receive normalized ROI/network trajectories, not all 20,484 cortical
vertices directly. Its embedding must only be retained if it improves
participant-response prediction on held-out stimuli over simpler temporal
statistics.

### 2. BrainLM As A Neuroscience-Specific Alternative

- Model: `vandijklab/brainlm`
- Purpose: test whether a model pretrained on brain-activity recordings
  provides more useful temporal/spatial representations than MOMENT.
- Practicality: released checkpoints range from a small historical model to a
  roughly 2.63 GB 650M checkpoint; the repository totals approximately 9.4 GB
  including optimizer files.

BrainLM is scientifically closer to TRIBE output, but it is not a direct
drop-in. Its expected brain representation and training distribution differ
from TRIBE's fsaverage5 predicted cortical trajectories. A projection layer and
fine-tuning are required. Test it after the MOMENT baseline, not before.
The released BrainLM checkpoint expects AAL-424 ROI trajectories. Brain-DiT's
released downstream configuration likewise expects AAL-424 and 200 time
points. TRIBE's cortical fsaverage5 surface output cannot be honestly relabeled
as AAL-424, which contains volumetric and subcortical regions. Keep both models
ineligible until a validated surface/volume plus subcortical translation exists.

### 2b. Stronger fMRI Representation Candidates

The research comparison set should also include:

- **Brain-DiT conditional:** the strongest transfer hypothesis because its
  pretraining spans resting, task, naturalistic, disease, and sleep fMRI
  states. It consumes ROI time series and is the `research_max` candidate.
- **Brain-JEPA:** an efficient neuroscience-specific candidate using
  spatiotemporal ROI dynamics. Its released checkpoint is substantially smaller
  than Brain-DiT and may be the better demo model if held-out accuracy is
  statistically equivalent.
- **NeuroSTORM:** a strong large-scale fMRI foundation model, but it expects 4D
  volumetric fMRI. It requires a defensible surface-to-volume adapter before it
  can consume TRIBE fsaverage5 predictions.

None is assumed superior before the matched held-out participant-response
benchmark. Pretraining task and input geometry matter more than paper headline
metrics.

### 3. Minitaur As A Human-Choice Teacher

- Model: `marcelbinz/Llama-3.1-Minitaur-8B-adapter`
- Apple-Silicon conversion candidate:
  `HillPhelmuth/Llama-3.1-Minitaur-8B-mlx-4Bit`
- Purpose: provide a human-choice-oriented behavioral baseline or teacher,
  instead of relying only on an assistant-tuned Qwen model.

Minitaur should consume a compact, auditable behavioral state and trial
description. It must not be treated as a decoder of raw cortical values. Its
predictions can be compared with Qwen, ensembled, or distilled into the custom
interaction decoder.

### 4. User-Item Interaction Model For The Missing Bridge

The central model should be a small recommender-style cross-interaction model:

```text
user/persona features x stimulus/TRIBE features -> response distribution
```

Suitable architectures include a two-tower model with cross-attention,
DeepFM/DCN-style feature crosses, or a small gated transformer. Unlike another
general LLM, this structure explicitly learns that the same shared TRIBE
response can cause different reactions for different people.

Pretrained recommender weights are generally less transferable because feature
schemas and outcomes differ. Reuse the architecture, but train the interaction
weights on paired participant/stimulus responses.

### 5. Temporal Graph Models For Propagation Calibration

- Model family: Temporal Graph Networks, JODIE, and related dynamic graph
  models.
- Purpose: learn or validate exposure, influence, and propagation dynamics from
  timestamped interaction traces.

These models complement OASIS. They should calibrate network transition
probabilities or act as propagation baselines, not replace the upstream
neuro-behavior decoder.

## Integration Gate

No reused model should enter the live simulation merely because it produces
plausible embeddings. Each component must pass a paired held-out test:

| component | required comparison | acceptance criterion |
| --- | --- | --- |
| MOMENT/BrainLM encoder | simple temporal statistics | improves held-out participant-response likelihood or calibration |
| persona interaction model | shared population prior | improves held-out subgroup and participant predictions |
| Minitaur teacher | Qwen-only behavior policy | improves held-out human-choice prediction |
| temporal graph model | current OASIS transition rules | improves held-out propagation trajectory metrics |
| complete bridge | LLM-only and shuffled-neuro conditions | true matched TRIBE features outperform both |

If true TRIBE features do not beat shuffled TRIBE features, the system has not
demonstrated that the neural representation adds useful stimulus-specific
information.

## Model Promotion Rule

Select models on the accuracy-efficiency Pareto frontier, not by parameter
count. The largest or newest model establishes an accuracy ceiling. Promote a
lighter model when its held-out result is statistically indistinguishable from
the best candidate within a predeclared tolerance.

Record for every candidate:

- held-out predictive score and bootstrap confidence interval
- calibration error
- matched-versus-shuffled TRIBE improvement
- median and p95 stage latency
- peak resident/unified memory
- artifact size
- failure rate

Default promotion rule:

1. Reject any model that does not beat the required baseline or matched-neuro
   ablation.
2. Find the best validated score.
3. Retain candidates within one standard error of the best score and with no
   material calibration regression.
4. Among retained candidates, choose the lowest peak-memory model, then lowest
   p95 latency.
5. Keep the accuracy leader available as an offline quality mode if it provides
   a measurable but expensive improvement.

This should yield two profiles when justified:

- `demo_fast`: smallest statistically equivalent validated stack.
- `research_max`: highest validated accuracy regardless of latency.

## Most Relevant Released Model

### Centaur / Minitaur

- `marcelbinz/Llama-3.1-Centaur-70B`
- `marcelbinz/Llama-3.1-Minitaur-8B`
- `marcelbinz/Llama-3.1-Minitaur-8B-adapter`
- `marcelbinz/Psych-101`

Centaur was fine-tuned on Psych-101, containing trial-level choices from more
than 60,000 participants and over 10 million choices across 160 psychological
experiments. It is designed to predict human choices in experiments expressed
in natural language.

Why it is relevant:

- It is optimized for human-choice prediction rather than assistant-style
  correctness.
- It can serve as a behavioral baseline or teacher model.
- Psych-101 provides a strong benchmark for general human decision simulation.

Why it is not sufficient:

- It does not consume TRIBE cortical predictions.
- It does not provide validated emotion or subgroup-response decoding.
- It does not simulate social exposure, networks, or propagation.
- Minitaur-8B full BF16 weights are approximately 16.1 GB. Its adapter is only
  approximately 93 MB, but requires a compatible Llama 3.1 8B base model.

Recommended use:

1. Benchmark Minitaur against Qwen on held-out Psych-101 tasks.
2. Use Minitaur as a behavioral teacher or secondary agent-decision backend.
3. Distill useful choice-prediction behavior into a smaller local adapter.
4. Do not treat it as a neuroscience decoder.

## Useful Validated Datasets

### OpenLAV

- 188 openly licensed videos.
- Ratings from 422 participants.
- Average of approximately 71 ratings per video.
- Includes valence, arousal, appraisal ratings, emotion labels, and personality
  traits.

Best first dataset for learning:

`TRIBE video response + participant traits -> participant affective ratings`

### Emo-FilM

- Naturalistic films with fMRI from 30 participants.
- Detailed emotion annotations from 44 raters.
- Includes appraisal, motivation, expression, physiology, and feeling labels.

Best dataset for validating whether TRIBE response features retain information
that predicts detailed human annotations.

### EMAP

- Short affective video clips.
- Neuro/peripheral physiological measurements.
- Valence and arousal ratings from 145 individuals.
- Participant demographics and video metadata.

Useful for testing participant variability and multimodal physiological
calibration.

### OpenfMRI Affective Videos (`ds000205`)

- Dynamic naturalistic audiovisual stimuli.
- Individual fMRI response and core-affect labels.

Useful for comparing TRIBE predictions with measured fMRI and individual
valence/arousal responses.

### Political Advertisement Experiments

- Yale archive of 59 randomized real-time campaign-ad experiments.
- Swayable archive described in published research: 617 ads, 146 experiments,
  and more than 500,000 respondents.

Useful for validating final subgroup persuasion predictions when stimulus files
and participant-level outcomes are available.

## Models That Are Not The Missing Bridge

- Text, audio, facial, and video emotion classifiers provide useful semantic
  baselines, but do not decode TRIBE output.
- EEG foundation models operate on measured EEG, not predicted cortical BOLD.
- Brain-to-image and fMRI decoding models reconstruct stimuli from measured
  brain activity; they solve the reverse problem.
- Generic sentiment models predict linguistic sentiment, not participant
  reaction.

## Recommended Custom Model

Build a small hierarchical neuro-behavior interaction model:

```text
inputs:
  TRIBE temporal/network features
  stimulus semantic features
  persona/subgroup features
  baseline belief and prior behavior
  source/exposure context

outputs:
  attention probability
  valence/favorability distribution
  arousal distribution
  trust shift
  avoidance/engagement probability
  sharing/comment/action probabilities
  uncertainty
```

Suggested architecture:

1. Temporal TRIBE feature encoder that preserves peaks and sustained response.
2. Persona/subgroup embedding learned from observed participant attributes.
3. Interaction layer using gated cross-features or a small transformer.
4. Multi-task probabilistic heads for ratings and actions.
5. Hierarchical random effects for participant and stimulus variation.
6. Calibration layer with uncertainty estimates.

This model should be small enough to train locally after TRIBE features are
precomputed. The primary bottleneck is validated paired data, not compute.

## Realistic Quality Ceiling

Without paired participant-response data, the bridge can only be a transparent
heuristic. It may improve simulation realism but cannot support strong accuracy
claims.

With OpenLAV/EMAP/Emo-FilM and domain-specific outcome data, a research-grade
prototype can:

- prove whether TRIBE adds predictive value beyond stimulus-only models
- learn non-uniform persona interactions
- produce calibrated uncertainty
- reduce repeated prompt anchoring
- initialize explicit OASIS agent states

It cannot honestly guarantee accurate individual behavior or universal
cross-domain prediction. Performance must be measured on held-out stimuli,
participants, subgroups, and domains.

## Requirements

### Data

- Original text, audio, or video stimuli.
- Participant-level ratings and choices, not only aggregate labels.
- Participant attributes relevant to subgroup response.
- Repeated measurements across participants and stimuli.
- Historical social exposure and action traces for propagation calibration.
- Final domain outcomes for a separate outcome model.

### Engineering

- TRIBE temporal feature export and caching.
- Dataset adapters with strict train/test separation.
- Persona feature schema.
- Numerical per-agent state in OASIS.
- Direct action-probability modifiers.
- Calibration, ablation, and uncertainty evaluation.

### Compute And Storage

- Current Mac Studio is sufficient for TRIBE feature generation and training a
  small interaction model.
- MOMENT-1-small is approximately 152 MB, MOMENT-1-base approximately 454 MB,
  and MOMENT-1-large approximately 1.39 GB per weight format. The repository
  may expose duplicate PyTorch and Safetensors copies, so download only the
  required file.
- BrainLM's full repository is approximately 9.4 GB because it includes
  multiple checkpoints and optimizer states. Download one selected checkpoint,
  not the repository.
- Minitaur-8B can likely run quantized locally, but the current machine has
  limited free disk space and no compatible Llama 3.1 8B base model installed.
- Do not download the 16.1 GB BF16 model until disk space is increased.

### Local Runtime And Downloads

The upstream `momentfm==0.1.4` package pins stale NumPy, Transformers, and
Hugging Face versions. Its code has been smoke-tested successfully against the
newer backend stack. Install the patched runtime without downgrading shared
dependencies:

```bash
backend/scripts/install_neuro_bridge_runtime.sh
```

Resume all selected external-SSD model and benchmark downloads:

```bash
backend/.venv/bin/python backend/scripts/download_behavior_components.py all
```
