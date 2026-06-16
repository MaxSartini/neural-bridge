# Neural Bridge Roadmap

This is the post-VEATIC-124 v2 roadmap. VEATIC proved the core cortical/TRIBE hypothesis for arousal event/spike ranking. The roadmap now focuses on preserving that evidence, turning it into a stable training contract, and building better heads on top of the proven signal.

## Proven Baseline

Completed:

- VEATIC-124 manifest and cortical/TRIBE cache are complete.
- Arousal future-spike/event ranking has validated signal from cortical/TRIBE features.
- PCA feature modes beat AR, shuffled, random, timestamp, and video/time controls on the strongest spike/event rows.
- Official split spike rows pass controls across current feature families.
- Grouped-video spike F1 improves over AR for PCA modes.
- Balanced event-vs-stable sampling confirms event-conditioned discrimination.
- Temporal context v2 shows short causal windows can improve selected spike-ranking rows.
- Alignment repair selected the final benchmark policy: keep current 0s alignment primary and report offset grids as diagnostics.
- Small v2 evidence reports are tracked in this repo.

This is the current scientific foundation, not a hypothesis waiting for another dataset to validate it.

## 1. Evidence Freezing And Reproducibility

Goal: make the proven v2 baseline impossible to lose or confuse with old runs.

- [ ] Create a protected external snapshot for the VEATIC-124 v2 manifest, cache metadata, benchmark JSON/CSV outputs, and tracked reports.
- [ ] Add checksums and a manifest that identifies the authoritative files.
- [ ] Mark superseded pre-v2 artifacts as deleted, archived, or retained with an explicit reason.
- [ ] Add one verification command that rechecks the v2 evidence bundle without re-encoding videos.
- [ ] Keep small summary reports in git; keep heavy caches and raw outputs external.

## 2. Benchmark Contract

Goal: preserve the exact rules that made the v2 result credible.

- [x] Audit spike rows that prefer non-zero offsets.
- [x] Select 0s alignment as the primary non-leaky benchmark baseline.
- [x] Keep offset grids as diagnostics rather than final-score corrections.
- [x] Confirm no future-looking feature leakage in causal/delta/window features.
- [ ] Encode the alignment, split, threshold, transform, and reporting rules into a reusable benchmark contract.
- [ ] Add a compact status command that verifies the benchmark contract against the current artifact set.

## 3. v2 Training Tensor Contract

Goal: make new heads consume the proven v2 data without changing the benchmark under them.

- [ ] Define immutable tensor contracts for `cortical_pca_64`, `cortical_pca64_delta`, `cortical_global`, and raw cortical trajectories.
- [ ] Record target definitions, masks, split fields, temporal windows, and train-only transform state.
- [ ] Add loader tests that fail on leakage-prone transforms.
- [ ] Store immutable training tensors externally with manifest and checksum metadata.
- [ ] Build a minimal baseline head before adding recursive or larger architectures.

## 4. Next Model Heads

Goal: improve event/spike ranking from the v2 baseline without weakening controls.

- [ ] Start with simple, auditable heads over the frozen tensor contract.
- [ ] Compare against the v2 PCA/ridge baseline, AR, shuffled, random, timestamp, and video/time controls.
- [ ] Keep grouped-video and blocked validation as required gates.
- [ ] Only promote recursive heads after simple heads define the floor.
- [ ] Keep CUDA-only HRM-style dependencies out of the main Mac/MPS environment unless isolated.

## 5. Productized Evidence Workflow

Goal: make a fresh session or teammate able to inspect, verify, and extend the proven result without archaeology.

- [ ] Add a status check for external assets, VEATIC cache, model paths, and tracked evidence artifacts.
- [ ] Add a compact benchmark dashboard or CLI summary for the v2 baseline.
- [ ] Add a one-command evidence verifier for the tracked reports and external artifact snapshot.
- [ ] Remove or archive stale run folders once their useful evidence is preserved.
- [ ] Keep machine-specific paths only in local `.env` files.
- [ ] Keep generated heavy outputs out of git.

## De-Scoped

- Additional OpenLAV validation as a roadmap item. OpenLAV was useful as an early signal, but VEATIC-124 v2 proved the core hypothesis.
- Subcortical expansion as a roadmap item. The current priority is the proven cortical/TRIBE signal.
- Video `83` as an active roadmap concern. Its resampling policy is documented and does not block the v2 claim.
- Simulation/LLM-agent integration as a primary roadmap focus.
- Finance or quant-desk prediction.
- Generic chatbot benchmarks as proof of Neural Bridge.
- Exact continuous arousal-value forecasting as the current headline.
- Test-selected lag correction as a headline result.
- CUDA-only training stacks in the main local environment.
