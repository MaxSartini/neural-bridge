# Neural Bridge Roadmap

This is the post-v2 roadmap. It starts from what VEATIC-124 v2 has already proven and lists only the work that still moves the project forward.

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

This baseline should be treated as the current scientific foundation, not a hypothesis waiting for basic proof.

## 1. Evidence Freezing And Reproducibility

Goal: make the proven v2 baseline impossible to lose or confuse with old runs.

- [ ] Create a protected external snapshot for the VEATIC-124 v2 manifest, cache metadata, benchmark JSON/CSV outputs, and tracked reports.
- [ ] Add checksums and a manifest that identifies the exact authoritative files.
- [ ] Mark superseded pre-v2 artifacts as deleted, archived, or retained with an explicit reason.
- [ ] Add one verification command that rechecks the v2 evidence bundle without re-encoding videos.
- [ ] Keep small summary reports in git; keep heavy caches and raw outputs external.

## 2. Alignment Policy Preservation

Goal: preserve the resolved timing policy and prevent future work from silently replacing it with a test-derived lag correction.

- [x] Audit spike rows that prefer non-zero offsets.
- [x] Select 0s alignment as the primary non-leaky benchmark baseline.
- [x] Keep offset grids as diagnostics rather than final-score corrections.
- [x] Confirm no future-looking feature leakage in causal/delta/window features.
- [ ] Resolve the video `83` prediction/manifest length mismatch policy.
- [ ] Carry the alignment policy into the training tensor contract and future dashboards.

Done means future model work cannot accidentally claim a lag-corrected score as the benchmark headline.

## 3. v2 Training Tensor Contract

Goal: make new heads consume the proven v2 data without changing the benchmark under them.

- [ ] Define immutable tensor contracts for `cortical_pca_64`, `cortical_pca64_delta`, `cortical_global`, and raw cortical trajectories.
- [ ] Record target definitions, masks, split fields, temporal windows, and train-only transform state.
- [ ] Add loader tests that fail on leakage-prone transforms.
- [ ] Store immutable training tensors on the external drive with manifest and checksum metadata.
- [ ] Build a minimal baseline head before adding recursive or larger architectures.

## 4. Next Model Heads

Goal: improve event/spike ranking from the v2 baseline without weakening controls or replacing the resolved alignment policy.

- [ ] Start with simple, auditable heads over the frozen tensor contract.
- [ ] Compare against the v2 PCA/ridge baseline, AR, shuffled, random, timestamp, and video/time controls.
- [ ] Keep grouped-video and blocked validation as required gates.
- [ ] Only promote recursive heads after simple heads define the floor.
- [ ] Keep CUDA-only HRM-style dependencies out of the main Mac/MPS environment unless isolated.

## 5. Subcortical And Multimodal Expansion

Goal: test whether new neuro inputs add signal beyond the proven cortical baseline.

- [ ] Extract and freeze a separate VEATIC-124 subcortical cache before scoring it.
- [ ] Preserve the Harvard-Oxford ROI mapping and provenance contract.
- [ ] Compare cortical-only, subcortical-only, and cortical-plus-subcortical feature sets.
- [ ] Extend the same event/spike-ranking gates to OpenLAV or other human-response datasets.
- [ ] Do not mix new modalities into the baseline without an explicit ablation.

## 6. Simulation Injection

Goal: turn validated neural features into agent conditioning only if they improve measured simulation outcomes.

Required conditions:

- `llm_only`
- `true_neuro_current_mapping`
- `shuffled_neuro_prior`
- `neutral_neuro_prior`
- `inverted_neuro_prior`
- `oracle_behaviour_prior`
- `true_neuro_no_prompt_injection`

Additional gates:

- [ ] Dose-response tests.
- [ ] Leave-one-axis-out tests.
- [ ] Seed replication.
- [ ] Direct numeric state modifiers separate from prompt wording.
- [ ] Paired real-outcome evaluation where possible.

## 7. Product And Repo Cleanup

Goal: make the new Neural Bridge repo match the current project, not its history.

- [ ] Add a status check for external drive, TRIBE cache, Neo4j, local LLM host, and benchmark artifacts.
- [ ] Add a compact benchmark dashboard or CLI summary for the v2 baseline.
- [ ] Remove stale legacy docs and run folders once their useful evidence is preserved.
- [ ] Keep local compatibility paths out of company-facing docs; document machine-specific paths only in local `.env` files.
- [ ] Keep generated heavy outputs out of git.

## De-Scoped

- Finance or quant-desk prediction.
- Generic chatbot benchmarks as proof of Neural Bridge.
- Exact continuous arousal-value forecasting as the current headline.
- Test-selected lag correction as a headline result.
- CUDA-only training stacks in the main local environment.
