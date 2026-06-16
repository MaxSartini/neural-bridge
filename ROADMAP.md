# Neural Bridge Roadmap

This roadmap reflects the current Neural Bridge v2 direction as of 2026-06-16. It intentionally drops the older MiroFish and finance-prediction language.

## Current State

Neural Bridge is now a cleaned local repo with heavyweight research assets kept on the external drive. The most important proven path is VEATIC-124 cortical/TRIBE event and spike ranking, with temporal-context v2 showing that short causal context can improve selected future arousal spike rows.

Already in place:

- Fresh git repo under `/Users/maxsartini/Neural Bridge`.
- External asset root at `/Volumes/onn. Drive/Neural Bridge`.
- Compatibility symlink from the old external MiroFish path.
- Complete 124-video VEATIC cortical cache.
- Current 124-video manifest with 10,357 1 Hz target rows.
- Benchmark feature modes for global, delta, PCA, and PCA-delta cortical features.
- Confirmatory v2 evidence that cortical/TRIBE PCA features improve arousal future-spike/event ranking over AR, shuffled, random, timestamp, and video/time controls.
- Temporal context v2 evidence that 2s causal context improves selected spike-ranking rows over current-only evaluation.
- Local app stack: Flask backend, Vue frontend, Neo4j graph services, and local LLM integration.
- Recursive model source checkouts read and classified as research candidates, not main-runtime dependencies.

## Phase 1 - Preserve The v2 Evidence Bundle

Goal: make the proven VEATIC-124 v2 evidence reproducible before more model work lands.

- [x] Bring the small v2 report artifacts into the cleaned repo.
- [ ] Copy the current VEATIC 124 manifest, cache metadata, benchmark outputs, and key reports into a protected baseline snapshot directory on the external drive.
- [ ] Add a baseline manifest that records source paths, file counts, feature modes, split policy, and checksum policy.
- [ ] Decide the official handling of video `83`, which currently needs resampling because prediction rows and manifest rows differ.
- [ ] Add a short command recipe for re-running only verification, not extraction.
- [ ] Mark superseded first-50 and pre-124 outputs as archived, deleted, or retained with a reason.

Done means a future thread can reproduce the v2 evidence bundle without guessing which run folder is authoritative.

## Phase 2 - Alignment And Timing Hardening

Goal: turn the strongest v2 result into a cleaner timing claim.

- [ ] Run a dedicated annotation/feature alignment pass for spike rows that prefer non-zero offsets.
- [ ] Keep test-selected lag corrections out of final headline metrics.
- [ ] Distinguish causal context gains from label/timing-structure gains with controls.
- [ ] Keep temporal-context v2's defensible claim: short causal context can improve selected future arousal spike ranking.
- [ ] Reject universal early-warning, global lag, and exact continuous arousal-value claims unless new evidence supports them.

Done means timing interpretation is stable enough for the next model head.

## Phase 3 - Training Data Contract

Goal: define exactly what model experiments are allowed to consume.

- [ ] Write a production tensor loader contract for cortical features.
- [ ] Define accepted inputs for raw cortical trajectories, `cortical_global`, `cortical_global_delta`, `cortical_pca_64`, and `cortical_pca64_delta`.
- [ ] Record target columns, masking policy, group splits, temporal windows, and causal-window rules.
- [ ] Add loader smoke tests that fail on leakage-prone transforms or missing train-only fit state.
- [ ] Decide where immutable training tensors live on the external drive.

Done means model candidates can be compared against the same data without silently changing preprocessing.

## Phase 4 - Benchmark Hardening

Goal: make result claims harder to fool.

- [ ] Keep full-frame VEATIC rows as the headline baseline.
- [ ] Keep event-conditioned and pre-event diagnostics separate unless they use balanced event-vs-stable discrimination.
- [ ] Preserve train-only thresholding and train-only PCA.
- [x] Add compact v2 reports for official split, blocked validation, grouped-video validation, event-conditioned retests, and temporal context.
- [ ] Keep CPU/MPS drift audits attached to thresholded metrics.
- [ ] Add a single command that emits the current "safe to compare" result bundle.

Done means a new feature or model cannot look better just because it used a weaker split, a tuned test threshold, or a changed device path.

## Phase 5 - Subcortical Path

Goal: evaluate subcortical features without contaminating the cortical baseline.

- [ ] Keep the cortical 124 baseline frozen first.
- [ ] Produce a separate subcortical VEATIC cache only when storage, runtime, and mapping policy are explicit.
- [ ] Preserve the Harvard-Oxford ROI mapping contract.
- [ ] Treat subcortical ROI trajectories as predictors, not labels.
- [ ] Compare cortical-only, subcortical-only, and cortical-plus-subcortical feature sets with the same splits and controls.

Done means subcortical results are additive evidence, not a hidden change to the baseline.

## Phase 6 - Simulation Injection Validation

Goal: prove whether neural features help agent simulations before presenting them as a simulator upgrade.

Required ablations:

- `llm_only`
- `true_neuro_current_mapping`
- `shuffled_neuro_prior`
- `neutral_neuro_prior`
- `inverted_neuro_prior`
- `oracle_behaviour_prior`
- `true_neuro_no_prompt_injection`

Additional checks:

- [ ] Dose-response tests.
- [ ] Leave-one-axis-out tests.
- [ ] Seed replication.
- [ ] Direct numeric state-modifier tests separate from prompt wording effects.
- [ ] Report simulation outputs against real paired outcomes where possible.

Done means neural conditioning improves a measured task for the right reason, not because prompts became more persuasive or more verbose.

## Phase 7 - Recursive Model Research

Goal: test TRM/HRM-style recursive heads only after the baseline and loader contract are stable.

Current stance:

- TinyRecursiveModels is a plausible small research candidate.
- HRM is CUDA/FlashAttention oriented and not safe to import or train directly in the Mac/MPS environment as-is.
- Do not install CUDA, Triton, or FlashAttention into the main Neural Bridge environment.

Next steps:

- [ ] Write a Mac-safe adapter plan before installing anything.
- [ ] Start with CPU or MPS-safe synthetic forward tests.
- [ ] Compare recursive heads to simple baselines on the frozen loader contract.
- [ ] Promote only if they improve the benchmark without weakening controls.

## Phase 8 - App and Developer Experience

Goal: make the cleaned repo pleasant to run and hard to misconfigure.

- [ ] Add a `/api/status` style check for Neo4j, LLM host, external drive, model paths, and benchmark cache availability.
- [ ] Add a lightweight benchmark status page or CLI summary.
- [ ] Improve error messages when external assets are missing.
- [ ] Keep old MiroFish path references only where they are intentional compatibility shims.
- [ ] Remove redundant generated outputs, stale run folders, and legacy docs once each has either been archived or superseded.
- [ ] Add focused tests for config path resolution and benchmark loader contracts.

## Explicitly De-Scoped

These should not drive current work:

- Finance/quant-desk predictor positioning.
- Generic chatbot benchmarks as proof of Neural Bridge accuracy.
- CUDA-only training stacks in the main local environment.
- Copying large model weights, raw datasets, benchmark caches, or generated outputs into git.
- Merging old run folders into the new repo without a specific reproducibility purpose.
