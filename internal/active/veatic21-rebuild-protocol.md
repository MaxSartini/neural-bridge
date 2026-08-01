# VEATIC 2.1 Rebuild Protocol

This checklist is derived from
`internal/active/veatic21-master-scientific-specification.md`. The master specification wins
if any wording differs. Read only the sections named by `CURRENT_STATE.md` for the currently
authorized action.

## Authority and protected-root check

- [ ] Read `AGENTS.md`, the complete master specification, and `CURRENT_STATE.md` in order.
- [ ] Confirm branch is `main` and the worktree has no unrelated overlapping edits.
- [ ] Confirm the exact authorized phase/action.
- [ ] Confirm these protected roots exist and are outside every write/delete boundary:
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/`
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/`
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/`
- [ ] Reject any request or computed path that would modify a protected root.
- [ ] Reject every AGAIN runtime/code/artifact path from VEATIC execution inputs.

## Canonical input boundary

- [ ] Discover all 124 numeric video IDs independently in the TRIBE and V-JEPA roots.
- [ ] Join video `v` only to video `v`.
- [ ] Join `rows.csv` row index `i` only to TRIBE payload position `i`.
- [ ] Treat `rows.csv` as label and interpolation authority.
- [ ] Treat TRIBE `cortical_prediction` as the primary real representation.
- [ ] Treat `temporal_diagnostics53` as an explicit video-derived diagnostic/control/fusion
      block, never an implicit part of the real cortical lane.
- [ ] Treat luma/motion/quality as audit/nuisance fields and retain primary rows.
- [ ] Exclude `tribe_grouped_video_feature` from feature discovery.
- [ ] Never open, map, hash, copy, or inspect `vjepa21_hidden_states.npz`.
- [ ] Never rerun V-JEPA or TRIBE.
- [ ] Assemble the allowlisted TRIBE payload, `rows.csv`, and small alignment metadata into
      `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`.
- [ ] Verify source/destination hashes and atomically seal the bundle.
- [ ] After sealing, require all Phase 00+ downstream reads to use only the consolidated root.
- [ ] Refuse to overwrite or mutate an existing consolidated root.

## Phase 00 — protected-input foundation

- [ ] Implement the auditor from scratch under a fresh VEATIC namespace.
- [ ] Add literal protected-root deletion guards and forbidden-hidden-state read guards.
- [ ] Verify video-ID equality and exact `0..123` coverage.
- [ ] Verify per-video status and required files.
- [ ] Verify row-count equality and contiguous `(video_id,row_index)` identity.
- [ ] Verify timestamps, 2 Hz cadence, source-frame/interpolation values, arousal, and valence.
- [ ] Verify every cortical matrix shape, dtype, and finiteness.
- [ ] Inventory every TRIBE array and freeze its role.
- [ ] Produce allowlisted input digests without touching hidden states.
- [ ] Copy only allowlisted files into one per-video consolidated schema and prove byte
      identity with the protected sources.
- [ ] Emit a per-video manifest, schema report, mismatch ledger, protected-root audit,
      forbidden-input audit, derivation ledger, and artifact manifest.
- [ ] Run focused tests, authority-contract tests, formatting/type checks, and the full suite.
- [ ] Inspect all outputs, update the handoff with exact hashes, commit, and push.
- [ ] Do not fit a model, PCA, target threshold, or projection.

## Phase 01 — alignment, dynamics, targets, and splits

- [ ] Build the immutable supervised table from Phase 00 identity.
- [ ] Preserve arousal, valence, masks, interpolation, quality, and history availability.
- [ ] Measure VEATIC autocorrelation, movement, duration, event support, threshold stability,
      and between-video heterogeneity before cortical scoring.
- [ ] Derive candidate history depths, forecast windows, washout gaps, and quantiles from
      VEATIC; keep AGAIN-like values only as declared comparability anchors.
- [ ] Register max-change, absolute-change, onset/surprise, signed-change, and residualized
      candidates where supported.
- [ ] Freeze target-overlap ledgers and reject history/future overlap.
- [ ] Freeze grouped-video, blocked-forward, nested-inner, fresh-seed, and zero-label locked
      ownership.
- [ ] Do not read cortical outcome scores.

## Phase 02 — strong target-specific AR

- [ ] Iteration 1: simple current/previous/mean/slope baselines and support audit.
- [ ] Iteration 2: regularized linear/ranking AR across VEATIC-derived histories.
- [ ] Iteration 3: convergence and active-boundary expansion.
- [ ] Iteration 4: compact nonlinear AR challenger where justified.
- [ ] Iteration 5: fresh-seed/fold confirmation and immutable prediction seal.
- [ ] Use response history only; no cortical or diagnostic video features.
- [ ] Use hierarchical screens and successive promotion, not a blind millions-cell product.
- [ ] Freeze exact fold/seed AR scores for every later real/control comparison.

## Phase 03 — raw cortical benchmark

- [ ] Benchmark raw/full and deterministic label-free raw summaries on actual hardware.
- [ ] Score frozen AR, raw-only, AR+raw, explicit diagnostic fusion, diagnostics-only,
      shuffled, random, time, and quality/motion/luma lanes.
- [ ] Keep blocked and grouped results separate.
- [ ] Preserve a negative raw result; do not tune a projection from outer outcomes.

## Phase 04 — fold-owned representation discovery

- [ ] Fit PCA/projections only on owned training rows.
- [ ] Derive widths from VEATIC rank/variance/memory and test active boundaries.
- [ ] Compare current, difference, causal mean/std/slope, PCA-then-temporal, and
      temporal-then-PCA families.
- [ ] Compare PCA-only, AR+PCA, frozen-AR residual, and explicit diagnostics fusion.
- [ ] Run every matched control with equivalent budget.
- [ ] Freeze the representation before learned-head discovery.

## Phase 05 — learned bridge and event head

- [ ] Restore selected checkpoints and use deterministic evaluation mode from the first cell.
- [ ] Reuse one byte-identical frozen AR score under real and all residual controls.
- [ ] Compare direct fusion with inner-owned no-harm residual correction.
- [ ] If blocked improvement fails, run AR-dominance decomposition before changing models.
- [ ] Activate only preregistered VEATIC washout/target-redesign branches.
- [ ] Fairly screen current MLP, delta MLP, short causal convolution, gated/low-confidence
      temporal residual, and one justified recurrent/attention challenger.
- [ ] Search capacity/optimizer/loss/budget through staged screens and boundary expansion.
- [ ] Freeze one event recipe and confirm it with fresh blocked and grouped evidence.

## Phase 06 — event stabilization

- [ ] Test a predeclared equal-weight independent-checkpoint reference ensemble first.
- [ ] Derive checkpoint count from VEATIC; never inherit AGAIN's count.
- [ ] Compare ensemble against members, frozen AR, and matched controls.
- [ ] Prohibit member selection or weight tuning on confirmation outcomes.
- [ ] Run blocked confirmation before separately locked grouped confirmation.

## Phase 07 — continuous arousal

- [ ] Derive continuous targets and ranking-aware AR independently.
- [ ] Re-search loss, context, head, optimizer, checkpointing, and controls.
- [ ] Use Spearman and top-tail true-movement lift as primary endpoints.
- [ ] Treat exact-value metrics as a separate claim family.
- [ ] Run independent blocked and grouped confirmations and stabilization.

## Phase 08 — valence

- [ ] Audit valence reliability/dynamics and derive level, signed-change, magnitude, and
      transition targets.
- [ ] Rebuild strong valence AR, raw controls, projection, learned bridge, confirmation, and
      stabilization from VEATIC evidence.
- [ ] Do not inherit the arousal representation/head without a challenger test.

## Phase 09 — zero-label at inference

- [ ] Begin only after supervised arousal event, continuous arousal, and valence closure.
- [ ] Stage 0 freezes development/locked ownership, inference allowlist, row-0 behavior,
      feature transforms, predictions-before-label seal, controls, metrics, seeds, and gates.
- [ ] Stage A prioritizes direct supervised causal temporal video-only learning.
- [ ] Compare current-row, diagnostics-only, no-video, sequence-shuffled, and label-permuted
      controls.
- [ ] Keep response-history teacher systems as report-only ceilings.
- [ ] Stage B writes/checksums locked predictions before opening labels and scores once.
- [ ] Report full-video, cold-start, video-block uncertainty, and panel consistency.

## Comprehensive search-sufficiency checklist

- [ ] Every scientifically distinct family has at least one converged fair-budget cell.
- [ ] Active hyperparameter boundaries are expanded or explicitly unresolved.
- [ ] Undertrained/nonconverged cells are incomplete, not negative.
- [ ] Every attempted cell has a terminal disposition.
- [ ] Promotion and ties were frozen before scores were opened.
- [ ] Fresh confirmation uses a completely frozen recipe.
- [ ] No known AGAIN implementation mistake was reenacted.
- [ ] Any reopened AGAIN-failed branch has a VEATIC-specific preregistered rationale.
- [ ] Full candidate arithmetic, successes, failures, and invalid cells are reproducible.

## Controls from the first applicable cell

- [ ] simple response-history baselines;
- [ ] strongest selected AR;
- [ ] raw/real representation;
- [ ] shuffled representation;
- [ ] matched random representation;
- [ ] diagnostics-only;
- [ ] time/mask-only;
- [ ] luma/motion/quality nuisance;
- [ ] train-only video mean/base rate;
- [ ] current-row temporal ablation;
- [ ] training-owned label permutation;
- [ ] no-video zero-label control;
- [ ] frozen-AR checksum identity and residual no-harm.

## Metrics and uncertainty

- [ ] Event primary: PR-AUC, uplift over prevalence, delta vs AR, delta vs best control.
- [ ] Continuous primary: Spearman and preregistered top-tail true-movement lift.
- [ ] Valence separates level, signed direction, and movement magnitude.
- [ ] Exact-value metrics cannot silently become ranking claims or vice versa.
- [ ] Report fold/video/seed/checkpoint-group consistency and maximum group contribution.
- [ ] Use video-block uncertainty; never treat 2 Hz rows as IID replicates.
- [ ] Retain valid negatives from zero-event videos; leave undefined per-video PR-AUC undefined.

## Hardware and execution

- [ ] Benchmark representative real end-to-end cells before every material main run.
- [ ] Compare safe CPU process, MLX concurrency, GPU batch, and pipeline configurations.
- [ ] Require numerical/convergence/artifact equivalence.
- [ ] Measure repeated throughput, utilization, memory, swap, and thermal state.
- [ ] Freeze the fastest safe topology before main execution.
- [ ] Use one coordinator, deterministic work assignment, atomic outputs, shard ledgers, and a
      verified merge.
- [ ] Treat any executor/topology change after launch as a new run identity.
- [ ] Benchmark inference as seriously as training.

## Phase transition

- [ ] Inspect every compact and external output.
- [ ] Independently verify registry coverage and artifact hashes.
- [ ] Run focused, authority-contract, formatting/type, and full tests.
- [ ] Create the compact concluded phase record.
- [ ] Replace `CURRENT_STATE.md` while retaining mandatory authority anchors.
- [ ] Record exact code/input/result/manifest/checksum hashes and one next action.
- [ ] Commit and push the coherent transition directly to `origin/main`.
- [ ] Start no later phase before the transition exists on remote `main`.
