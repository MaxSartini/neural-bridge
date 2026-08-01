# VEATIC 2.1 Rebuild Protocol

This checklist is derived from
`internal/active/veatic21-master-scientific-specification.md`. The master specification wins
if any wording differs. Read only the sections named by `CURRENT_STATE.md` for the currently
authorized action.

Active transition: Phase 00 passed. Phase 01 VEATIC target/geometry/ownership derivation is the
only executable scientific work. Phase 02 is fully specified but remains blocked until the
Phase 01 transition reaches `origin/main`.

## Authority and protected-root check

- [ ] Read `AGENTS.md`, the complete master specification, the supervised combination
      specification, and `CURRENT_STATE.md` in order.
- [ ] Confirm branch is `main` and preserve unrelated changes.
- [ ] Confirm the exact authorized action.
- [ ] Confirm these protected roots are present and outside every write/delete boundary:
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/`
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/vjepa-2.1/`
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`
  - `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/again/`
- [ ] Reject every computed path that would modify a protected root.
- [ ] Reject every non-VEATIC runtime, code, fitted artifact, row, or prediction as a VEATIC
      execution input.

## Canonical input boundary

- [ ] Use only
      `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge-input/`
      for downstream VEATIC reads.
- [ ] Require numeric video IDs `0..123` and join video `v` only to video `v`.
- [ ] Join `rows.csv` row index `i` only to TRIBE payload position `i`.
- [ ] Treat `rows.csv` as label and interpolation authority.
- [ ] Treat `cortical_prediction` as the primary real representation.
- [ ] Treat `temporal_diagnostics53` as an explicit diagnostic/control/fusion block.
- [ ] Treat luma/motion/quality as audit and nuisance fields.
- [ ] Exclude `tribe_grouped_video_feature` from feature discovery.
- [ ] Never open, map, hash, copy, or inspect `vjepa21_hidden_states.npz`.
- [ ] Never rerun V-JEPA or TRIBE.

## Phase 00 — protected-input foundation

- [x] Verify all 124 videos, 20,657 rows, row identity, cadence, alignment, labels, cortical
      shape/dtype/finiteness, schema roles, source/destination hashes, forbidden-input guards,
      and read-only sealing.
- [x] Create no predictive target, projection, model, or accuracy claim.

## Phase 01 — VEATIC targets, geometry, and ownership

- [ ] Read every authoritative VEATIC label row and only the non-cortical audit allowlist.
- [ ] Measure arousal and valence autocorrelation, partial correlation, movement, duration,
      interpolation, threshold stability, and between-video heterogeneity.
- [ ] Extend ACF/PACF with explicit tapering video and pair support.
- [ ] Compute each same-video future-trajectory primitive once per label and geometry.
- [ ] Derive washout, horizon, response-history, event-threshold, and duration-panel candidate
      values from VEATIC support.
- [ ] Register continuous movement and event views with exact target-overlap ledgers.
- [ ] Freeze strict-forward blocked/inner memberships and whole-video grouped folds with exact
      row/video hashes.
- [ ] Use every supported video in blocked evidence and every video across the grouped fold
      set.
- [ ] Emit the numeric inputs required by
      `internal/active/veatic21-supervised-spike-continuous-combination.md`.
- [ ] Read no cortical or temporal-diagnostic value and fit no predictive model.

## Phase 02 — supervised spike + continuous combination

Implement only
`internal/active/veatic21-supervised-spike-continuous-combination.md`.

### Required dependency build

- [ ] Materialize target tensors, target masks, thresholds, and exact ownership.
- [ ] Fit the train-owned continuous-target residualizer where selected.
- [ ] Train and seal a strong event response-history opponent.
- [ ] Train and seal a separate strong continuous response-history opponent.
- [ ] Reuse each opponent prediction byte-identically under its real and control lanes.
- [ ] Build causal cortical temporal aggregates from `cortical_prediction`.
- [ ] Fit scaler and projection separately for every distinct target mask and outer ownership.
- [ ] Construct causal projected sequences with same-video reset and explicit start padding.
- [ ] Add `temporal_diagnostics53` only as an explicit current-row block.
- [ ] Train the dual-correction event residual specialist.
- [ ] Train the separately optimized continuous residual specialist.
- [ ] Run every matched control with identical ownership, budget, checkpointing, and metrics.
- [ ] Freeze fresh independent checkpoint membership before outer scoring.
- [ ] Compare equal-weight ensembles with all members, opponent ensembles, and control
      ensembles.
- [ ] Publish one paired package returning confirmed `spike_score` and
      `continuous_score`.

### VEATIC-owned selection

- [ ] Derive compact local candidates for geometry, history, projection width, causal context,
      hidden capacity, optimizer, loss balance, scale/gate limits, checkpoint budget, seed
      count, and ensemble size.
- [ ] Use unit/integrity tests first, then all eligible rows from all 124 videos.
- [ ] Expand only an active numeric boundary or an undertrained configuration.
- [ ] Do not create candidates from outer results.
- [ ] Give every registered cell a terminal disposition.

### Confirmation

- [ ] Freeze the complete event and continuous recipes before fresh confirmation.
- [ ] Keep blocked and grouped claims separate.
- [ ] Require each specialist to beat its exact opponent and strongest matched control.
- [ ] Require positive fold/checkpoint-group consistency and no single-group domination.
- [ ] Require restored checkpoints, deterministic evaluation mode, and complete row,
      projection, prediction, and checksum audits.
- [ ] Keep the two specialists as the paired output unless a preregistered shared-head
      challenger matches or beats both locked endpoints.

## Phase 03 — valence

- [ ] Begin only after the supervised arousal pair passes.
- [ ] Derive VEATIC valence level, signed-direction, magnitude, and transition targets.
- [ ] Build valence-specific opponents, projections, heads, controls, ensembles, metrics, and
      claims.
- [ ] Do not assume any arousal numeric setting transfers.

## Phase 04 — zero-label at inference

- [ ] Begin only after supervised arousal and valence pass.
- [ ] Register development and locked whole-video confirmation ownership at Phase 04 entry.
- [ ] Permit labels during training but no label, response history, teacher score, or labelled
      warm start during held-out inference.
- [ ] Prioritize direct supervised causal temporal video-only learning.
- [ ] Compare current-row, diagnostics-only, no-video, sequence-shuffled, and
      label-permutation controls.
- [ ] Seal predictions before opening confirmation labels.

## Phase 05 — paper and product

- [ ] Report complete results, uncertainty, negative findings, and exact provenance.
- [ ] Keep deployment refits distinct from evidence models.
- [ ] Benchmark cold-start and steady-state inference, batch scaling, CPU/GPU utilization,
      unified memory, deterministic equivalence, raw-video integration, no-harm behaviour,
      and external client-style validation.

## Controls

- [ ] prevalence/constant and simple response-history baselines;
- [ ] strongest target-specific response-history opponent;
- [ ] real cortical residual;
- [ ] shuffled projected cortical residual;
- [ ] matched random projected residual;
- [ ] diagnostics-only residual;
- [ ] train-only video-mean residual;
- [ ] time/mask residual;
- [ ] luma/motion/quality residual;
- [ ] current-row cortical ablation;
- [ ] training-owned label-permutation residual;
- [ ] frozen-opponent checksum identity and residual no-harm.

## Metrics and uncertainty

- [ ] Event primary: PR-AUC, uplift over prevalence, delta versus the exact opponent, and delta
      versus the strongest matched control.
- [ ] Continuous primary: Spearman and registered top-tail true-movement lift.
- [ ] Record event PR-AUC and top-k recall derived from the continuous score as identified
      joint evidence.
- [ ] Treat exact-value metrics as a separate claim.
- [ ] Use video/fold blocks for uncertainty; never treat 2 Hz rows as IID.
- [ ] Report fold/video/seed/checkpoint-group consistency and maximum group contribution.

## Hardware and execution

- [ ] Benchmark representative real end-to-end cells before each material run.
- [ ] Compare safe CPU process counts, MLX concurrency, GPU batches, and CPU/GPU pipelines.
- [ ] Require numerical, convergence, split, metric, and artifact equivalence.
- [ ] Measure repeated throughput, utilization, memory, swap, and thermal state.
- [ ] Freeze the fastest safe topology before the main run.
- [ ] Use one coordinator, deterministic work assignment, atomic outputs, append-only shard
      ledgers, and a verified no-gap/no-duplicate merge.
- [ ] Treat any topology change after launch as a new run identity.

## Phase transition

- [ ] Inspect every compact and external output.
- [ ] Verify registry coverage and artifact hashes independently.
- [ ] Run focused, authority-contract, formatting/type, and full tests.
- [ ] Replace `CURRENT_STATE.md` while retaining mandatory authority anchors.
- [ ] Record exact code/input/result/manifest/checksum hashes and one next action.
- [ ] Commit and push the coherent transition directly to `origin/main`.
- [ ] Start no later phase before the transition exists on remote `main`.
