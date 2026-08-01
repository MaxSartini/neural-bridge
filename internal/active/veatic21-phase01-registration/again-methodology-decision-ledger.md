# AGAIN Methodology Decision Ledger for the VEATIC 2.1 Rebuild

This is a method-transfer audit, not an input manifest. The historical run root and study
documents were reviewed together because the run root preserves the actual execution graph
and the study tree explains intermediate intent and corrections. No AGAIN code, data,
prediction, fitted transform, model, checkpoint, or selected number enters VEATIC.

## Dataset translation rule

AGAIN had 995 videos and 243,575 dense 2 Hz rows. VEATIC has 124 videos and 20,657 dense 2 Hz
rows. VEATIC durations span 22–358 rows (11–179 seconds) and average about 83.3 seconds;
AGAIN averaged about 122.4 seconds. Therefore no row count, temporal window, fold count,
threshold, width, seed count, or model capacity is scaled mechanically by the ratio of
average durations. Phase 01 derives time scales from VEATIC ACF/PACF and movement dynamics,
then checks row and video support across the actual duration distribution.

`blocked_temporal_70_30` is not a 70% sample of videos. The actual AGAIN artifact reports
162,758 outer-train and 70,331 outer-test rows for the mature event target, or approximately
69.83%/30.17% after eligibility masks. The learned-head audit records strict forward time,
zero overlap, and `inner_temporal_within_outer_train_by_video_80_20`. VEATIC audits outer
fractions 0.50–0.85 and inner fractions 0.70–0.85 before selecting its own ownership.

## Phase-by-phase dispositions

### Phase 00 — dense foundation

- Evidence: 995/995 videos, 243,575 rows at true 2 Hz, no failed video, explicit sampling and
  quality provenance. There were 134 label-alignment exclusions across 38 videos and 4,816
  later quality-excluded rows across 966 videos; neither class was silently repaired.
- Retain: exact per-video identity, cadence, completeness, label equality, source-position,
  interpolation, and quality audits before modelling.
- Do not transfer: AGAIN row tables or encoding artifacts. VEATIC already owns its sealed
  124-video V-JEPA 2.1 → TRIBE v2 cortical-prediction bundle.

### Phase 01 — label alignment and target masks

- Evidence: explicit labelled-row and insufficient-history masks; thresholds were training
  owned.
- Retain: lossless continuous trajectories first, derived event views second; same-video
  causal histories; no boundary crossing; explicit negatives and undefined per-video event
  metrics.
- VEATIC expansion: do this independently for both arousal and valence, including level,
  signed change, absolute movement, max positive/negative movement, and onset/surprise.

### Phase 02 — AR opponent, five executions

The execution path was initial, revision 1, revision 2, revision 3, then final. Grouped spike
PR-AUC moved materially across these attempts (about 0.1562, 0.1562, 0.1447, 0.1562, then
0.1473), proving that “AR baseline” is a model-selection problem rather than a formality. The
final reference included target-specific blocked and grouped scores: spike 0.2036/0.1473,
short delta 0.2619/0.2084, and absolute delta 0.1160/0.1182.

- Retain: five-step VEATIC AR ladder—simple history baselines, regularized linear/ranking AR,
  convergence/boundary expansion, compact nonlinear challenger where justified, then fresh
  confirmation. Freeze exact target/fold/seed predictions for later comparisons.
- Do not transfer: AGAIN lags, alphas, targets, thresholds, or AR architectures.

### Phase 03 — raw cortical information

The raw label-free summary was target dependent. For grouped spike, raw cortical PR-AUC
0.1366 lost to AR 0.1473; AR+raw 0.1703 helped grouped, but blocked AR+raw 0.1677 lost to AR
0.2036. Raw lanes lost for short delta, while raw absolute-delta 0.1265 beat AR 0.1182.

- Retain: mandatory raw cortical-only and direct-fusion benchmarks with shuffled, random,
  time, quality, and diagnostics controls. A negative raw result does not end the programme.
- Cut as an assumed answer: naïve raw fusion and any belief that one target's result describes
  all movement constructs.

### Phase 04 — fold-owned projection and representation order

Four families were tested: current row, first difference, PCA then temporal aggregation, and
temporal aggregation then PCA, across multiple widths. The historical event winner was a
2-second temporal mean followed by PCA256 with AR and diagnostics, but the benchmark
performed 216 fits where 54 were planned and exposed avoidable duplicated work.

- Retain: projection fitted inside each ownership; test temporal-before-projection and
  projection-before-temporal as distinct families; fit a safe maximum nested PCA width once
  and slice lower widths when mathematically equivalent; expand an active boundary.
- Do not transfer: 2 seconds, width 256, diagnostics fusion, or the winning family identity.
  Current/delta/PCA-only failures in AGAIN are not universal scientific laws, but they need
  fair VEATIC screens rather than automatic promotion.

### Phase 05.0 — learned bridge and evaluation-mode correction

The initial learned bridge was scored with dropout active. The eval-mode rescore restored
selected checkpoints and changed the evidential interpretation: grouped real 0.2301 beat AR
0.2247 and the best control 0.2043, but blocked real 0.2219 lost to AR 0.2655.

- Retain: checkpoint restoration plus deterministic evaluation mode before every validation
  or test score; store training curves and restored-checkpoint identity.
- Reject: every train-mode test score. It is invalid, not a failed scientific candidate.

### Phase 05.1 — frozen-AR residual

The identical frozen AR floor underneath real and matched residual controls improved grouped
performance to 0.2383 versus AR 0.2247 but still missed blocked performance (0.2636 versus
0.2655). The correct null for a residual label permutation is the frozen AR floor, not raw
prevalence.

- Retain: byte-identical AR predictions beneath every residual lane; no-harm gating selected
  on inner data; residual-permutation interpretation.
- Reject: comparisons with separately refitted or nonidentical AR floors.

### Phase 05.2 — blocked diagnostics and clean confirmation

The targeted residual diagnostic showed only a tiny blocked increment and failed meaningful
controls. A clean confirmation reached 0.266304 versus AR 0.265967 but lost to train-only
video mean 0.266320. The dominance decomposition showed current/recent arousal was driving
the blocked task. An early continuous residual on the original target also lost to AR on
Spearman and top-tail lift.

- Retain: when blocked evidence fails, diagnose AR persistence, video means, threshold
  stability, residual variance, and temporal overlap before changing model capacity.
- Cut: promoting microscopic deltas, aggregate-only wins, or candidates that fail the
  train-only video-mean control.

### Phase 05.3 — target redesign

The rows `+4..+10` washout target was motivated by the AR-dominance diagnosis. Target
redesign alone did not win: binary 0.266369 versus AR 0.266214 still lost to video mean
0.266384, and continuous endpoints failed. Its fold-owned PCA and all target-dependent
artifacts had to be recomputed.

- Retain: a diagnosis-activated, prospectively registered VEATIC washout/onset family; refit
  every threshold, AR, projection, and head after any target identity change.
- Do not transfer: rows `+4..+10` or q90 as VEATIC selections. They remain one compact
  comparability anchor.

### Phase 05.4–05.5 — temporal head discovery and confirmation

Current-row MLP, delta MLP, short temporal convolution, and low-AR-confidence temporal
residual families were compared. The short temporal convolution won blocked selection:
0.273856 versus AR 0.266214 and best control 0.265493. Ten-seed blocked confirmation yielded
0.267074 versus AR 0.260234 and best control 0.259337, positive in 9/10 seeds. Grouped
compatibility yielded 0.231383 versus AR 0.217495 and control 0.217421, positive in 50/50
groups. A residual-null gate was later corrected without rerunning outcomes.

- Retain: causal temporal head-family screening with fair budgets, fresh blocked then
  separately locked grouped confirmation, seed/group consistency, and corrected residual
  null semantics.
- Do not transfer: the convolution's exact length, channels, gates, cap, optimizer, or seed
  count. The attached historical checkpoint is an early label-permutation control, not a
  portable winner.

### Phase 06 — stabilization branches

Single-seed Optuna looked promising but failed locked 10-seed confirmation. Robust multiseed
selection and trial 4 also failed fresh confirmation. A fixed within-seed blend failed; a
three-checkpoint trial-4 ensemble stabilized variance but lost to the original reference.
The predeclared equal average of three independently trained reference checkpoints passed:
blocked 0.266891 versus AR 0.259724/control 0.258930, and grouped 0.234368 versus AR
0.218050/control 0.217972, with 5/5 and 15/15 positive groups respectively.

- Retain: after a single recipe wins, first test a predeclared equal-weight independent-
  checkpoint ensemble; derive VEATIC checkpoint count from VEATIC variability.
- Cut by default: one-seed Optuna, outcome-selected blends, and inherited ensemble weights.

### Phase 07 — continuous specialization and joint evidence

Continuous movement required its own target-specific AR, loss, head, and confirmation. The
final grouped confirmation produced Spearman 0.260301 versus AR 0.240537/control 0.240252 and
top-5% lift 0.097598 versus AR 0.089566/control 0.089709, positive in 15/15 groups. Its
continuous predictions also ranked events at PR-AUC 0.223190 versus AR 0.208805 and strongest
control 0.209609, positive in 15/15 groups. That event result was supporting because it was
not the preregistered Phase 07 promotion endpoint; it nevertheless demonstrates coexistence
of continuous and event information. A later successful run superseded the earlier narrow
5/5 blocked-consistency miss as the final continuous conclusion.

- Retain: continuous specialization, ranking and top-tail primary endpoints, exact-value
  endpoints as a separate claim, and preservation of supporting event metrics.
- Next VEATIC hypothesis: confirm event and continuous specialists independently, then test
  a newly registered combined challenger that must preserve both locked abilities. Do not
  pretend AGAIN formally trained a single simultaneous winner.

### Zero-label programme

Stage 0 distillation and self-rollout failed and were eliminated. Direct supervised causal
video-only Stage A passed. Locked held-out-video confirmation produced Spearman 0.178513
versus strongest control 0.100488 (+77.65%), top-5% lift 0.076608 versus 0.044852 (+70.80%),
and event PR-AUC 0.171062 versus 0.135230 (+26.50%), positive in 5/5 groups with cold-start
and video-block checks.

- Retain: zero-label means labels may supervise training, but held-out inference receives no
  current/past response, teacher score, or labelled warm start. Seal predictions before
  opening confirmation labels. Direct supervised temporal video-only is the primary lane.
- Cut by default: distillation and self-rollout unless new VEATIC evidence prospectively
  reopens them.

## VEATIC execution order produced by the audit

1. Phase 01 derives targets, time scales, event support, and ownership using labels only.
2. Phase 02 iterates strong target-specific AR opponents to convergence.
3. Phase 03 tests raw cortical information and mandatory controls.
4. Phase 04 discovers fold-owned compression and causal temporal representation order.
5. Phase 05 discovers and confirms the event specialist, with AR-dominance diagnosis and
   target redesign available only when activated by evidence.
6. Phase 06 stabilizes the selected event recipe.
7. Phase 07 independently discovers and confirms continuous arousal.
8. Phase 08 independently runs the full valence ladder.
9. After event and continuous specialists pass, register and test a combined challenger.
10. Only after supervised abilities are settled, train and confirm genuine video-only,
    zero-label-at-inference lanes.
11. Preserve untouched scientific evidence models. A separate production refit may then use
    all labelled videos, and client videos receive video-only inference.

This order keeps the successful AGAIN reasoning while removing known dead ends and forcing
every numeric and architectural answer to be VEATIC 2.1 specific.
