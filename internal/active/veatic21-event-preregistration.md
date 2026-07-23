# VEATIC 2.1 Spike Discovery Protocol

## Objective

Earn a VEATIC-specific label-assisted arousal-spike model that adds stable, meaningful
video-derived information over freshly fitted VEATIC autoregression. This stage does not
select continuous, valence, zero-label, or production-generalist models.

## Canonical input

- Dataset: the 124 VEATIC videos at 2 Hz.
- Neural Bridge input: `cortical_prediction` from
  `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/tribe-v2/compact-20260716/per_video/<video_id>/tribe_v2_cortical_predictions.npz`.
- TRIBE artifact: `veatic-2.1-tribe-v2-compact-20260716`.
- TRIBE tree SHA-256:
  `0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd`.
- V-JEPA 2.1 is the encoder inside the upstream TRIBE v2 cache-generation stack. It is
  not a competing Neural Bridge input.
- Canonical rows: 20,657; excluded black/static/end-screen rows: 923; usable rows: 19,734.

## Ownership and label access

- The first eligible 70% of every video is the 13,753-row development region.
- The last eligible 30% is the sealed 5,981-row benchmark tail.
- All target calibration, thresholds, AR fitting, PCA selection, learned training,
  checkpointing, controls, and winner selection use development ownership only.
- The sealed tail opens once, only after one complete spike recipe is frozen.
- Production refit starts from scratch on all usable rows after benchmark confirmation and
  is not benchmark evidence.

## Method inherited from the complete AGAIN chain

The transferable method is the complete Phase 0–4–5/5.5 sequence, not a Phase 5 head in
isolation:

1. Verify dense substrate identity, alignment, exclusions, row ownership, and label support.
2. Derive targets and thresholds from owning training labels only.
3. Fit a strong fresh target-specific AR comparator inside every target/fold/seed.
4. Fit label-blind PCA only on the owning rows and reuse the fitted maximum-width basis by
   prefix; candidate widths are `64`, `128`, `256`, and `512`.
5. Train VEATIC-specific causal temporal residual heads over the frozen AR logits and
   cortical projection.
6. Compare identical folds, seeds, and training ownership; then expand only earned
   candidates to stability seeds.
7. Run matched false-signal controls, leakage checks, and whole-fold/seed no-harm fallback.
8. Freeze one inner-validation winner, seal its predictions and dependencies, then open the
   benchmark tail once.

AGAIN contributes this procedure and its failure exclusions. No AGAIN target, threshold,
PCA, scaler, head, checkpoint, weight, gate, score, or numeric winner is reusable here.

## Active discovery stages

1. **Full target and fresh-AR discovery** — evaluate all 90 calibrated target hypotheses
   across five grouped-video folds and comparison seeds `20260722`, `20260723`, and
   `20260724`. Inner AR regularization and inner event thresholds are fitted within their
   owning training panels.
2. **PCA and representation discovery** — compare `64/128/256/512` fold-owned projections
   and VEATIC-calculated learned alternatives on the target shortlist produced by stage 1.
3. **Model and training discovery** — compare the causal temporal residual and gated
   multiscale temporal residual families with a VEATIC-owned registered matrix.
4. **Stability** — rerun the shortlist across the fixed stability panel without changing
   folds, seed identities, target definitions, or candidate semantics.
5. **Controls and no-harm** — require improvement over fresh frozen AR and matched
   sequence-shuffled, random, causal-prefix video-mean, diagnostics-only, and circular-label
   controls. Any AR fallback is selected for the whole fold/seed from inner validation,
   never from row outcomes.
6. **Winner freeze and sealed confirmation** — freeze one recipe from inner evidence,
   create predictions before labels open, then evaluate the sealed tail once.

## Checkpoint contract

- Every completed validation checkpoint from epoch 1 is merit-eligible.
- Training cannot terminate before epoch 50.
- There is no fixed maximum epoch; genuine improvement may continue beyond 400 epochs.
- Stop after the registered validation plateau only when optimizer loss has stabilized.
- Fail a non-converged candidate after 400 consecutive non-improving validations without
  optimizer convergence; this is not an absolute epoch ceiling.
- Exact metric ties keep the earlier checkpoint. The final checkpoint receives no
  preference.

## Current artifacts

- Preregistration:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/event-spike-v1.json`
- Train-only target calibration:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/event-spike-v1-calibration.json`
- PCA manifest and fold payloads:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/features/veatic-2.1/neural-bridge/cortical-pca-v1/manifest.json`
- Current replace-in-place child plan:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/stage1-child-plan.json`
- Non-promotable executor validation:
  `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/stage1-executor-validation/cell-001`

## Next command

```bash
uv run python -m neural_bridge.veatic21 benchmark-stage1-ar
```

This command is resumable by target. After it completes, rerun:

```bash
uv run python -m neural_bridge.veatic21 prepare-stage1
```

That replaces `stage1-child-plan.json` with a `spike_discovery` plan bound to the complete
AR summary. It does not open the sealed tail or select a winner.

## Required order after spike

Continuous arousal begins only after spike confirmation; valence begins after continuous;
VEATIC zero-label-at-inference begins after those label-assisted abilities. Confirmed
VEATIC abilities then join AGAIN and future dataset abilities in the cumulative production
generalist for unseen client video.
