# VEATIC 2.1 Neural Bridge Lifecycle Control Crosswalk

## Scope

This is the procedure-level crosswalk that must be checked before any further VEATIC 2.1
training or promotion. It maps the settled Neural Bridge lessons from phases 0–7 onto the
current VEATIC programme. AGAIN contributes control semantics and rigor only: no AGAIN
target, threshold, fitted artifact, checkpoint, architecture width, seed, score, or numeric
winner is reusable.

TRIBE and V-JEPA are upstream inputs. The claim-bearing system is the downstream Neural
Bridge target, frozen-AR floor, fold-owned projection, causal temporal residual, controls,
fallback, stability, and confirmation procedure.

## Mandatory control families

| Family | VEATIC 2.1 requirement | Evidence timing |
| --- | --- | --- |
| Dense substrate identity | Exact video inventory, row identity, time grid, feature/label alignment, quality flags, and external tree hashes must verify. No silent row repair or interpolation. | Before any fit; already enforced by the canonical substrate. |
| Ownership and leakage | Target values, event thresholds, PCA/scalers, normalization, checkpoint selection, and fallback decisions use owning training/inner-validation rows only. No future context, video-boundary crossing, global PCA, held-out fitting, or test-label checkpointing. | Before and during every cell. |
| Fresh target-specific AR | Each target/fold/seed receives its own fitted AR. The identical AR logits must be frozen beneath the real residual and every matched residual control. | Fit once per cell identity, then hash-reuse across lanes. |
| AR-only/no-video | The frozen AR lane is the primary persistence floor and the exact no-video ablation. | Every comparison cell. |
| Cortical-only/current-row | Score a video-derived lane without AR and a current-row-only cortical residual to distinguish temporal bridge value from raw/current-frame readout. | Comparison panel before promotion. |
| Full causal temporal residual | PCA current row plus causal past-difference blocks and availability indicators; no future rows. | Real lane. |
| Sequence-shuffled cortical | Permute representation identity within video while preserving row inventory and matched model capacity, then train independently. | Every comparison cell. |
| Matched random cortical | Replace the projected representation with deterministic seed-namespaced random features of matched shape, then train independently. | Every comparison cell. |
| Causal-prefix video mean | Use only the causally available prefix mean within each video; never a full-video or held-out-label mean. | Every comparison cell. |
| Diagnostics-only | Use the registered video diagnostics without cortical predictions. This subsumes time/acquisition, quality, motion, luminance, and related generic-video explanations available in the canonical 53-wide diagnostic input. | Every comparison cell. |
| Label permutation | Apply a nonzero within-video circular permutation to residual training/inner-validation labels, retain the identical frozen AR floor, restore the chosen checkpoint, and score against the true held-out inner labels. Interpret this as a residual-null over AR, not a prevalence null. | Every comparison cell. |
| Raw/direct-fusion ablation | Preserve the settled negative lesson that raw cortical readout or naïve AR-plus-raw fusion is not Neural Bridge. On VEATIC, perform only a bounded fold-safe diagnostic if an existing current artifact cannot answer it; do not import historical fitted projections. | One bounded diagnostic panel, not a numeric-selection source. |
| Evaluation mode and checkpoint replay | Restore the best eligible checkpoint, disable training-only behavior, verify checkpoint hashes, replay predictions, and never prefer the final checkpoint. | Every learned lane. |
| Seed and ensemble robustness | Use fixed seed identities. Equal-weight checkpoint groups are declared before evaluation; no member selection, weight fitting, seed deletion, or viewed-seed reuse. | After a control-passing recipe earns stability. |
| Whole-cell no-harm | Select residual versus AR from inner validation only. A failed residual falls back to AR for the entire fold/seed; no row-wise oracle is allowed. | Every real/control cell and final recipe. |
| Separate generalization questions | Grouped held-out-video and blocked-forward-time evaluations remain distinct. A diagnostic branch cannot tune, veto, or redefine another branch after viewing it. | Before confirmation. |
| Metric completeness | Primary pooled average-precision skill plus pooled PR-AUC, analytic chance, per-video defined-only PR-AUC, top-1/5/10% event recall, Brier score, paired video-cluster bootstrap uncertainty, zero-event accounting, and fold/seed consistency. | Every completed control summary. |
| Artifact and access audit | Request/config/code hashes, row digests, fitted-artifact hashes, label-access events, partial-run quarantine, prediction sealing, and exact matrix coverage must verify. | Every plan, cell, summary, and confirmation. |

## Promotion gates

A VEATIC spike candidate may proceed from comparison to stability only if all of the following
are true on the complete comparison panel:

1. The real residual improves on its identical fresh AR floor in aggregate.
2. The real residual improves on the strongest aggregate matched control.
3. Paired medians versus AR and the strongest control are positive.
4. Improvement is not concentrated in one target, fold, seed, or checkpoint group.
5. The label-permutation residual adds no material signal beyond the identical AR floor.
6. Frozen-AR identity, fold-safe PCA, causal context, row alignment, eval mode, checkpoint
   replay, checksum, and label-access audits all pass.
7. Whole-fold/seed fallback is applied without a row-level oracle.
8. Every failed gate remains recorded; a failure stops stability or confirmation.

Exact numeric margins must be calculated and registered from VEATIC 2.1 evidence before the
control run. They must not be copied from AGAIN.

## Reuse map for current artifacts

- The 90 completed PCA-512 causal comparison cells provide the real lane, fresh-AR scores,
  validation row identities, targets, model scores, selected scores, checkpoints, and
  preprocessing artifacts.
- The 82 stopped width-128 cells are exploratory width evidence only; they are not a control
  lane and not a completed recipe matrix.
- Completed stability cells are reusable only after the comparison control gate passes. They
  must not be retrained merely because controls were executed late.
- Control lanes must reuse the exact comparison row ownership, threshold, PCA basis, seed,
  recipe, and frozen-AR floor. Only the declared controlled factor may change.
- The sealed benchmark tail remains inaccessible until one recipe, target, fallback policy,
  checkpoint ensemble, control verdict, and prediction set are frozen.

## Required execution order from here

1. Stop stability at its resumable boundary. **Done.**
2. Register the complete VEATIC-owned comparison control matrix and VEATIC-calculated gates.
3. Backfill matched controls against the existing 90 real comparison cells with one MLX
   worker, reusing their frozen AR and fold-owned PCA artifacts where exact identity verifies.
4. If any mandatory control gate fails, reject or redesign before spending more stability
   compute. Preserve the failure.
5. If all gates pass, resume the already-saved stability panel, then form only the
   preregistered equal-weight checkpoint groups.
6. Freeze one fully controlled inner-validation winner before any sealed-tail access.
