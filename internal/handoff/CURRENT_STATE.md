# Canonical Current State

Updated: 2026-07-23

## Confirmed

- VEATIC 2.1 contains 124 videos and 20,657 dense 2 Hz rows. The quality mask retains 19,734 rows: 13,753 development rows and a sealed 5,981-row last-30% tail.
- Train-only calibration retains all 90 arousal-spike targets. No target, representation, PCA width, learned head, or training recipe has won.
- Five fold-owned VEATIC cortical PCA bases are cached through 512 dimensions. This is reusable preparation only: no PCA width has been benchmarked or selected.
- The event screen at `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/event-target-screen/tribe-cortical-tribe-grouped-mean-vjepa-temporal-mean.json` is complete: 21,600 unique cells covering 90 targets, 3 sources, 5 folds, 4 causal forms, and 4 ridge strengths, with fresh matched AR and no sealed-label access.
- Its complete viability summary is `/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/event-target-screen/viability-summary.json`. All 270 target/source families are retained; zero raw/linear families have positive mean skill delta versus fresh AR. This is a preliminary diagnostic only, not learned-model discovery, and it does not test either Neural Bridge residual head.
- `/Volumes/onn. Drive/Neural Bridge Artifacts/preregistrations/veatic-2.1/stage1-child-plan.json` binds all 90 targets, all 270 target/source families, all fold-owned PCA widths, both registered label-assisted head families, matched folds, and matched seeds. It is a plan, not a trained result.
- No learned Stage-1 residual-head experiment, PCA-width benchmark, model comparison, training-method comparison, fold/seed confirmation, control/no-harm gate, or winner selection has run. The existing `src/neural_bridge/veatic21/runner.py` is linear confirmation plumbing and must not be represented as the Stage-1 trainer.

## Fixed constraints

- Crack arousal spike first; continuous arousal, valence, and zero-label follow only after their preceding gate is genuinely won.
- Use VEATIC-specific targets, projections, models, controls, and numeric choices. AGAIN supplies hypotheses and rigor, never fitted objects or copied numbers.
- Every checkpoint from epoch 1 is merit-eligible. Train at least 50 epochs before termination, continue while validation can improve, impose no epoch ceiling, and break exact metric ties toward the earlier checkpoint.
- Keep folds and seeds matched across candidates. Require incremental value over fresh frozen AR, controls, fold/seed stability, leakage checks, and no-harm. Any AR fallback is selected for a whole fold/seed from inner validation, never per row.
- Keep the sealed tail closed until one complete spike recipe is frozen.

## Exact next action

Implement and test the actual learned Stage-1 discovery-cell executor in `/Users/maxsartini/Neural Bridge/src/neural_bridge/veatic21/stage1.py`. It must consume the current child plan and fold-owned feature/PCA artifacts, fit fresh fold-owned AR, train both registered bounded causal residual-head families with matched folds and seeds, apply the checkpoint contract exactly, write resumable scientific metrics, and never open the sealed tail. Then run one bounded non-claiming cell to validate the executor before expanding the full discovery matrix.

Protocol: `/Users/maxsartini/Neural Bridge/internal/active/veatic21-event-preregistration.md`.
