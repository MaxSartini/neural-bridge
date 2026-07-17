# Neural Bridge Roadmap

Phase 7 remains the AR-assisted performance frontier. The first cached-feature video-only deployment bridge is now confirmed on a prospectively locked 299-video AGAIN pool. The project has moved from “can the signal survive without response labels at inference?” to “can we run it end to end from raw video and confirm it beyond AGAIN?”

## Current Deployment Win: Locked Zero-Label Bridge

The fixed direct-supervised temporal lane passed `140/140` locked rows, beat the strongest false-signal/no-video controls by `+77.65%` Spearman, `+70.80%` top-5% lift, and `+26.50%` event PR-AUC, and won all `5/5` full-video panels on every endpoint.

This establishes cached-feature zero-label inference on AGAIN. Raw-video end-to-end runtime and external/client validity remain separate milestones.

## Research Ceiling: Phase 7

The fresh grouped held-out-video continuous confirmation passed `420/420` rows with failed gates `[]`:

- real / AR / best-control Spearman: `0.2603011121` / `0.2405371348` / `0.2402523335`;
- delta versus AR / best control: `+0.0197639773` / `+0.0200487786`;
- real / AR / best-control top-5% lift: `0.0975979581` / `0.0895663763` / `0.0897088493`;
- delta versus AR / best control: `+0.0080315818` / `+0.0078891089`;
- wins versus both comparators on both metrics: `15/15` fold-groups;
- positive fold means: `5/5`;
- checkpoint-ensemble uplift: `+0.0077966938` Spearman and `+0.0025021192` top-5%.

This proves controlled grouped continuous future-arousal movement ranking/lift for the selected washout target/head.

## What Has Been Established

- VEATIC established the original future arousal event-ranking signal on edited affective media.
- AGAIN replicated and strengthened the result at scale on `995` cleaned gameplay videos.
- Raw predicted cortical/fMRI features alone are insufficient; the Neural Bridge design creates the usable signal.
- The selected binary target/head passes blocked and grouped controls, the unified selected-head `420/420` audit, and fresh Phase 6 checkpoint-ensemble confirmations.
- The Phase 7 washout continuous target/head passes fresh grouped continuous confirmation against target-specific AR and full matched controls.
- Fixed checkpoint averaging is a real stabilizer, not a cosmetic change.
- The effect is not dependent on a single favorable seed or fold.

## Milestone 1: End-to-End Raw-Video Runtime

- freeze the confirmed prediction recipe;
- run upstream feature generation plus the video-only bridge from a raw client-style file;
- measure Apple Silicon latency, memory, failure handling, and deterministic provenance;
- produce continuous heat maps and event bands without false exact-value claims.

## Milestone 2: Cross-Domain Training Pilot

VEATIC currently uses an older, less comprehensive encoding setup than AGAIN. A V-JEPA 2.1 re-encode and balanced joint training may improve stability and domain transfer, especially for the video-only student.

Do not begin with a full expensive re-encode. First run a bounded pilot that answers whether the investment is justified:

- harmonize future windows, row rate, labels, masks, and target definitions;
- use domain-balanced sampling so AGAIN does not swamp VEATIC;
- use a shared trunk with domain-specific calibration heads if needed;
- include leave-VEATIC-out and leave-AGAIN-out evaluation;
- compare joint training against domain-specific Phase 7-quality baselines;
- preregister the minimum useful gain and stability criteria.

If the pilot transfers positively, scale the VEATIC 2.1 re-encode. If not, retain VEATIC as independent external validation rather than forcing joint training.

## Milestone 3: Product Evaluation Surface

Now that the video-only bridge has passed:

- convert ranked future movement into response heat maps and event bands;
- expose confidence and calibration rather than false precision;
- support cut-versus-cut comparison and weak-segment diagnostics;
- generate a response-readiness report with traceable model/evidence metadata;
- validate on prospective client-style videos before commercial accuracy claims.

## Research Rules

- Phase 7 is the current headline; Phase 5.5 and Phase 6 are its evidence foundation.
- Do not resume broad same-family Optuna tuning; it has already been tested and bounded.
- Do not delete or hide difficult seeds.
- Do not combine ranking/lift and exact-value claims after the fact.
- Do not launch all-target or architecture-zoo sweeps without a narrow preregistered hypothesis.
- Preserve matched AR and controls; beating weak controls alone is not enough.
- Never tune or select again on the locked 299-video pool.

## Repository Rules

- Keep current authority in `README.md`, `docs/neural_bridge_phase7_evidence.md`, `docs/current_project_state.md`, and `docs/current_claim_status.json`.
- Preserve historical reports and evidence snapshots as the scientific record.
- Track lightweight claim-bearing reports, manifests, and checksum anchors.
- Keep dense caches, tensors, model weights, checkpoints, and heavy generated outputs outside git.
- Sync the unified CodeGraph and refresh the compact Context-Mode handoff index only after canonical files are validated and the repository is clean.
