# Study Journey

Neural Bridge advanced through ten connected research stages. Each stage had a distinct question, evidence standard, failure condition, and reason to continue. Negative results are retained because they explain why the final system exists; terminal metrics are not presented as if they appeared from a single model search.

Current shared code lives in [`src/neural_bridge/`](../src/neural_bridge/). Historical phase scripts are provenance, not the active API. Each linked closure preserves compact evidence and identifies any heavy externally registered artifacts.

The fresh [VEATIC 2.1 rebuild](veatic-2.1/README.md) is active after its Phase 01 alignment
gate. Phase 00 passed 27/27 substrate controls and Phase 01 passed 28/28 label/target controls
over all 124 videos and 20,657 rows. Its next step is a comprehensive fresh AR search over
all 21 active no-washout target candidates; it has no selected target or modeling result yet.

| Stage | Question | Decisive result | Scientific consequence |
| --- | --- | --- | --- |
| [Original VEATIC](original-veatic/v2-closure/README.md) | Can short causal temporal change improve future event/spike ranking? | blocked PR-AUC `0.2536` vs AR `0.1969`, shuffled `0.1840`, and random `0.1944`; balanced event-vs-stable `0.3394` | established the event-first hypothesis; contributed no fitted artifact or exact recipe to later datasets |
| [AGAIN Phase 0](again/phase-00-dense-foundation/README.md) | Can a complete dense 2 Hz substrate be built and audited? | `995/995` videos, `243,575` rows, explicit timestamps, quality flags, and an empty failure ledger | made every later row, mask, and feature auditable |
| [Phase 1](again/phase-01-label-alignment/README.md) | Can continuous labels, future targets, and eligibility be aligned without invented rows or test-owned thresholds? | `243,441` labeled rows; `134` unmatched rows and `4,153` insufficient-history rows retained explicitly | established the supervised table and fold-owned target contract |
| [Phase 2](again/phase-02-ar-baseline/README.md) | How strong is learned response persistence on each target and protocol? | final blocked/grouped target-specific AR after four materially different development revisions | created the hard frozen reference every later real and control lane had to beat |
| [Phase 3](again/phase-03-raw-cortical/README.md) | Are fixed summaries of 20,484 predicted cortical vertices already useful? | raw features were target-dependent and often weaker than AR; direct fusion helped spike ranking but hurt short-delta ranking | rejected “rich upstream features are automatically useful” and motivated representation learning |
| [Phase 4](again/phase-04-pca-bridge/README.md) | Can train-fold-fitted compression and temporal aggregation improve the raw representation safely? | grouped spike PR-AUC `0.1716` vs AR `0.1473` and direct fusion `0.1703` | validated fold-safe fitting and temporal representation, while exposing the limits of fixed PCA |
| [Phase 5/5.5](again/phase-05-learned-bridge/README.md) | Can a learned residual bridge and event-specific temporal head survive matched controls? | raw `0.1366` → residual bridge `0.2383`; first redesigned target rejected; selected head passed blocked and grouped confirmation | solved event ranking under the stronger AGAIN system and established the reusable learned bridge |
| [Phase 6](again/phase-06-event-stabilization/README.md) | Which stabilization method survives fresh confirmation? | Optuna single-seed and two blends rejected; declared checkpoint ensemble reached `0.2344` vs AR `0.2180` (**`+7.48%`**), positive `15/15` | turned the event win into a repeatable prospective procedure |
| [Phase 7](again/phase-07-continuous/README.md) | Can the bridge rank continuous future movement and its highest tail? | grouped Spearman `0.2603` vs `0.2405` (**`+8.22%`**); top-5% lift `0.0976` vs `0.0896` (**`+8.97%`**), positive `15/15` | extended Neural Bridge beyond binary events into continuous response intelligence |
| [Zero-label-at-inference](again/zero-label/README.md) | Does useful signal survive without observed arousal or response history at inference? | distillation and self-rollout rejected; frozen direct-temporal system passed three endpoints on 299 untouched videos | established the latest locked video-only result without reopening the confirmation pool |

## How to read the record

- **Discovery** identifies promising targets, representations, or heads; it does not create a final claim.
- **Controls** test whether alignment, video content, diagnostics, or labels are actually responsible for the apparent signal.
- **Promotion** occurs only after the declared comparison survives its development gates.
- **Confirmation** uses fresh or locked evidence and cannot silently become another tuning pass.
- **Blocked-temporal** and **held-out-video** protocols remain distinct because they answer different generalization questions.

[Return to the concluded scorecard](../results/README.md) · [Read the methods and reproducibility contract](../docs/README.md)
