# VEATIC 2.1 Event/Spike Discovery

Status: official per-video 70/30 calibration complete; future-tail labels sealed.

## Product objective

Neural Bridge is one production video model, not one model per dataset. A client video is
canonically sampled and encoded by the single V-JEPA 2.1-inside-TRIBE v2 stack, then scored
by one shared temporal head for spike, continuous arousal, valence, and confidence. VEATIC
2.1, AGAIN, and future datasets broaden the skills and domains learned by that head.

VEATIC 2.1 adds film, television, documentary, home-video, character, context, and valence
coverage. AGAIN adds gameplay and already-confirmed spike/continuous coverage. Dataset
specialists are discovery and verification instruments. The production model will use a
fresh joint projection and head trained across compatible dataset training pools, with no
dataset identifier and no response labels at inference. Unseen ads are the first external
product gate, not a separate product mode.

## Scientific boundary

Original VEATIC uses the same 124-video affect dataset, so its label semantics and failures
are relevant same-dataset evidence. Its old upstream features, PCA, scalers, thresholds,
heads, and scores are obsolete. AGAIN contributes cross-dataset methods and failure modes.
Every VEATIC 2.1 fitted object comes from the current VEATIC V-JEPA/TRIBE caches and labels.
No model family is the VEATIC winner yet: names in the preregistration are candidates that
VEATIC inner validation may select or reject. AGAIN contributes the discovery algorithm,
not its discovered head, input, target, dimensions, gates, loss, or numerical configuration.

## Dataset semantics from the primary sources

VEATIC contains 124 silent visual clips: 104 Hollywood, 15 home-video, and 5 documentary
or reality-TV clips, lasting 10 seconds to 2 minutes 37 seconds. Observers continuously
rated the selected character on a bounded two-dimensional valence/arousal grid; released
traces are consensus ratings. The task is therefore third-person, target-character affect
from spatial and temporal visual context. Its published benchmark uses the first 70% of
frames from every video for training and the last 30% for testing.

AGAIN's clean set contains 995 roughly two-minute gameplay sessions from 122 participants,
nine games, and three genres. Each player replayed and continuously annotated their own
experience using unbounded ordinal RankTrace arousal; the clean traces are normalized per
session. It is first-person experienced arousal, not observer-rated character affect, and it
has no valence label.

Consequently, relative spike/change supervision is the most directly compatible cross-
dataset bridge. Absolute continuous arousal scales are not assumed interchangeable: a
future shared model must learn and validate their alignment with dataset-aware training
losses, while remaining dataset-agnostic at client inference. VEATIC alone supervises
valence until another compatible valence dataset is added.

Primary sources: [VEATIC WACV paper](https://whitneylab.berkeley.edu/PDFs/Ren_WACV_2024.pdf),
[AGAIN dataset site](https://again.institutedigitalgames.com/), and
[AGAIN paper](https://arxiv.org/pdf/2104.02643).

Dataset-specific PCA is permitted only as fold-fitted discovery evidence. After benchmarking,
the fixed VEATIC specialist recipe is refitted from scratch on all 124 videos. The final
generalist then gets a fresh learned projection or joint PCA fitted on every compatible
VEATIC and AGAIN training video; incompatible fitted PCAs are never blended.

Within a single exact fold and representation, one maximum 512-component VEATIC PCA basis
is fitted once. Its 64, 128, 256, and 512 prefixes, plus available prefixes reaching the
declared VEATIC variance targets, are reused across targets, quantiles, heads, and fixed
seeds. Explained variance proposes candidates; VEATIC inner validation selects the width.
The cache key includes substrate, quality mask, split, training-row identities, representation,
transform, scaler, solver, configuration, and numerical-implementation version. Unrelated
head or checkpoint changes do not invalidate PCA; any mathematical ownership change does.
Thus a changed PCA key forces a fresh fit, while
inner-fold, benchmark-train, and final all-row PCA scopes are never interchanged.

Fixed PCA is the reusable label-blind baseline, not the model ceiling. A supervised learned
bottleneck from the cortical TRIBE representation is trained with labels inside each exact
training fold and compared on the same validation videos and seeds. Benchmark-tail labels
never fit either projection.

## Label access and split

The label-blind temporal split is frozen at
`artifacts/preregistrations/veatic-2.1/event-spike-v1.json` with digest
`53b960a1a13629a32a017f6ebeefbc478708e6c0d62ce76f60659b036e6f76d3`.

- All 124 videos contribute their first 70% of usable rows to calibration, supervised
  training, grouped inner validation, and model selection: 13,753 rows.
- All 124 videos contribute their last 30% of usable rows to the sealed future-tail
  benchmark: 5,981 rows.
- The 923 black/high-duplicate rows are removed before each video's 70/30 boundary is
  calculated. They cannot enter PCA, heads, target windows, or scoring.
- After the benchmark, its weights are discarded and the frozen recipe refits fresh PCA,
  scalers, thresholds, and head on all 19,734 usable rows. That refit produces no benchmark
  claim.

Training is fully supervised. Only inference and the benchmark-test firewall are
label-free.

VEATIC's 124 videos are small beside AGAIN's 995. Rows are therefore never treated as 19,734
independent samples: selection and uncertainty are grouped by video, the search space is
preregistered and constrained, and benchmark-test results cannot trigger repair or tuning.
The all-124 refit recovers scarce training signal; larger future film, television, and ad
datasets are still required to improve coverage and external certainty.

## VEATIC-derived targets

First-70%-row calibration is sealed at
`artifacts/preregistrations/veatic-2.1/event-spike-v1-calibration.json` with digest
`f9cd786a5449deeefdef325966144fd7cec2bd6ea449dc8144656ccb31463b1b`.

- Current 2 Hz VEATIC labels independently produced movement milestones at 1, 3, and 5 s.
- Candidate future-increase windows are 0.5--1 s, 0.5--3 s, and 1--5 s with washout.
- Every supported q80--q95 event rate is retained: 18 target candidates covering 5--20%
  prevalence. No AGAIN or original-VEATIC threshold was reused.
- All targets retain both classes and sufficient positive-video support in all five
  benchmark-train folds.
- Target and rarity will be chosen from supervised incremental validation, then frozen
  before benchmark test.

The same-dataset original VEATIC event rate makes q85--q90 especially worth testing, but it
does not select it. The similar 1/3/5-second scale is an independent calculation from the
new benchmark-train scope, not an inherited window.

## Discovery and promotion

- Recalculate a strong current-plus-history VEATIC AR floor for every target and fold.
  Phase one is label-assisted: a fresh learned video residual must demonstrate information
  beyond that AR floor. It need not also solve label-free inference immediately.
- The safe residual candidate is `frozen AR + bounded learned video correction`. Within
  each fold/seed, the correction is used only when its inner-validation delta over AR is
  positive; otherwise that run emits unchanged frozen AR and contributes no residual-win
  evidence. This is whole-run validation selection, never an impossible per-row switch based
  on observed error. Gate/bound values and slice no-harm margins are VEATIC-calculated.
- Only after the label-assisted target, representation, and head family freeze does phase
  two train fresh VEATIC zero-label candidates. Production inference remains video-only.
  Both configurations freeze before the future-tail labels are opened once.
- The canonical Neural Bridge input is the cortical prediction from the V-JEPA 2.1-inside-
  TRIBE v2 stack. Cached V-JEPA and grouped TRIBE views are internal stack ablations only,
  not independent sources or production fusion branches.
- Raw/linear lanes are diagnostics. Stage-one Neural Bridge candidates add fresh causal or
  gated multiscale video residuals to frozen VEATIC AR predictions. Matched direct-supervised
  current-row and temporal video-only heads remain baselines until zero-label conversion.
- Fit all transforms on the current training fold. PCA widths are calculated from VEATIC
  explained variance; no fitted PCA from either historical programme is loadable.
- Controls are no-video, diagnostics-only, within-video sequence shuffle, matched random,
  label permutation, current-row, and the recalculated AR ceiling.
- Select using prevalence-normalized average-precision skill, continuous movement ranking,
  top-tail lift, paired video-cluster uncertainty, and fold/seed consistency. Numerical
  gates are calculated from VEATIC benchmark-train variance, never copied from AGAIN metrics.
- The objective is useful affect information, not an impossible exact-decimal forecast of a
  noisy consensus trace. Spike timing/ranking, direction, meaningful magnitude bands,
  continuous association, tail retrieval, and calibration drive promotion; point error is a
  reported guardrail and cannot let a bland mean predictor win.
- Every candidate uses the same fixed three comparison seeds, folds, and sampler orders so
  improvement is paired rather than confounded by run luck. A separate fixed nine-seed
  stability panel is opened only after the winner freezes and can never drive tuning.
- Every checkpoint from epoch 1 is eligible to win, but no run may terminate before epoch
  50. Training continues beyond 50—including 400+ epochs when useful—until the frozen
  VEATIC plateau/convergence rule fires. The best validation checkpoint over the complete
  trajectory wins; the last checkpoint has no preference and the epoch ceiling is only a
  generous runaway-compute guard.
- Keep held-out-video, blocked-temporal, cold-start, and eventual external-ad claims
  separate.

The superseded 99-video compact internal-stack ablation covered 2,880 cells. Fresh PCA widths were
182--191 for the cached V-JEPA view and 215--224 for grouped TRIBE; every compact linear
lane lost to AR. It remains Phase-3-style failure evidence but is not eligible for selection
under the corrected 70/30 benchmark. Repeating it would only rerun a known diagnostic dead
end; cortical TRIBE advances through Neural Bridge heads.

The reusable cortical PCA cache is now complete at
`artifacts/features/veatic-2.1/neural-bridge/cortical-pca-v1` with manifest digest
`f2eaf69965e26a43114f016767901be74c22f59b16351a05c914330133a61739`.
Across the five exact inner folds, the 80/90/95% variance widths are consistently 8/20/57--59;
the 99% width is 179--189, and 512 components retain 99.978--99.979%. Predictive selection
still compares the fixed 64/128/256/512 candidates and variance-derived prefixes; variance
alone does not choose the winner. All five heavy bases and projections were verified as
cache hits after an unrelated preregistration change, proving correct reuse ownership.

The historical phase ladder is retained because it found the breakthroughs efficiently:
Phase 0 locks substrate, masks, provenance, and leakage barriers; Phase 1 derives targets
from the current labels; Phase 2 fits strong task-specific AR floors; Phase 3 cheaply rejects
raw/linear dead ends; Phase 4 calculates fold-safe temporal representations and PCA; Phase 5
tests fresh learned label-assisted residual families; Phase 5.5 uses matched controls and
rejects fragile targets; Phase 6 checks fixed-seed stability and predeclared ensembles; and
Phase 7 confirms ranking and top-tail utility. The later zero-label programme starts only
from the frozen label-assisted discovery, while treating AGAIN's failed distillation and
self-rollout attempts as warnings rather than defaults.

## Exact next action

Read target support directly from `event-spike-v1-calibration.json` and PCA evidence directly
from `cortical-pca-v1/manifest.json`. No neural-memory artifact exists; run one bounded local
capacity/batch probe instead of searching for one, then seal the child training plan. Use the cached projections
to screen the 18 VEATIC targets with fresh label-assisted residual families and matched
video-only/control lanes. Freeze the stable target, representation, and label-assisted head
before beginning zero-label conversion; keep every last-30% label closed until both stages
freeze. Apply the same VEATIC-calculated method—not fitted artifacts—to continuous arousal
and valence, sharing immutable caches where their ownership is identical. After the single
benchmark opening, refit the frozen VEATIC recipe on all usable rows, then train the fresh
VEATIC+AGAIN generalist.
