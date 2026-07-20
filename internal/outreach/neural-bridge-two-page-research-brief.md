# Neural Bridge: Forecasting Future Human Arousal from Video-Derived Predicted Neural Response

*Research brief — Max Sartini*

[Download the typeset two-page PDF](./Neural-Bridge-Research-Brief-Max-Sartini.pdf)

## Research question

Can video-derived predicted cortical-response dynamics forecast how human arousal will move seconds into the future—beyond recent behaviour, generic video information, and matched false-signal controls—and retain that signal when no audience-response labels are available at inference?

## Results

Yes. Across the peer-reviewed WACV 2024 VEATIC benchmark and IEEE Transactions on Affective Computing’s AGAIN dataset, Neural Bridge repeatedly converted weak raw predicted-cortical features into strong, controlled future-response signal, culminating in a prospectively locked 299-video result with no observed arousal at inference.

**How to read the evidence.** Precision–recall area under the curve (PR-AUC) measures rare-event ranking without rewarding the common non-event class; autoregression (AR) is a trained persistence model based on recent arousal.

**1 — The bridge creates the value.** On matched AGAIN events, raw predicted-cortical features reached **0.1366 PR-AUC**, AR reached **0.1473**, and direct fusion reached **0.1703**. The learned temporal residual bridge reached **0.2383**: **+74.51% over raw**, **+61.86% over AR**, and **+39.95% over direct fusion**. Under blocked-time evaluation it reached **0.2671**: **2.15× raw** and **+31.16% over AR**, while direct fusion actually hurt AR. Rich upstream features were not the result; the bridge made them useful.

**2 — The result repeats across datasets and endpoints.** On VEATIC’s 124 contextual clips, blocked future-event PR-AUC was **0.2536** versus trained AR at **0.1969** (**+28.80%**), shuffled features at **0.1840**, and random features at **0.1944**. AGAIN Phase 7 then completed **420/420** declared grouped comparison cells and beat AR and matched controls in **15/15** fold-checkpoint groups: Spearman was **0.2603** versus **0.2405**, with top-5% movement lift at **0.0976** versus **0.0896**.

**3 — It survives the hardest deployment condition tested.** One video-only model was selected on 696 development videos, frozen, and evaluated once on 299 untouched videos. It completed **140/140** rows, won all endpoints in **5/5** frozen panels, and achieved **0.1785 versus 0.1005 Spearman (+77.65%)**, **+70.80%** top-5% lift, and **+26.50%** event PR-AUC over the strongest matched false-signal or no-video controls. Every paired whole-video bootstrap lower 95% bound was positive, including the first-30-second cold-start tier.

## Motivation

Forecasting how content will move people before audience feedback exists would turn response measurement from a post-hoc report into a design instrument. Today, creators, educators, media teams, and interactive systems either optimise semantic proxies or wait for panels, surveys, biometrics, and released-product behaviour. Those signals arrive after many consequential editing, selection, pacing, and timing decisions. Neural Bridge targets pre-response intelligence: identifying moments likely to precede strong human movement while the content can still be changed.

Ordinary video understanding does not settle this question. Recognising objects, actions, dialogue, or genre is not the same as representing how a stimulus drives a human response through time. Neural Bridge instead uses predicted cortical/fMRI response features generated from video by upstream models trained on brain-response data. They are video-side predictions, not neural recordings from benchmark viewers, and they provide a response-shaped intermediate representation between pixels and aggregate affect.

The benchmark is difficult for structural reasons. Arousal is strongly autocorrelated, rare events invite base-rate shortcuts, and random row splits leak the same video’s identity and temporal structure. A model that merely repeats recent arousal can look impressive. The decisive question is therefore whether predicted neural-response dynamics still add signal after trained persistence, held-out videos, train-owned fitting, future-only targets, and controls that preserve plausible inputs while destroying the claimed mechanism. Neural Bridge survives that test across two highly contrasting affect datasets.

## Methods

I designed Neural Bridge’s evaluation so a superficial model could not pass. Frozen upstream encoders produced video features and predicted cortical-response trajectories; the claimed contribution is the downstream target design, causal temporal bridge, train-owned residualisation, control system, and validation discipline that converts those trajectories into future-response rankings.

Every real lane faced a target- and protocol-specific frozen autoregressive floor plus matched alternatives: raw cortical features, direct AR-plus-raw fusion, current-row video, no-video timing and masks, shuffled temporal sequences, random features, generic diagnostics, video means, and label permutation. These controls ask whether performance comes from persistence, dimensionality, timestamps, video identity, label prevalence, or accidental alignment. Real and control lanes shared folds, seeds, masks, and ensemble rules.

Evaluation separated videos between training and testing. PCA, normalisation, thresholds, autoregressive models, heads, and selection belonged to the training split. Future-event thresholds were train-owned; the continuous AGAIN target measured the largest arousal increase two to five seconds ahead at 2 Hz. Discovery, blocked-time tests, grouped confirmation, and the prospectively locked evaluation remained separate rather than being pooled into a favourable score.

The final zero-label-at-inference model was trained with labelled development data but received no observed arousal, response history, teacher score, or labelled warm start on held-out videos. Multiple seeds, frozen panels, prediction seals, audit contracts, and paired whole-video bootstraps tested repeatability at the unit that must generalise: the video.

## Limitations

Neural Bridge makes a precise claim, not a small one: it forecasts aggregate future arousal ranking and event structure in VEATIC and AGAIN. It does not claim causal neural identification, exact individual trajectories, clinical inference, mind reading, or universal emotion recognition. The neural-response features are predictions generated from video, not direct viewer recordings.

The zero-label result is supervised learning with label-free inference. The two datasets span very different content, annotation, and evaluation regimes, so their convergent event evidence is stronger than a single-domain result, while continuous and locked video-only magnitudes remain protocol-specific. Grouped video separation, hard controls, prospective locking, prediction seals, and whole-video bootstraps address major internal failure modes; independent external replication is the appropriate test of the claim’s broader reach.
