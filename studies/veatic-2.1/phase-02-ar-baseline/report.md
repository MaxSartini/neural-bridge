# VEATIC 2.1 Phase 02 Fresh Target-Specific AR Baseline

Status: **PASS**

Phase 02 fitted a fresh VEATIC-only autoregressive event floor for the sealed continuous
future-maximum-increase target `t+1..t+6`. The common causal-history
mask retained 19,169 rows across all 124 videos.
Candidate lag depths `[0, 1, 2, 4, 6]` came only from the Phase 01 VEATIC PACF landmark and
selected target width. Every lag and ridge choice was selected by nested inner-validation raw
PR-AUC. Each outer q90, normalization, final model, and decision threshold remained owned by
its outer-training partition.

Five grouped-video 70/30 cells produced median held-out AR PR-AUC
`0.315086` (range `0.278621`–`0.383829`).
The separately reported per-video forward blocked-temporal 70/30 cell produced AR PR-AUC
`0.276250`. Fold metrics also contain prevalence/chance, AP skill, ROC-AUC,
precision, recall, F1, Brier score, top-1/5/10% recall and lift, defined-only per-video
PR-AUC, and positive counts. Paired whole-video bootstrap intervals compare AR against chance
and the training-owned strongest simple causal-history baseline.

Exact outer-test rows, event labels, continuous targets, and AR/current/slope/chance
probabilities are frozen per target/protocol/fold/seed with file and array checksums. Phase 03
must reuse the exact rows and AR predictions for every matched lane.

The target begins at `t+1` after causal history ending at `t`; history/target
overlap is zero and the boundary gap is 0 rows. The registered prospective
washout candidates remain inactive and unselected because activation requires later
control-complete development evidence. No cortical values were loaded, no PCA or bridge was
fit, and no AGAIN runtime code, data, numeric choice, seed, split, fitted object, or prediction
entered this phase. MLX ran all learned fitting and scoring on `gpu:0` in one worker process.

Code SHA-256: `48ea2c2ec687d777098882bd3f00721e715743314d080b0ab1a18fe4a8c291ef`
Prediction manifest SHA-256: `89c7c3c6444fc93e1a30e5274f93ee4d79eddbdee92802fc561b073ef47048dc`
Model manifest SHA-256: `6be0059028ff1d910cbd2c7f3f3067087b7615f71aae96fd750089d11d84e32d`
Split manifest SHA-256: `ade612dd40457918561fbbfdfa6786993df2198576d77612b05ca03b39ffeb8c`
Dominance decomposition SHA-256: `21e4e081094df6b4b2b2c3e206deae44f05d501c875e89e0a189d95cc1739595`
