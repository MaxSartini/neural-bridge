# VEATIC 2.1 Phase 05 Learned Frozen-AR Bridge

Status: **PASS**

Phase 05 evaluated a fresh VEATIC-derived residual-head family on the exact sealed width-64,
current-row Phase 04 representation. Every real/control head received the identical matching
frozen AR logit as an immutable floor. Checkpoints were selected only on inner validation,
restored, and scored in deterministic eval mode. Residuals without strictly positive inner
raw PR-AUC value were suppressed to bit-exact AR predictions.

The globally selected recipe was `relu-bottleneck-8` with hidden width
`8`. Grouped median PR-AUC was `0.315086` for
frozen AR and `0.315152` for the no-washout real residual. Blocked
PR-AUC was `0.276250` and `0.276250`, respectively. The controlled
no-washout residual claim **FAILED**.

Legal persistence dominance: **True**. Phase 06 authorization:
**False**. VEATIC washout-design authorization: **True**.
No washout target was constructed or scored in this phase.

The matrix retained the exact Phase 04 PCA-only companion plus shuffled, random, train-only
video mean, diagnostics, time/video-time, quality/motion/luma, and label-permutation residual
controls. Exact Phase 02 targets, partitions, q90 ownership, AR models, and frozen outer AR
predictions were reused. No AGAIN code, runner, head, numeric recipe, fitted artifact, or
prediction entered runtime.

Code SHA-256: `d704d0052030472ea8ffe6504c9baf45e6b82c02005be1c040d9810649a9f217`
Search checkpoint manifest SHA-256: `3b3a224b6568402c3a41d264cd33b3ba9e9ab2d38cd836b8a98921b74fb9f26b`
Selected checkpoint manifest SHA-256: `654f9992f9718e08da08d4e55d6a6244346932b177c676e128f136cdb5595ba2`
Prediction manifest SHA-256: `e174fea9bfc3d127b35d10a6b3364e549cae313cb9ad0773820b26b7b101ce04`
Selected recipe SHA-256: `ed85a4edf2dafb8ac9e57c6b6dbe429c5543955c780a0d4c5f8733d116d773bf`
Summary SHA-256: `1a803ea9abb9b3ab1bcfbb2068066dd3ff31af52fff7a12a612f86365948c936`
