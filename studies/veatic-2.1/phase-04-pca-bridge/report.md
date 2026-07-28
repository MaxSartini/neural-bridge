# VEATIC 2.1 Phase 04 Fold-Owned PCA Bridge

Status: **PASS**

Phase 04 fit one outer-training-only maximum rank-512 PCA basis in each of five
grouped-video cells and one blocked-temporal cell using every owned eligible row. Scaling,
randomized subspace iteration, and eigensolution used float32 MLX operations on `gpu:0` in one
worker. Nested prefixes `[64, 128, 256, 512]` were audited for finiteness,
orthogonality, ordered/cumulative explained variance, reconstruction residual, exact score
checksums, and independent-seed subspace stability.

The VEATIC-derived temporal family crossed current row with causal trailing means at depths
`4` and `6`, derived from the Phase 01 PACF landmark and target width. Complete controls were
evaluated for all width/temporal candidates. A single representation—width
`64`, temporal depth `0`—was selected globally using only
control-adjusted inner-validation fusion behavior before any outer PCA prediction was scored.

Grouped median PR-AUC was `0.315086` for frozen AR,
`0.128426` for selected real PCA-only, and
`0.309344` for AR-plus-selected PCA. Blocked PR-AUC was
`0.276250`, `0.115511`, and `0.259457`, respectively.
Linear PCA fusion claim: **FAIL**. The selected
representation is frozen for the next ordered learned residual question regardless of whether
this linear probe itself clears the claim gate.

No held-out row selected PCA width, temporal context, feature, seed, or model. Exact Phase 02
targets, splits, q90 ownership, and frozen AR predictions were reused. No washout or learned
bridge was fit, no forbidden hidden state or grouped upstream feature was opened, and no AGAIN
runtime input or numeric selection entered the phase.

Code SHA-256: `4d1092d6bbd134c9bd633a69292667e7d10fa2b881917191b9b5c74648955b66`
PCA accuracy audit SHA-256: `d55ae7ebd9a4c0ea15e7307edbc6e643283aa4d1be3813e1fab8d1ddf5a28a37`
Projection cache manifest SHA-256: `03cefe1a72d72021bc08ca2fcb2731d32a2deebc9fced5706407a3afb0f9ec4d`
Prediction manifest SHA-256: `ee3c73f873fe11fcab88fba840ceb7b6f63b5369933c877adf20668a99969623`
Selected representation SHA-256: `e906ff541c01113998e8a4d0081a71fe92417e82137b81c0286fc2414c38adb0`
Summary SHA-256: `70347ca38b29f40acef6660cfc75d4983253f4150feebad0352fc89f384990c0`
