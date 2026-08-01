# VEATIC 2.1 Phase 01 Prospective Registration

Status: registered, not executed

Phase 01 derives the target, temporal, and split search space from all 124 VEATIC 2.1 videos
without reading a cortical outcome. The implementation reads the authoritative `rows.csv`
labels and only the explicit non-cortical audit allowlist in the sealed consolidated bundle.

The executor benchmark covered all 124 videos and 20,657 rows. Six CPU processes had the
fastest repeated median for the real CSV/NPZ load, distribution audit, and arousal ACF64
workload. GPU execution is not appropriate for this small, decompression-and-statistics pass;
later PCA and learned-model phases must benchmark MLX GPU independently.

The registration distinguishes three concepts that must not be conflated:

- blocked-forward evidence uses earlier/later eligible rows inside every sufficiently
  supported video;
- grouped and locked confirmation withhold whole videos;
- a later production refit may use all labelled videos only after the evidence recipe is
  frozen and scored, and cannot produce a held-out accuracy estimate.

AGAIN's `70/30` outer blocked split, `80/20` inner split, q90, rows `+4..+10`, 2-second
temporal aggregation, PCA256, and three-checkpoint ensemble are comparison anchors only.
Nothing in this registration selects them for VEATIC.

See [`again-methodology-decision-ledger.md`](again-methodology-decision-ledger.md) for the
historical branch audit and [`experiment-registration.json`](experiment-registration.json)
for the executable request.
