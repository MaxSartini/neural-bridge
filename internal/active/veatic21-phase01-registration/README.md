# VEATIC 2.1 Phase 01 Prospective Registration

Status: registered, not executed

Phase 01 derives target, temporal-geometry, and ownership candidates from all 124 VEATIC 2.1
videos without reading a cortical outcome. The implementation reads the authoritative
`rows.csv` labels and only the explicit non-cortical audit allowlist in the sealed
consolidated bundle.

The executor benchmark covered all 124 videos and 20,657 rows. Six CPU processes had the
fastest repeated median for the real CSV/NPZ load, distribution audit, and arousal ACF64
workload. GPU execution is not appropriate for this small, decompression-and-statistics pass;
the consolidated Phase 02 projection and learned-model build must benchmark MLX GPU
independently.

The registration fixes these ownership rules:

- blocked-forward evidence uses earlier/later eligible rows inside every sufficiently
  supported video;
- grouped confirmation withholds whole videos by fold while using every video across the
  complete fold set;
- a later production refit may use all labelled videos only after the evidence recipe is
  frozen and scored, and cannot produce a held-out accuracy estimate.

ACF extends to 120 seconds with explicit tapering support. A target geometry must retain at
least 90% of videos with ten eligible rows per participating video. The run writes exact
blocked and grouped ownership hashes. Each Phase 02 event-threshold candidate is bound to
its exact nonzero-washout maximum-positive-arousal geometry and must retain at least 20 event
rows and event support in at least half of the videos in every grouped test fold. Within each
geometry, thresholds above the VEATIC training-fold median relative range are rejected before
the compact candidate set is emitted.

The execution pattern is unit/integrity tests followed by one comprehensive run loading all
124 videos and using every eligible row for each supported geometry in the complete
registered derivation set.

See [`experiment-registration.json`](experiment-registration.json) for the executable
request.
