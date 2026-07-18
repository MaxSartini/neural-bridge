# AGAIN Phase 0: Dense Foundation

Phase 0 is substrate evidence, not a model claim.

The foundation contains `995/995` successful per-video outputs and `243,575` saved 2 Hz rows. There are no failed-video entries, missing required outputs, partial transfers, or surviving stale-success tracebacks. Canonical identity uses `video_id`, `row_index`, and saved `time_seconds`.

`131` videos legitimately begin at `0.5s`; no synthetic zero row may be invented. Quality flags remain explicit data: `4,816` rows across `966` videos were quality-excluded by later protocols, driven by duplicate-frame flags, while no videos carried black-frame flags.

`evidence/` retains schema, split definitions, video metadata, the per-video postpass manifest, encoding/stream audits, and the explicit empty failure ledger. The 41.47 GB feature collection, derived arrays, and 101 MB row index remain externally registered. `runners/build_dense_tribe_postpass.py` is scientifically renamed; its former machine-oriented filename survives only in provenance.
