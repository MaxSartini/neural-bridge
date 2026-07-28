# VEATIC 2.1 Phase 01 Label Alignment and Target Substrate

Status: **PASS**

Phase 01 reconstructed all 20,657 supervised rows from matching V-JEPA `rows.csv`
files and confirmed exact `(video_id, row_index, time_seconds)` identity against both the
sealed Phase 00 inventory and final TRIBE timestamps. Arousal, valence, and native
interpolation provenance were finite and exact. All 923 quality-flagged rows remain attached
metadata; no row was filtered.

The frozen label-only rule selected the initial no-washout future-maximum-increase target at
rows `1..6`
(0.5..3.0s).
The selection was the first native-row endpoint to retain at least 90% complete-table
coverage, reach median within-video arousal ACF <= 0.90, and retain repeated descriptive
top-decile support in at least 80% of videos. Descriptive global q90 values were used only for
support/timing summaries; no global binary label was stored. Phase 02 must fit q90 inside each
outer-training partition.

The prospective washout family is not activated or selected. Its candidate windows are
[[5, 10], [6, 11]], derived only from VEATIC
PACF decay, positive-rise duration, selected-event duration, coverage, and support. AGAIN
offsets, seconds, targets, and numeric results were not inherited.

No outer split, cortical value, cortical target result, PCA, AR fit, or learned model entered
Phase 01. All 20 alignment controls passed. Phase 02 fresh target-specific
AR is the single next authorized action after this transition is committed and pushed.

Code SHA-256: `c0a6c781bb3ab0cdf530708d4fd114d6dba4a93884d03c0833069791f018d639`
Alignment SHA-256: `349eceb1635fd50863ab9c6bb627fa6471dd3914a4035abea2392eee45bf57b7`
Target-source SHA-256: `ad8b167dff44ae6a0c1c78ef3e501cc622e6320be9a912d879c3d9fc99863a4f`
Mask SHA-256: `2fe43426a67e2e4d39382b09ed5a812fbe966f0ce5ddb61adf7e901a053b2f43`
Row-ownership SHA-256: `69676e189414a85433ebfd87966684f2353fa69a2ac6a1cd801015a424cf13cd`
Substrate arrays SHA-256: `ce4acca4b2b72320bf224ac057342be34f27c4ea713f2a7f5eed97d3f0125088`
