# VEATIC-124 v2 Evidence Freeze

This is the reproducibility contract for the current Neural Bridge baseline.

## Claim Scope

The frozen bundle covers VEATIC-124 v2 video-dominant cortical/TRIBE evidence for arousal event and spike ranking. It does not claim exact continuous arousal forecasting, a finished downstream product model, or a proven full text+audio+video multimodal cache.

The later raw-representation tensor export builds on this evidence freeze. It does not mutate the frozen v2 evidence snapshot or replace the `cortical_pca64_delta` baseline.

## Authoritative Bundle

Tracked manifest:

```bash
benchmarks/veatic/veatic_v2_evidence_manifest.json
```

Protected external snapshot:

```bash
${NEURAL_BRIDGE_EXTERNAL_ROOT}/evidence_snapshots/veatic_124_v2_20260616
```

The snapshot contains:

- copies of the authoritative lightweight tracked evidence files;
- cache metadata copies for all 124 cache entries;
- a cache inventory with raw output presence and byte counts;
- checksums for tracked evidence files and cache metadata;
- a superseded-artifact ledger.

The snapshot intentionally does not duplicate the large `tribe_raw_output.npz` arrays by default. Those remain in the authoritative external cache:

```bash
${NEURAL_BRIDGE_EXTERNAL_ROOT}/benchmarks/veatic/tribe_cache
```

## Verification

Run this before using the v2 baseline as a reference:

```bash
npm run evidence:verify
```

That command checks tracked evidence checksums, protected snapshot copies, cache metadata checksums, raw output presence, raw output byte counts, cache status counts, and modality coverage. It does not re-encode videos.

Mutable orientation docs captured in the protected snapshot are freeze-time context. Current live guidance is checked by `npm run audit:repo`, while immutable benchmark outputs, cache metadata, and snapshot copies remain checksum-verified by `npm run evidence:verify`.

Raw `tribe_raw_output.npz` checksums are optional because the default v2 snapshot records raw output presence and byte counts but does not hash the large arrays. For a future slower raw-array integrity bundle that still does not re-encode, create that new snapshot with:

```bash
python3 backend/scripts/freeze_veatic_v2_evidence.py --create-snapshot --include-raw-cache-checksums
```

## Freezing

Only run this when intentionally creating a new authoritative frozen bundle:

```bash
npm run evidence:freeze
```

The freeze command creates the tracked manifest and a read-only external snapshot. If the baseline changes, create a new snapshot ID rather than mutating the existing v2 snapshot.

## Tracked Versus External

Keep in git:

- small summary reports;
- benchmark JSON/CSV result tables already curated as v2 evidence;
- evidence docs and manifest files;
- scripts needed to verify or reproduce the bundle from existing caches.
- lightweight raw-representation and tensor-export summaries, manifests, and row-metadata samples.

Keep external:

- raw videos and datasets;
- model weights;
- TRIBE raw output arrays;
- feature caches;
- large generated scratch outputs.
- model-ready tensor payloads under `${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1`.

## Modality Boundary

The frozen v2 cache is video-dominant. Current verification expects 124 raw cache outputs, with modality coverage matching the strict audit: most entries are video-only and only the audio-bearing VEATIC items contain text+audio+video-derived metadata. Do not describe this frozen bundle as a full multimodal result.

## Post-freeze Tensor Contract

The current model-ready tensor contract is:

```bash
${NEURAL_BRIDGE_EXTERNAL_ROOT}/tensors/veatic_124_raw_representation_v1
```

Tracked summaries live at:

```bash
outputs/veatic_124_raw_representation_tensor_export_v1
```

It exports `pca_sequence_128_causal_past_2s_mean`, `roi_parcel_features`, `topk_vertices_512`, and `cortical_pca64_delta_frozen_baseline` across `blocked`, `official`, and `grouped_0..4` splits for the three primary targets. Verification passed with `84` tensor contracts, `420` `.npy` tensor files, `14` PCA cache entries reused, and `0` PCA cache rebuilds.
