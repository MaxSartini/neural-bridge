# AGAIN Phase 0: Dense Foundation

## Outcome

Phase 0 built the complete dense substrate on which every AGAIN claim depends. It is data-foundation evidence, not a predictive-model claim.

## Research question

Can the frozen video-to-cortical pipeline produce a complete, row-addressable 2 Hz representation for the cleaned AGAIN dataset without silently losing videos, inventing timestamps, or obscuring quality failures?

## Contract

- Canonical row identity: `video_id`, `row_index`, and saved `time_seconds`.
- Frozen upstream representation: dense V-JEPA/TRIBE-derived predicted cortical features.
- Quality information remains explicit; downstream protocols decide eligibility rather than deleting source rows.
- Heavy features remain external and hash-registered; compact manifests and audits remain in Git.

## Decisive evidence

| Audit | Result |
| --- | ---: |
| Per-video outputs | **`995/995` successful** |
| Saved dense rows | **`243,575` at 2 Hz** |
| Failed-video ledger | **`0` entries** |
| Videos legitimately starting at `0.5s` | `131` |
| Later quality-excluded rows | `4,816` across `966` videos |
| Videos carrying black-frame flags | `0` |

There were no missing required outputs, partial transfers, or surviving stale-success tracebacks. Duplicate-frame flags drove the later quality exclusions; no synthetic `0.0s` rows were inserted for the 131 videos whose first valid saved row is `0.5s`.

## Why this phase mattered

A model score is uninterpretable if row identity, timing, completeness, or quality ownership can drift. Phase 0 converted a large external feature collection into an auditable scientific substrate and made silent data repair impossible.

## Audit trail

[`evidence/`](evidence/) retains the schema, split definitions, video metadata, per-video postpass manifest, encoding and stream audits, and the explicit empty failure ledger. The `41.47 GB` feature collection, derived arrays, and `101 MB` row index remain externally registered. [`runners/build_dense_tribe_postpass.py`](runners/build_dense_tribe_postpass.py) preserves the scientifically named build entrypoint; its older machine-oriented name survives only in provenance.

## Transition

With completeness and time ownership sealed, Phase 1 could align continuous arousal labels and define eligible future targets without falling back to the older sparse representation.

[Continue to Phase 1 — label alignment](../phase-01-label-alignment/README.md) · [Return to the journey](../../README.md)
