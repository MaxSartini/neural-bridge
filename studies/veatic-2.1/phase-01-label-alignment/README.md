# VEATIC 2.1 Phase 01 — Label Alignment and Target Substrate

Status: **PASS**

Phase 01 reconstructed all 20,657 supervised rows from the authoritative matching V-JEPA
`rows.csv` files, reconfirmed exact row/time identity against Phase 00 and final TRIBE
timestamps, validated finite labels and native interpolation provenance, and retained all
quality flags without filtering.

The frozen label-only rule selected the initial no-washout future-maximum-increase target
`t+1..t+6` (0.5–3.0 seconds). It is the earliest native endpoint satisfying the registered
complete-table coverage, median within-video ACF decay, and distributed descriptive-support
rules. The prospective washout family `t+5..t+10` and `t+6..t+11` was derived from VEATIC PACF
decay, rise time, event duration, coverage, and support; it remains inactive and unselected.

No global binary label, outer split, cortical value, cortical result, PCA, AR fit, or learned
model entered Phase 01. Descriptive global q90 values were used only for support/timing
summaries. Phase 02 must fit q90 inside each outer-training partition.

## Entrypoint and evidence

The VEATIC runner is `src/neural_bridge/veatic21/phase01.py`, invoked with:

```bash
uv run python -m neural_bridge.veatic21 phase01
```

This directory contains the compact result, target registration, dynamics summary,
derivation ledger, report, external artifact manifest, and provenance record. The complete
15-file bundle remains at:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/again-method-restart-20260723/phase-01-label-alignment`

The concluding focused suite passed 45/45 VEATIC tests. Phase 02 fresh target-specific AR is
the single next authorized action after this transition is present on `origin/main`.
