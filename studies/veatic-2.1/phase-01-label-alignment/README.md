# VEATIC 2.1 fresh Phase 01 label alignment

Phase 01 passed all 28 mandatory controls over all 124 videos and all 20,657 exact 2 Hz
rows. It reconstructed labels only from the matching V-JEPA `rows.csv` files, proved exact
TRIBE row-time identity, preserved native interpolation provenance and row-level quality
metadata, and retained every row.

The shortest VEATIC video contains 22 rows. The VEATIC-derived bound is therefore every
future window `(start, end)` satisfying `1 <= start <= end <= 21`: 231 candidates in total.
The 21 `start=1` no-washout candidates are active for the comprehensive Phase 02 AR search;
the other 210 candidates are prospective washout hypotheses and remain inactive. Phase 01
selected no target, created no global binary label or split, and read no cortical value.

This directory is the compact defensible record. Exact copies of `result.json`, `report.md`,
`alignment-schema.json`, `veatic-derivation-ledger.json`, and `artifact-manifest.json` match
the external evidence. Heavy aligned arrays, target matrices, the complete 231-candidate
registry, dynamics, causal diagnostics, request, and checksums remain under:

`/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/fresh-method-rebuild-20260728/phase-01-label-alignment`

Only Phase 02's comprehensive fresh target-specific AR benchmark is authorized next. It must
evaluate all 21 active no-washout candidates under independently reported grouped-video and
blocked-forward protocols, with VEATIC-derived split design, history/feature/capacity/
regularization/optimizer/calibration searches, complete ledgers, and search-sufficiency
gates. No cortical benchmark, PCA, or learned head is authorized yet.
