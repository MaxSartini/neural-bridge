# Executable Validation And Code Index

This folder pairs the current claim dossier with the tests, benchmark runners, and scripts that make the claim auditable.

## Latest Local Validation

- `python3 -m pytest -q tests`
- Result: `93 passed in 6.42s`
- Date: `2026-06-30`
- `npm run audit:repo`
- Result: `repo_readiness pass controlled_evidence_items=206`

## Files

- `executable_validation_manifest.csv` - reviewer-readable index of relevant tests, scripts, tools, and benchmark artifact folders.
- `executable_validation_manifest.json` - machine-readable version of the same index.
- `test_suite_result_20260630.json` - exact latest local deterministic test-suite result record.

## Policy

This is an executable index, not a benchmark rerun. No training, PCA generation, V-JEPA/TRIBE work, dense-cache mutation, grouped run, 504 run, or heavy output generation was performed to create this folder.

Runtime probes and environment checks are marked as non-claim tools. Claim-bearing evidence remains the reports and phase-numbered evidence bundles linked from the main dossier.
