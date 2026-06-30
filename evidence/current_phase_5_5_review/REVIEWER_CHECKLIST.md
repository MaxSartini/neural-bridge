# Reviewer Checklist

1. Read `README.md` at the repo root for the best results first.
2. Read `docs/neural_bridge_phase5_5_evidence_ladder.md` for the canonical narrative.
3. Open this folder's `CLAIM_LEDGER.md` to map each claim to exact numbers and artifacts.
4. Inspect `05_again_phase_3_raw_cortical/` to verify the raw-cortical-alone failure: blocked `raw_cortical_only` `0.124315` vs AR-only `0.203622`.
5. Inspect `12_again_phase_5_5_binary_blocked_confirmation/` for the 10-seed blocked confirmation CSV/JSON/MD evidence.
6. Inspect `13_again_phase_5_5_grouped_compatibility/` for grouped compatibility and updated verdict artifacts.
7. Inspect `01_veatic_v2_foundation/` for the VEATIC foundation CSV/JSON/MD evidence.
8. Inspect `02_again_phase_0_inventory_dense_cache/` through `06_again_phase_4_pca_bridge/` for AGAIN Phase 0-4 substrate and bridge evidence.
9. Inspect `09_again_phase_5_2_blocked_residual_and_ar_audits/` for failed continuous/residual diagnostics and AR decomposition.
10. Inspect `10_again_phase_5_3_target_redesign_and_foldsafe_pca/` for target redesign and fold-safe PCA rationale.
11. Inspect `14_executable_validation_and_code/` for the relevant AGAIN/VEATIC v2 scripts, tests, suites, benchmark artifact folders, and latest deterministic test-suite result.
12. Check `artifact_manifest.csv` for source paths, copied review paths, byte sizes, and SHA-256 checksums.
13. Check `omitted_files.csv` for intentionally omitted artifacts, usually partial metrics or files above the 70 MB per-file cap.
14. Verify that no tracked dossier file is a checkpoint, tensor, dense cache, `.npy`, `.npz`, or full heavy output root.
