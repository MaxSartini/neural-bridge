# Neural Bridge Docs

Current-facing documentation:

- `current_project_state.md` - short operating snapshot for the cleaned repo.
- `veatic_v2_evidence_summary.md` - current VEATIC-124 v2 scientific evidence.
- `external_assets_manifest.md` - source-versus-external asset boundary.
- `veatic_v2_evidence_freeze.md` - frozen v2 evidence bundle, checksums, and no-reencode verifier.
- `veatic_raw_representation_audit.md` - post-v2 raw representation audit and tensor-contract recommendation.
- `superseded_artifacts.md` - deleted, retained, and non-authoritative artifact policy.
- `PROJECT_MEMORY.md` - current memory pointer and handoff policy.
- `../AGENTS.md` - Codex/fresh-session operating contract.

Current benchmark entrypoint:

- `backend/scripts/run_veatic_strict_benchmark.py` - consolidated VEATIC-124 strict suite and contract dry-run.
- `backend/scripts/freeze_veatic_v2_evidence.py` - protected external snapshot and checksum verifier.
- `backend/scripts/run_veatic_frozen_tensor_trained_heads_benchmark.py` - implemented MPS trained-head benchmark over frozen tensor contracts.
- `backend/scripts/again_sparse_tribe_teacher_500.py` - implemented bounded AGAIN sparse teacher pilot runner and reports.
- `tools/export_veatic_raw_representation_tensors.py` - verified export for model-ready post-v2 tensor contracts.
- `backend/scripts/audit_repo_readiness.py` - stale-term, heavyweight-artifact, and orientation-file audit.

Detailed evidence artifacts live outside this folder:

- `benchmarks/veatic/veatic_124_*`
- `outputs/veatic_124_temporal_*`
- `outputs/veatic_124_raw_representation_tensor_export_v1/`
- `reports/again_*20260622_005732.md`
- `reports/again_sparse_tribe_teacher_500_*_20260622_pca_width_reanalysis_v2.md`
- `reports/again_sparse_tribe_teacher_2000_*_20260622_2000_small_pca_confirmatory.md`

Pre-v2 inventories and audits were intentionally removed from `docs/` so this folder reflects the current Neural Bridge era rather than the old 5/20/50-video transition work.
