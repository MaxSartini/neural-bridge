# Tools

`tools/` contains operational helpers, runtime probes, export utilities, and cache-building wrappers. These are not the canonical place for claim-bearing benchmark results.

## Relevant Tool Groups

- H100/dense cache operations: `run_h100_tribe_postpass.py`, `audit_again_video_boundaries.py`, `audit_again_cleaned_dataset.py`, `audit_again_alignment_offset.py`
- V-JEPA/MLX runtime and conversion: `convert_vjepa21_vitg_hf_to_mlx.py`, `convert_vjepa21_vitb_lukasugar_to_mlx.py`, `probe_mlx_vjepa21_cortical.py`, `probe_veatic_vjepa21_real_video_runtime.py`, `benchmark_vjepa21_single_gpu_microbatch.py`, `profile_mlx_vjepa21_kernel.py`
- VEATIC tensor/evidence export: `export_veatic_raw_representation_tensors.py`, `export_veatic_raw_representation_metadata_bundle.py`
- AGAIN sparse historical work: `build_again_scout_sparse_pipeline.py`, `run_again_sparse_tribe_teacher_500.py`, `run_again_real_scout_selector_validation.py`
- Evidence packaging: `create_phase0_to_phase5_evidence_bundle.py`

## Non-Claim Policy

Runtime probes can validate environment behavior, model conversion, or throughput. They do not establish Neural Bridge performance claims. Current claim-bearing evidence is in `reports/`, `evidence/phase_*`, and `evidence/current_phase_5_5_review/`.
