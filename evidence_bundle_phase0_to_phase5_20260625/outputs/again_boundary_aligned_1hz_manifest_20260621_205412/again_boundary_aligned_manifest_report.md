# AGAIN Boundary-Aligned 1Hz Manifest

## Summary

- alignment policy: `use_annotation_covered_video_time_only`
- videos: `995`
- rows: `121996`
- future spike/change feasible rows: `119011`

## Boundary Rule

Rows start at the audited benchmark start and stop at each video's annotation-covered end time.
The post-annotation tail is not used for benchmark rows, even when it contains visible motion.
Future targets are seconds-based and are only marked feasible through `annotation_end_seconds - 3s`.

## Guardrails

tribe_encoding_run=false
models_trained=false
veatic_outputs_modified=false
final_benchmark_manifest_created=false

## Files

- `outputs/again_boundary_aligned_1hz_manifest_20260621_205412/again_boundary_aligned_1hz_manifest.csv`
- `outputs/again_boundary_aligned_1hz_manifest_20260621_205412/again_boundary_aligned_video_summary.csv`
- `outputs/again_boundary_aligned_1hz_manifest_20260621_205412/again_boundary_aligned_manifest_summary.json`
