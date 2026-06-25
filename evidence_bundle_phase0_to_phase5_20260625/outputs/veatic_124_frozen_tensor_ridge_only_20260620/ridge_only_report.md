# VEATIC Frozen Tensor Ridge-Only Benchmark

This run uses the existing VEATIC benchmark lane semantics with a frozen tensor adapter.

- benchmark_mode: `existing_suite_with_frozen_tensor_adapter`
- head: `ridge_only`
- computed_scores: `true`
- reused_benchmark_result_rows: `false`
- computed_ar_fresh: `true`
- computed_controls_fresh: `true`
- full_veatic_124: `true`
- video_83_included: `true`
- exclude_video_83_run: `false`
- lane_rows: `504`
- fold_rows: `360`

No promotion JSON or final claim is produced by this run.

## Gate Checks

- benchmark_mode: `pass`
- full_veatic_124: `pass`
- exclude_video_83_run: `pass`
- video_83_included: `pass`
- required_lanes_computed: `pass`
- promotion_json_not_written: `pass`
