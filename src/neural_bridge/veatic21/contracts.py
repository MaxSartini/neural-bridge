"""Immutable contracts for the fresh VEATIC 2.1 Phase 00 audit."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path("/Users/maxsartini/Neural Bridge")
ARTIFACT_ROOT = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")

TRIBE_PER_VIDEO_ROOT = ARTIFACT_ROOT / (
    "features/veatic-2.1/tribe-v2/veatic 2.1 raw cortical predictions/per_video"
)
VJEPA_ROOT = ARTIFACT_ROOT / (
    "features/veatic-2.1/vjepa-2.1/veatic 2.1 v jepa 2.1 stuff"
)
LIFECYCLE_ROOT = ARTIFACT_ROOT / "runs/veatic-2.1/fresh-method-rebuild-20260728"
PHASE00_ROOT = LIFECYCLE_ROOT / "phase-00-dense-foundation"

MASTER_SPECIFICATION = REPOSITORY_ROOT / (
    "internal/active/veatic21-master-scientific-specification.md"
)
REBUILD_PROTOCOL = REPOSITORY_ROOT / "internal/active/veatic21-rebuild-protocol.md"
CURRENT_STATE = REPOSITORY_ROOT / "internal/handoff/CURRENT_STATE.md"

EXPECTED_VIDEO_IDS = tuple(str(index) for index in range(124))
EXPECTED_ROW_COUNT = 20_657
EXPECTED_ROW_HZ = 2.0
EXPECTED_TIME_STEP_SECONDS = 0.5
EXPECTED_NATIVE_LABEL_FPS = 25.0
EXPECTED_CORTICAL_WIDTH = 20_484
EXPECTED_CORTICAL_DTYPE = "float16"
EXPECTED_DIAGNOSTIC_WIDTH = 53
EXPECTED_GROUPED_SHAPE_TAIL = (2, 1_408)
EXPECTED_QUALITY_COUNTS = {
    "black_rows": 76,
    "duplicate_rows": 871,
    "both_rows": 24,
    "union_rows": 923,
    "unflagged_rows": 19_734,
    "total_rows": 20_657,
}
EXPECTED_SOURCE_MATCH_COUNTS = {
    "native_exact": 10_357,
    "linear_native_frames": 10_300,
}
BLACK_FRACTION_THRESHOLD = 0.50
DUPLICATE_FRACTION_THRESHOLD = 0.95

TRIBE_TREE_SHA256 = "851d55ccaac7c587495f65cdfbfbcf6bfe22a66a7ab3da2a048d0422e4087a60"
TRIBE_TREE_FILES = 373
TRIBE_TREE_SIZE_BYTES = 866_111_964
VJEPA_ALLOWED_TREE_SHA256 = (
    "cee65f87ff1e118353acd0c6f86c7f8c925e4e612b47884caea0544f6250e1cd"
)
VJEPA_ALLOWED_TREE_FILES = 744
VJEPA_ALLOWED_TREE_SIZE_BYTES = 7_103_590

FORBIDDEN_HIDDEN_STATE_FILENAME = "vjepa21_hidden_states.npz"
VJEPA_ALLOWED_FILENAMES = frozenset(
    {
        "_PAYLOAD_SHA256.json",
        "_UPLOAD_COMPLETE.json",
        "manifest.json",
        "preprocessing.json",
        "rows.csv",
        "status.json",
    }
)
VJEPA_PAYLOAD_VERIFIED_FILENAMES = frozenset(
    {"manifest.json", "preprocessing.json", "rows.csv", "status.json"}
)
TRIBE_VIDEO_FILENAMES = frozenset(
    {"manifest.json", "status.json", "tribe_v2_cortical_predictions.npz"}
)

ROWS_CSV_COLUMNS = (
    "video_id",
    "video_name",
    "video_relpath",
    "arousal_relpath",
    "valence_relpath",
    "row_index",
    "time_seconds",
    "row_hz",
    "clip_start_seconds",
    "clip_end_seconds",
    "native_label_fps",
    "native_label_frame_count",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "source_arousal",
    "source_valence",
    "source_match_quality",
    "encode_policy",
    "arousal",
    "valence",
)
ENCODE_POLICY = "exact_2hz_native_label_support_no_extrapolation"

TRIBE_KEY_SCHEMA = (
    "time_seconds",
    "sample_frame_indices",
    "sample_time_seconds",
    "selected_state_indices",
    "source_frame_position",
    "source_floor_frame_index",
    "source_ceil_frame_index",
    "source_interp_alpha",
    "source_arousal",
    "source_valence",
    "arousal",
    "valence",
    "luma_mean",
    "luma_std",
    "frame_luma_std_mean",
    "motion_absdiff_mean",
    "black_frame_fraction",
    "duplicate_frame_fraction",
    "quality_black_frame_flag",
    "quality_duplicate_frame_flag",
    "quality_exclusion_flag",
    "quality_weight_suggested",
    "cortical_prediction",
    "tribe_grouped_video_feature",
    "temporal_diagnostics53",
)

LABEL_ARRAY_NAMES = frozenset({"source_arousal", "source_valence", "arousal", "valence"})
PHASE00_ACCESSED_TRIBE_ARRAYS = frozenset(
    {
        "time_seconds",
        "cortical_prediction",
        "black_frame_fraction",
        "duplicate_frame_fraction",
        "quality_black_frame_flag",
        "quality_duplicate_frame_flag",
        "quality_exclusion_flag",
        "temporal_diagnostics53",
    }
)

MANDATORY_CHECK_NAMES = (
    "exact_tribe_video_inventory",
    "exact_vjepa_video_inventory",
    "cross_root_video_identity",
    "run_status_complete",
    "complete_tribe_per_video_layout",
    "matching_vjepa_allowed_layout",
    "tribe_manifest_status",
    "vjepa_manifest_status",
    "upload_marker_payload_manifest",
    "allowed_vjepa_file_hashes",
    "per_video_row_count_identity",
    "total_row_count",
    "sequential_row_index",
    "exact_time_grid",
    "exact_video_id",
    "rows_schema_encode_policy",
    "cortical_shape_dtype",
    "cortical_finite",
    "uniform_tribe_key_schema",
    "copied_time_identity",
    "quality_flag_semantics",
    "quality_counts_all_rows_retained",
    "tribe_tree_digest",
    "vjepa_metadata_tree_digest",
    "forbidden_hidden_state_firewall",
    "no_modeling_operations",
    "again_runtime_firewall",
)

FORBIDDEN_AGAIN_RUNTIME_ROOTS = (
    REPOSITORY_ROOT / "src/neural_bridge/again",
    REPOSITORY_ROOT / "studies/again",
    ARTIFACT_ROOT / "features/again",
    ARTIFACT_ROOT / "derived/again",
    ARTIFACT_ROOT / "runs/again",
    ARTIFACT_ROOT / "sealed/again",
)
