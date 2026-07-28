"""Immutable Phase 00 contracts for the VEATIC 2.1 rebuild."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path("/Users/maxsartini/Neural Bridge")
ARTIFACT_ROOT = Path("/Volumes/onn. Drive/Neural Bridge Artifacts")
TRIBE_ROOT = ARTIFACT_ROOT / "features/veatic-2.1/tribe-v2/compact-20260716"
VJEPA_ROOT = ARTIFACT_ROOT / "features/veatic-2.1/vjepa-2.1/compact-20260716"
LIFECYCLE_ROOT = ARTIFACT_ROOT / "runs/veatic-2.1/again-method-restart-20260723"
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

TRIBE_TREE_SHA256 = "0d4adc27dd9d226de87d0cfc4df92de14cb7450de6671857e0665418ad26f6dd"
TRIBE_TREE_FILES = 373
TRIBE_TREE_SIZE_BYTES = 866_111_964
VJEPA_ALLOWED_TREE_SHA256 = "cee65f87ff1e118353acd0c6f86c7f8c925e4e612b47884caea0544f6250e1cd"
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

PHASE00_FEATURE_ARRAYS = frozenset(
    {
        "time_seconds",
        "source_frame_position",
        "source_floor_frame_index",
        "source_ceil_frame_index",
        "source_interp_alpha",
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
        "temporal_diagnostics53",
    }
)
LABEL_ARRAY_NAMES = frozenset({"source_arousal", "source_valence", "arousal", "valence"})
PHASE00_REQUIRED_ARRAYS = (
    "time_seconds",
    "black_frame_fraction",
    "duplicate_frame_fraction",
    "quality_black_frame_flag",
    "quality_duplicate_frame_flag",
    "quality_exclusion_flag",
    "quality_weight_suggested",
    "cortical_prediction",
)

MANDATORY_CHECKS = (
    "exact_tribe_video_inventory",
    "exact_vjepa_video_inventory",
    "cross_root_video_identity",
    "run_status_complete",
    "tribe_per_video_files",
    "vjepa_allowed_files",
    "tribe_manifest_status",
    "vjepa_manifest_status",
    "upload_marker_payload_manifest",
    "vjepa_allowed_file_hashes",
    "per_video_row_count_identity",
    "total_row_count",
    "sequential_row_identity",
    "native_two_hz_time_grid",
    "csv_video_identity",
    "csv_schema_and_encode_policy",
    "cortical_layout_and_dtype",
    "cortical_finite",
    "uniform_tribe_key_schema",
    "tribe_csv_time_identity",
    "quality_flag_layout_and_union",
    "quality_counts_and_all_rows_retained",
    "tribe_tree_digest",
    "vjepa_allowed_tree_digest",
    "forbidden_hidden_state_not_read_or_hashed",
    "no_model_or_selection_work",
    "again_source_and_runtime_firewall",
)


class InputBoundaryError(ValueError):
    """Raised before a forbidden path can be opened, loaded, or hashed."""


def reject_forbidden_runtime_path(path: Path | str) -> Path:
    """Fail closed on hidden-state and AGAIN runtime dependencies."""

    candidate = Path(path)
    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    resolved_parts = tuple(part.casefold() for part in candidate.resolve(strict=False).parts)
    for lowered_parts in (candidate_parts, resolved_parts):
        if FORBIDDEN_HIDDEN_STATE_FILENAME.casefold() in lowered_parts:
            raise InputBoundaryError(f"forbidden V-JEPA hidden-state path: {candidate}")
        if "again" in lowered_parts:
            raise InputBoundaryError(f"forbidden AGAIN runtime path: {candidate}")
    return candidate


def validate_runtime_manifest_paths(paths: tuple[Path | str, ...]) -> None:
    for path in paths:
        reject_forbidden_runtime_path(path)
