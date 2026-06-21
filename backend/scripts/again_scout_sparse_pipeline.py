"""AGAIN scout-to-sparse-TRIBE planning utilities.

This module builds the cheap, label-safe parts of the AGAIN pipeline:
alignment-aware telemetry deltas, candidate selectors, sparse ViT-G/TRIBE
teacher queues, and control/split contracts. It intentionally does not run
TRIBE encoding or train models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


AGAIN_DATASET_NAME = "AGAIN_cleaned"
AGAIN_ALIGNMENT_POLICY = "use_annotation_covered_video_time_only"
SCHEMA_VERSION = "again_scout_sparse_pipeline_v1"

CONTROL_PLAYER_ID = "[control]player_id"
CONTROL_GAME = "[control]game"
CONTROL_SESSION_ID = "[control]session_id"
CONTROL_TIME = "[control]time_stamp"
CONTROL_GENRE = "[control]genre"
OUTPUT_AROUSAL = "[output]arousal"

TELEMETRY_CHANGE_COLUMNS = [
    "[general]input_intensity",
    "[general]input_diversity",
    "[general]activity",
    "[general]score",
    "[general]bot_count",
    "[general]bot_movement",
    "[general]object_intensity",
    "[general]event_intensity",
    "key_press_count",
    "idle_time",
    "player_score",
    "player_delta_distance",
    "player_delta_rotation",
    "player_health",
    "player_damaged",
    "player_shooting",
    "player_reloading",
    "reticle_delta_distance",
    "visible_bot_count",
    "bot_delta_distance",
    "bot_damaged",
    "bot_shooting",
    "bot_projectile_player_distance",
    "pick_ups_visible",
    "objects_destroyed",
    "player_death",
    "player_respawn",
    "bot_respawn",
    "visible_jump_count",
    "visible_speed_boost_count",
    "visible_obstacle_count",
    "visible_loop_count",
    "player_is_jumping",
    "bot_charging",
]

DEPLOYABLE_SELECTOR_FEATURES = [
    "telemetry_change_z",
    "cheap_video_audio_z",
    "vjepa_b_novelty_z",
    "vjepa_l_novelty_z",
]

ORACLE_LABEL_FIELDS = [
    "arousal",
    "future_spike_1_3s_ge_0.05",
    "future_spike_1_3s_ge_0.075",
    "future_change_p3s_movement_ge_0.05",
    "future_change_p3s_movement_ge_0.075",
    "pre_spike_2s",
    "pre_spike_4s",
    "pre_spike_6s",
    "pre_spike_8s",
]


@dataclass(frozen=True)
class ScoutModelSpec:
    name: str
    vjepa_version: str
    backend_repo: str
    model_role: str
    model_name: str
    mlx_path: str
    source_checkpoint_path: str
    checkpoint_sha256: str
    hidden_size: int
    num_layers: int
    num_heads: int
    image_size: int
    frames_per_clip: int
    patch_size: int = 16
    tubelet_size: int = 2
    mlx_optimized: bool = True
    status: str = "missing"
    notes: str = ""


@dataclass(frozen=True)
class SelectorConfig:
    selector_name: str
    feature_weights: dict[str, float]
    top_percent: float | None = None
    max_windows_per_video: int | None = None
    include_random_negatives: int = 5
    anchor_every_seconds: int | None = 60
    expansion_seconds: tuple[int, int] = (-8, 4)
    oracle: bool = False


@dataclass(frozen=True)
class SparseTeacherBudget:
    budget_name: str
    max_total_windows: int
    include_selectors: tuple[str, ...]
    require_background: bool = True
    require_anchors: bool = True
    require_games_represented: bool = True


def external_root() -> Path:
    configured = os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    raise EnvironmentError("Set NEURAL_BRIDGE_EXTERNAL_ROOT to the Neural Bridge external assets root.")


def default_again_dataset_root() -> Path:
    return external_root() / "data" / "external" / "AGAIN" / "cleaned"


def default_boundary_manifest_root() -> Path:
    return Path("outputs/again_boundary_aligned_1hz_manifest_20260621_205412")


def default_boundary_audit_root() -> Path:
    return Path("outputs/again_video_boundary_audit_20260621_204520")


def assert_again_only_output_path(path: Path) -> None:
    parts = {part.lower() for part in path.expanduser().parts}
    if "veatic" in parts:
        raise ValueError(f"AGAIN pipeline output cannot target a VEATIC path: {path}")
    if "again" not in str(path).lower():
        raise ValueError(f"AGAIN pipeline output path must be clearly AGAIN scoped: {path}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clean_feature_name(name: str) -> str:
    cleaned = (
        name.strip()
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in cleaned).strip("_").lower()


def video_id_from_metadata_row(row: dict[str, str]) -> str:
    video_name = row.get("video_name", "")
    return Path(video_name).stem


def load_video_metadata_map(metadata_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv_rows(metadata_path):
        key = (row.get(CONTROL_PLAYER_ID, ""), row.get(CONTROL_GAME, ""), row.get(CONTROL_SESSION_ID, ""))
        if all(key):
            out[key] = {**row, "video_id": video_id_from_metadata_row(row)}
    return out


def group_clean_annotations(
    clean_data_path: Path,
    metadata_path: Path,
    *,
    limit_videos: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    metadata = load_video_metadata_map(metadata_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_csv_rows(clean_data_path):
        key = (row.get(CONTROL_PLAYER_ID, ""), row.get(CONTROL_GAME, ""), row.get(CONTROL_SESSION_ID, ""))
        meta = metadata.get(key)
        if not meta:
            continue
        video_id = meta["video_id"]
        if limit_videos is not None and video_id not in grouped and len(grouped) >= limit_videos:
            continue
        enriched = dict(row)
        enriched["video_id"] = video_id
        enriched["video_path"] = str(default_again_dataset_root() / "videos" / meta.get("video_name", ""))
        enriched["time_seconds"] = safe_float(row.get(CONTROL_TIME))
        grouped.setdefault(video_id, []).append(enriched)
    for rows in grouped.values():
        rows.sort(key=lambda item: safe_float(item.get("time_seconds")))
    return grouped


def load_manifest_rows(manifest_path: Path, *, limit_videos: int | None = None) -> list[dict[str, str]]:
    rows = read_csv_rows(manifest_path)
    if limit_videos is None:
        return rows
    keep: set[str] = set()
    filtered: list[dict[str, str]] = []
    for row in rows:
        video_id = row.get("video_id", "")
        if video_id not in keep and len(keep) >= limit_videos:
            continue
        keep.add(video_id)
        filtered.append(row)
    return filtered


def _interp_series(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    finite = np.isfinite(times) & np.isfinite(values)
    if int(np.sum(finite)) == 0:
        return np.zeros_like(target_times, dtype=np.float64)
    times = times[finite]
    values = values[finite]
    order = np.argsort(times)
    return np.interp(target_times, times[order], values[order])


def telemetry_change_feature_rows(
    *,
    manifest_rows: list[dict[str, str]],
    annotations_by_video: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_video_manifest: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        by_video_manifest.setdefault(row["video_id"], []).append(row)

    output: list[dict[str, Any]] = []
    for video_id, rows in by_video_manifest.items():
        native = annotations_by_video.get(video_id, [])
        if not native:
            continue
        native_times = np.array([safe_float(row.get("time_seconds")) for row in native], dtype=np.float64)
        target_times = np.array([safe_float(row.get("time_start_seconds")) for row in rows], dtype=np.float64)
        feature_values: dict[str, np.ndarray] = {}
        delta_abs_arrays: list[np.ndarray] = []
        for col in TELEMETRY_CHANGE_COLUMNS:
            values = np.array([safe_float(row.get(col), 0.0) for row in native], dtype=np.float64)
            interp = _interp_series(native_times, values, target_times)
            delta = np.diff(interp, prepend=interp[:1])
            safe = clean_feature_name(col)
            feature_values[f"{safe}_delta"] = delta
            feature_values[f"{safe}_abs_delta"] = np.abs(delta)
            delta_abs_arrays.append(np.abs(delta))

        activity = _interp_series(
            native_times,
            np.array([safe_float(row.get("[general]activity"), 0.0) for row in native], dtype=np.float64),
            target_times,
        )
        previous_activity = np.roll(activity, 1)
        previous_activity[0] = activity[0]
        idle_to_active = ((previous_activity <= 0.05) & (activity > 0.05)).astype(int)
        active_to_idle = ((previous_activity > 0.05) & (activity <= 0.05)).astype(int)
        key_counts = _interp_series(
            native_times,
            np.array([safe_float(row.get("key_press_count"), 0.0) for row in native], dtype=np.float64),
            target_times,
        )
        action_switching = np.abs(np.diff(key_counts, prepend=key_counts[:1]))
        if delta_abs_arrays:
            combined = np.vstack(delta_abs_arrays).mean(axis=0)
        else:
            combined = np.zeros_like(target_times)
        combined = combined + 0.5 * idle_to_active + 0.25 * active_to_idle + 0.25 * action_switching

        for index, manifest in enumerate(rows):
            row: dict[str, Any] = {
                "dataset_name": AGAIN_DATASET_NAME,
                "video_id": video_id,
                "video_path": manifest.get("video_path", ""),
                "participant_id": manifest.get("participant_id", ""),
                "session_id": manifest.get("session_id", ""),
                "game": manifest.get("game", ""),
                "genre": manifest.get("genre", ""),
                "time_start_seconds": safe_float(manifest.get("time_start_seconds")),
                "alignment_policy": manifest.get("alignment_policy", ""),
                "telemetry_change_score": float(combined[index]),
                "idle_to_active_transition": int(idle_to_active[index]),
                "active_to_idle_transition": int(active_to_idle[index]),
                "action_switching_rate": float(action_switching[index]),
                "label_fields_excluded_from_selector": True,
            }
            for name, values in feature_values.items():
                row[name] = float(values[index])
            output.append(row)
    return add_group_zscores(output, ["telemetry_change_score"], ["video_id"], "telemetry_change_z")


def add_group_zscores(
    rows: list[dict[str, Any]],
    fields: list[str],
    group_fields: list[str],
    output_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(idx)
    for indices in grouped.values():
        values = np.zeros(len(indices), dtype=np.float64)
        for field in fields:
            values += np.array([safe_float(rows[i].get(field), 0.0) for i in indices], dtype=np.float64)
        std = float(np.std(values))
        mean = float(np.mean(values))
        z = (values - mean) / std if std > 1e-12 else np.zeros_like(values)
        for offset, idx in enumerate(indices):
            rows[idx][output_field] = float(z[offset])
    return rows


def default_selector_configs() -> list[SelectorConfig]:
    return [
        SelectorConfig("telemetry_change_top5", {"telemetry_change_z": 1.0}, top_percent=0.05),
        SelectorConfig("cheap_video_audio_top5", {"cheap_video_audio_z": 1.0}, top_percent=0.05),
        SelectorConfig("vjepa_b_novelty_top5", {"vjepa_b_novelty_z": 1.0}, top_percent=0.05),
        SelectorConfig(
            "telemetry_plus_video_audio_top5",
            {"telemetry_change_z": 0.5, "cheap_video_audio_z": 0.5},
            top_percent=0.05,
        ),
        SelectorConfig(
            "telemetry_video_audio_vjepa_b_top5",
            {"telemetry_change_z": 0.34, "cheap_video_audio_z": 0.33, "vjepa_b_novelty_z": 0.33},
            top_percent=0.05,
        ),
        SelectorConfig("telemetry_change_max30", {"telemetry_change_z": 1.0}, max_windows_per_video=30),
        SelectorConfig("telemetry_change_max60", {"telemetry_change_z": 1.0}, max_windows_per_video=60),
        SelectorConfig("telemetry_change_max120", {"telemetry_change_z": 1.0}, max_windows_per_video=120),
        SelectorConfig("oracle_spike_top5_upper_bound", {"oracle_spike_z": 1.0}, top_percent=0.05, oracle=True),
    ]


def selector_score(row: dict[str, Any], config: SelectorConfig) -> float:
    score = 0.0
    for field, weight in config.feature_weights.items():
        score += float(weight) * safe_float(row.get(field), 0.0)
    return score


def select_candidate_timestamps(
    feature_rows: list[dict[str, Any]],
    configs: list[SelectorConfig],
    *,
    seed: int = 17,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        grouped.setdefault(str(row["video_id"]), []).append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for config in configs:
        if not config.oracle:
            forbidden = set(config.feature_weights) & set(ORACLE_LABEL_FIELDS)
            if forbidden:
                raise ValueError(f"Deployable selector {config.selector_name} uses label fields: {sorted(forbidden)}")
        for video_id, rows in grouped.items():
            scored = [
                (selector_score(row, config), safe_float(row.get("time_start_seconds")), row)
                for row in rows
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            if config.top_percent is not None:
                keep_n = max(1, int(math.ceil(len(scored) * float(config.top_percent))))
            elif config.max_windows_per_video is not None:
                keep_n = min(len(scored), int(config.max_windows_per_video))
            else:
                keep_n = len(scored)
            keep_n = min(keep_n, len(scored))
            picked = scored[:keep_n]
            low_pool = scored[max(keep_n, len(scored) // 2) :]
            negatives = rng.sample(low_pool, min(len(low_pool), config.include_random_negatives)) if low_pool else []
            anchors: list[tuple[float, float, dict[str, Any]]] = []
            if config.anchor_every_seconds:
                seen_anchor_seconds = set(range(0, int(max(item[1] for item in scored)) + 1, config.anchor_every_seconds))
                for _, time_s, row in scored:
                    if int(round(time_s)) in seen_anchor_seconds:
                        anchors.append((0.0, time_s, row))
            for source, items in (
                ("candidate", picked),
                ("random_low_salience_negative", negatives),
                ("sparse_anchor", anchors),
            ):
                for score, time_s, source_row in items:
                    selected.append(
                        {
                            "dataset_name": AGAIN_DATASET_NAME,
                            "selector_name": config.selector_name,
                            "selector_is_oracle": bool(config.oracle),
                            "selection_source": source,
                            "video_id": video_id,
                            "video_path": source_row.get("video_path", ""),
                            "participant_id": source_row.get("participant_id", ""),
                            "session_id": source_row.get("session_id", ""),
                            "game": source_row.get("game", ""),
                            "genre": source_row.get("genre", ""),
                            "timestamp_seconds": float(time_s),
                            "selector_score": float(score),
                            "feature_weights_json": json.dumps(config.feature_weights, sort_keys=True),
                            "expansion_start_offset_seconds": int(config.expansion_seconds[0]),
                            "expansion_end_offset_seconds": int(config.expansion_seconds[1]),
                            "deployable_selector": not bool(config.oracle),
                        }
                    )
    return selected


def merge_candidate_regions(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault((row["selector_name"], row["video_id"]), []).append(row)
    merged: list[dict[str, Any]] = []
    for (selector_name, video_id), rows in grouped.items():
        intervals = []
        for row in rows:
            t = safe_float(row.get("timestamp_seconds"))
            start = max(0.0, t + safe_float(row.get("expansion_start_offset_seconds"), -8.0))
            end = max(start, t + safe_float(row.get("expansion_end_offset_seconds"), 4.0))
            intervals.append((start, end, row))
        intervals.sort(key=lambda item: (item[0], item[1]))
        current: dict[str, Any] | None = None
        current_sources: set[str] = set()
        for start, end, row in intervals:
            if current is None or start > safe_float(current["clip_end_seconds"]) + 1e-6:
                if current is not None:
                    current["selection_sources"] = ",".join(sorted(current_sources))
                    merged.append(current)
                current = {
                    "dataset_name": AGAIN_DATASET_NAME,
                    "selector_name": selector_name,
                    "selector_is_oracle": bool(row.get("selector_is_oracle")),
                    "video_id": video_id,
                    "video_path": row.get("video_path", ""),
                    "participant_id": row.get("participant_id", ""),
                    "session_id": row.get("session_id", ""),
                    "game": row.get("game", ""),
                    "genre": row.get("genre", ""),
                    "clip_start_seconds": float(start),
                    "clip_end_seconds": float(end),
                    "selected_timestamp_count": 0,
                    "max_selector_score": -math.inf,
                }
                current_sources = set()
            current["clip_end_seconds"] = max(safe_float(current["clip_end_seconds"]), float(end))
            current["selected_timestamp_count"] = safe_int(current["selected_timestamp_count"]) + 1
            current["max_selector_score"] = max(safe_float(current["max_selector_score"]), safe_float(row["selector_score"]))
            current_sources.add(str(row.get("selection_source", "")))
        if current is not None:
            current["selection_sources"] = ",".join(sorted(current_sources))
            merged.append(current)
    for row in merged:
        row["clip_duration_seconds"] = safe_float(row["clip_end_seconds"]) - safe_float(row["clip_start_seconds"])
    return merged


def fingerprint_sparse_teacher_window(row: dict[str, Any], *, teacher_model: ScoutModelSpec) -> str:
    payload = {
        "dataset": AGAIN_DATASET_NAME,
        "video_id": row.get("video_id", ""),
        "clip_start": round(safe_float(row.get("clip_start_seconds")), 4),
        "clip_end": round(safe_float(row.get("clip_end_seconds")), 4),
        "frame_count": 64,
        "resolution": 256,
        "vjepa_model_name": teacher_model.model_name,
        "vjepa_checkpoint_hash": teacher_model.checkpoint_sha256,
        "mlx_path": teacher_model.mlx_path,
        "hidden_layer_selection": "tribe_v2_selected_hidden_states",
        "preprocessing_version": "vjepa21_vitg_256_ffmpeg_videotoolbox_v1",
        "dtype": "float16",
        "tribe_head_version": "current_tribe_mlx_cortical_head",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def default_sparse_teacher_budgets() -> list[SparseTeacherBudget]:
    return [
        SparseTeacherBudget("sparse_vitg_budget_500", 500, ("telemetry_video_audio_vjepa_b_top5",)),
        SparseTeacherBudget("sparse_vitg_budget_1000", 1000, ("telemetry_video_audio_vjepa_b_top5",)),
        SparseTeacherBudget("sparse_vitg_budget_2000", 2000, ("telemetry_video_audio_vjepa_b_top5",)),
    ]


def build_sparse_teacher_queue(
    region_rows: list[dict[str, Any]],
    budgets: list[SparseTeacherBudget],
    *,
    teacher_model: ScoutModelSpec,
    output_external_root: Path,
) -> list[dict[str, Any]]:
    assert_again_only_output_path(output_external_root)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        candidates = [row for row in region_rows if row["selector_name"] in budget.include_selectors]
        candidates.sort(key=lambda row: (row.get("game", ""), -safe_float(row.get("max_selector_score")), row["video_id"]))
        selected = candidates[: budget.max_total_windows]
        for rank, row in enumerate(selected, start=1):
            fingerprint = fingerprint_sparse_teacher_window(row, teacher_model=teacher_model)
            cache_dir = output_external_root / "sparse_vitg_teacher_cache" / fingerprint[:2] / fingerprint
            rows.append(
                {
                    **row,
                    "budget_name": budget.budget_name,
                    "budget_rank": rank,
                    "teacher_model_name": teacher_model.name,
                    "teacher_model_path": teacher_model.mlx_path,
                    "teacher_checkpoint_sha256": teacher_model.checkpoint_sha256,
                    "strict_fingerprint": fingerprint,
                    "cache_dir": str(cache_dir),
                    "complete_path": str(cache_dir / "complete.json"),
                    "claim_path": str(cache_dir / "claim.json"),
                    "should_encode_vitg": not (cache_dir / "complete.json").exists(),
                    "force_required_to_recompute": True,
                }
            )
    return rows


def model_registry(root: Path | None = None) -> list[ScoutModelSpec]:
    root = root or external_root()
    vitb_mlx = root / "models" / "vjepa21_mlx" / "scout" / "vitb" / "vjepa2_1_vitb_dist_vitG_384.safetensors"
    vitb_pt = root / "models" / "vjepa21-pytorch" / "scout" / "vitb" / "vjepa2_1_vitb_dist_vitG_384.pt"
    vitl_dir = root / "models" / "vjepa21_mlx" / "scout" / "vitl"
    vitl_encoder = vitl_dir / "encoder.safetensors"
    vitg_dir = root / "models" / "vjepa21_mlx" / "vitg"
    vitg_model = vitg_dir / "model.safetensors"
    specs = [
        ScoutModelSpec(
            name="vjepa21_vitb_lukasugar_mlx_scout",
            vjepa_version="2.1",
            backend_repo="lukasugar/vjepa2.1-mlx",
            model_role="primary_scout",
            model_name="vjepa2_1_vit_base_384",
            mlx_path=str(vitb_mlx),
            source_checkpoint_path=str(vitb_pt),
            checkpoint_sha256=sha256_file(vitb_mlx) if vitb_mlx.exists() else "",
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            image_size=384,
            frames_per_clip=64,
            status="ready" if vitb_mlx.exists() else ("source_downloaded_needs_conversion" if vitb_pt.exists() else "missing"),
            notes="Primary V-JEPA 2.1 ViT-B scout; raw .pt is conversion input only.",
        ),
        ScoutModelSpec(
            name="vjepa21_vitl_dgrauet_mlx_scout",
            vjepa_version="2.1",
            backend_repo="dgrauet/vjepa2-mlx",
            model_role="fallback_scout",
            model_name="vjepa2_1_vit_large_384",
            mlx_path=str(vitl_dir),
            source_checkpoint_path="",
            checkpoint_sha256=sha256_file(vitl_encoder) if vitl_encoder.exists() else "",
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            image_size=384,
            frames_per_clip=64,
            status="ready" if vitl_encoder.exists() else "missing",
            notes="Preconverted V-JEPA 2.1 ViT-L MLX fallback scout.",
        ),
        ScoutModelSpec(
            name="vjepa21_vitg_tribe_sparse_teacher",
            vjepa_version="2.1",
            backend_repo="lukasugar/vjepa2.1-mlx-derived conversion",
            model_role="sparse_expensive_teacher",
            model_name="vjepa2_1_vitg_384",
            mlx_path=str(vitg_dir),
            source_checkpoint_path=str(root / "models" / "vjepa21-pytorch" / "vjepa2_1_vitg_384.pt"),
            checkpoint_sha256=sha256_file(vitg_model) if vitg_model.exists() else "",
            hidden_size=1408,
            num_layers=40,
            num_heads=22,
            image_size=256,
            frames_per_clip=64,
            status="ready" if vitg_model.exists() else "missing",
            notes="Current working TRIBE v2 V-JEPA 2.1 ViT-G path; sparse teacher only.",
        ),
    ]
    return specs


def split_contract_rows() -> list[dict[str, Any]]:
    return [
        {"split_name": "temporally_blocked_with_gap", "main_gate": False, "prevents_adjacent_window_leakage": True},
        {"split_name": "leave_video_out_grouped_kfold", "main_gate": True, "group_field": "video_id"},
        {"split_name": "leave_game_out", "main_gate": False, "group_field": "game"},
        {"split_name": "leave_genre_out_or_within_genre_cross_game", "main_gate": False, "group_field": "genre"},
        {"split_name": "leave_participant_or_session_out", "main_gate": False, "group_field": "participant_id/session_id"},
    ]


def control_contract_rows() -> list[dict[str, Any]]:
    controls = [
        ("autoregressive_arousal_history", "annotation_aware_scientific_baseline"),
        ("zero_change_continuous", "continuous_arousal_baseline"),
        ("majority_prevalence_classifier", "spike_baseline"),
        ("timestamp_only", "time_nuisance_baseline"),
        ("video_id_plus_time", "identity_time_nuisance_baseline"),
        ("game_id_plus_time", "again_game_time_nuisance_baseline"),
        ("telemetry_only", "player_behavior_baseline"),
        ("telemetry_change_only", "player_behavior_change_baseline"),
        ("cheap_video_audio_only", "cheap_stimulus_baseline"),
        ("vjepa_b_only", "small_scout_baseline"),
        ("vjepa_b_plus_telemetry", "strong_cheap_baseline"),
        ("shuffled_tribe_rows", "feature_shuffle_control"),
        ("split_local_shuffled_tribe_rows", "fold_local_feature_shuffle_control"),
        ("random_gaussian_same_dim", "dimension_nuisance_control"),
        ("split_local_random_gaussian_same_dim", "fold_local_dimension_nuisance_control"),
        ("label_shuffle_across_videos", "label_shuffle_control"),
        ("label_shuffle_within_video", "temporal_label_shuffle_control"),
        ("feature_shuffle_across_videos", "feature_identity_shuffle_control"),
        ("feature_shuffle_within_video", "temporal_feature_shuffle_control"),
        ("scout_score_shuffle", "selector_bias_control"),
        ("telemetry_shuffle_within_video", "telemetry_timing_control"),
        ("telemetry_shuffle_across_video_or_game", "telemetry_distribution_control"),
        ("random_window_budget_same_size", "sparse_teacher_budget_control"),
        ("timestamp_uniform_budget_same_size", "sparse_teacher_budget_control"),
        ("oracle_spike_selector_upper_bound", "oracle_only_not_cold_start"),
    ]
    return [
        {
            "control_name": name,
            "control_family": family,
            "canonical_control": True,
            "again_specific": "game" in name or "telemetry" in name or "scout" in name,
            "allowed_for_cold_start_claim": "oracle" not in name and "arousal_history" not in name,
        }
        for name, family in controls
    ]


def event_mask_contract_rows() -> list[dict[str, Any]]:
    masks = [
        "all_frames",
        "stable_negative_only",
        "event_only",
        "pre_event_1s",
        "pre_event_2s",
        "pre_event_3s",
        "pre_event_5s",
        "pre_event_8s",
        "event_plus_pre_3s",
        "event_plus_pre_5s",
        "event_plus_pre_8s",
        "balanced_event_vs_stable_1_to_1",
        "balanced_event_vs_stable_1_to_2",
        "balanced_event_vs_stable_1_to_3",
        "balanced_event_vs_stable_1_to_5",
    ]
    return [{"event_mask": mask, "threshold_selected_on_train_only": True} for mask in masks]


def build_run_manifest(output_root: Path, *, stage: str, limit_videos: int | None) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": AGAIN_DATASET_NAME,
        "again_only": True,
        "alignment_policy": AGAIN_ALIGNMENT_POLICY,
        "limit_videos": limit_videos,
        "full_again_dense_vitg_run": False,
        "sparse_vitg_teacher_only": True,
        "models_trained": False,
        "veatic_outputs_modified": False,
        "cuda_dependencies_installed": False,
        "cloud_gpu_used": False,
        "oracle_selectors_for_upper_bound_only": True,
    }


def write_preflight_outputs(output_root: Path, *, limit_videos: int | None = None) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    registry = [asdict(spec) for spec in model_registry()]
    write_json(output_root / "again_scout_model_registry.json", {"models": registry})
    write_csv_rows(output_root / "again_control_contracts.csv", control_contract_rows())
    write_csv_rows(output_root / "again_split_contracts.csv", split_contract_rows())
    write_csv_rows(output_root / "again_event_mask_contracts.csv", event_mask_contract_rows())
    manifest = build_run_manifest(output_root, stage="preflight", limit_videos=limit_videos)
    write_json(output_root / "run_manifest.json", manifest)
    return manifest
