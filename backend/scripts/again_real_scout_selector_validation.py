"""Real AGAIN scout selector validation.

This module validates whether label-free selectors find arousal spike and
pre-spike regions before spending sparse ViT-G/TRIBE compute. It runs cheap
telemetry/video/audio features and V-JEPA 2.1 scout inference, but it does not
train models and does not run ViT-G/TRIBE.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from backend.app.services.mlx_vjepa21_cortical import _preprocess_video_batch
from backend.scripts.again_scout_sparse_pipeline import (
    AGAIN_DATASET_NAME,
    AGAIN_ALIGNMENT_POLICY,
    assert_again_only_output_path,
    default_again_dataset_root,
    default_boundary_manifest_root,
    external_root,
    group_clean_annotations,
    load_manifest_rows,
    safe_float,
    safe_int,
    telemetry_change_feature_rows,
    write_csv_rows,
    write_json,
)
from backend.scripts.mlx_runtime_config import configure_mlx_memory_from_env
from backend.scripts.again_vjepa21_scout import (
    load_dgrauet_vitl_model,
    load_lukasugar_vitb_model,
    pool_scout_tokens,
    scout_spec_by_name,
    scout_window_fingerprint,
)


@dataclass(frozen=True)
class RealScoutConfig:
    limit_videos: int
    scout_stride_seconds: float = 4.0
    scout_clip_seconds: float = 4.0
    scout_frame_count: int = 16
    scout_image_size: int = 384
    cheap_image_size: int = 96
    scout_batch_size: int = 1
    scout_model_name: str = "vjepa21_vitb_lukasugar_mlx_scout"
    scout_input_dtype: str = "float16"
    compile_scout_forward: bool = True
    video_root_override: Path | None = None
    threshold: float = 0.05
    random_seed: int = 17


SELECTOR_BASES: dict[str, dict[str, float] | str] = {
    "telemetry_change": {"telemetry_change_z": 1.0},
    "cheap_video_audio": {"cheap_video_audio_z": 1.0},
    "vjepa_b_novelty": {"scout_novelty_z": 1.0},
    "telemetry_plus_video_audio": {"telemetry_change_z": 0.5, "cheap_video_audio_z": 0.5},
    "hybrid_telemetry_video_audio_vjepa_b": {
        "telemetry_change_z": 0.34,
        "cheap_video_audio_z": 0.33,
        "scout_novelty_z": 0.33,
    },
    "random_same_budget": "random",
    "oracle_spike_upper_bound": "oracle",
}

COVERAGE_MATCH_RANDOM_SELECTORS: dict[str, str] = {
    "random_coverage_matched_to_hybrid": "hybrid_telemetry_video_audio_vjepa_b",
    "random_coverage_matched_to_vjepa_b": "vjepa_b_novelty",
}

ALL_SELECTOR_NAMES = tuple(SELECTOR_BASES) + tuple(COVERAGE_MATCH_RANDOM_SELECTORS)

BUDGETS: dict[str, dict[str, float | int | str]] = {
    "top2pct": {"kind": "top_percent", "value": 0.02},
    "top5pct": {"kind": "top_percent", "value": 0.05},
    "top10pct": {"kind": "top_percent", "value": 0.10},
    "max30": {"kind": "max_windows", "value": 30},
    "max60": {"kind": "max_windows", "value": 60},
    "max120": {"kind": "max_windows", "value": 120},
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ffmpeg_square_filter(image_size: int) -> str:
    short_side = int(256.0 / 224.0 * image_size)
    scale = (
        f"scale='if(gt(iw,ih),-2,{short_side})':"
        f"'if(gt(iw,ih),{short_side},-2)'"
    )
    return f"{scale},crop={image_size}:{image_size}"


def decode_video_grid_ffmpeg(video_path: Path, *, fps: float, image_size: int) -> np.ndarray:
    if fps <= 0:
        raise ValueError("fps must be positive")
    vf = f"fps={fps:.8f},{_ffmpeg_square_filter(image_size)}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-hwaccel",
        "videotoolbox",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip())
    frame_bytes = image_size * image_size * 3
    if len(proc.stdout) < frame_bytes or len(proc.stdout) % frame_bytes:
        raise RuntimeError(f"ffmpeg returned incomplete RGB frames for {video_path}")
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, image_size, image_size, 3).copy()


def decode_audio_rms_ffmpeg(video_path: Path, *, sample_rate: int = 1000) -> tuple[np.ndarray, str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.array([], dtype=np.float32), proc.stderr.decode("utf-8", errors="replace").strip()
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    usable = (len(audio) // sample_rate) * sample_rate
    if usable <= 0:
        return np.array([], dtype=np.float32), "audio shorter than one second"
    return audio[:usable].reshape(-1, sample_rate).std(axis=1).astype(np.float32), ""


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.nanstd(values))
    if not math.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - float(np.nanmean(values))) / std


def normalize_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def select_validation_videos(manifest_rows: list[dict[str, str]], limit: int) -> list[str]:
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in manifest_rows:
        by_video.setdefault(row["video_id"], []).append(row)
    per_game: dict[str, list[tuple[float, str]]] = {}
    for video_id, rows in by_video.items():
        arousal = np.array([safe_float(row.get("arousal")) for row in rows], dtype=np.float64)
        variance = float(np.nanstd(arousal))
        game = rows[0].get("game", "")
        per_game.setdefault(game, []).append((-variance, video_id))
    for videos in per_game.values():
        videos.sort()
    selected: list[str] = []
    games = sorted(per_game)
    while len(selected) < limit and any(per_game.values()):
        for game in games:
            if len(selected) >= limit:
                break
            if per_game[game]:
                _, video_id = per_game[game].pop(0)
                selected.append(video_id)
    return selected


def rows_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["video_id"]), []).append(row)
    for group in out.values():
        group.sort(key=lambda row: safe_float(row.get("time_start_seconds")))
    return out


def compute_label_vectors(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, np.ndarray]:
    times = np.array([safe_float(row.get("time_start_seconds")) for row in rows], dtype=np.float64)
    spike_field = "future_spike_1_3s_ge_0.05" if threshold <= 0.050001 else "future_spike_1_3s_ge_0.075"
    spike = np.array([normalize_bool(row.get(spike_field)) for row in rows], dtype=bool)
    labels = {"spike": spike}
    for lead in (2, 4, 6, 8):
        pre = np.zeros(len(rows), dtype=bool)
        spike_times = times[spike]
        if len(spike_times):
            for idx, time_s in enumerate(times):
                pre[idx] = bool(np.any(np.abs((time_s + lead) - spike_times) <= 0.51))
        labels[f"pre_spike_{lead}s"] = pre
    labels["event_or_pre_8s"] = spike | labels["pre_spike_2s"] | labels["pre_spike_4s"] | labels["pre_spike_6s"] | labels["pre_spike_8s"]
    return labels


def compute_cheap_video_audio_features(
    video_path: Path,
    rows: list[dict[str, Any]],
    *,
    image_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    frame_grid = decode_video_grid_ffmpeg(video_path, fps=1.0, image_size=image_size)
    decode_seconds = time.perf_counter() - started
    frames = frame_grid.astype(np.float32) / 255.0
    gray = frames.mean(axis=-1)
    diff = np.zeros(len(frames), dtype=np.float64)
    if len(frames) > 1:
        diff[1:] = np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2))
    scene = zscore(diff)
    audio_rms, audio_error = decode_audio_rms_ffmpeg(video_path)
    audio_novelty = np.zeros(max(len(frames), 1), dtype=np.float64)
    if len(audio_rms) > 1:
        audio_delta = np.abs(np.diff(audio_rms, prepend=audio_rms[:1]))
        audio_novelty[: min(len(audio_novelty), len(audio_delta))] = audio_delta[: len(audio_novelty)]
    audio_z = zscore(audio_novelty[: len(frames)])
    combined = zscore(diff) + 0.5 * scene + 0.5 * audio_z
    rows_out: list[dict[str, Any]] = []
    for row in rows:
        second = safe_int(row.get("time_start_seconds"))
        idx = min(max(second, 0), len(frames) - 1)
        rows_out.append(
            {
                "video_id": row["video_id"],
                "time_start_seconds": safe_float(row.get("time_start_seconds")),
                "frame_diff_score": float(diff[idx]),
                "motion_proxy_score": float(diff[idx]),
                "scene_cut_score": float(scene[idx]),
                "audio_energy": float(audio_rms[idx]) if idx < len(audio_rms) else 0.0,
                "audio_novelty": float(audio_novelty[idx]) if idx < len(audio_novelty) else 0.0,
                "cheap_video_audio_score": float(combined[idx]),
                "cheap_video_audio_z": float(zscore(combined)[idx]),
            }
        )
    stats = {
        "cheap_decode_seconds": round(decode_seconds, 3),
        "cheap_frame_count": int(len(frames)),
        "audio_available": bool(len(audio_rms)),
        "audio_error": audio_error,
    }
    return rows_out, stats


def scout_cache_key(video_id: str, config: RealScoutConfig, spec_name: str, spec_hash: str) -> str:
    payload = {
        "video_id": video_id,
        "scout": spec_name,
        "checkpoint_sha256": spec_hash,
        "stride": config.scout_stride_seconds,
        "clip": config.scout_clip_seconds,
        "frames": config.scout_frame_count,
        "image_size": config.scout_image_size,
        "input_dtype": config.scout_input_dtype,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _mlx_input_array(tensor: np.ndarray, dtype_name: str) -> mx.array:
    if dtype_name == "float16":
        return mx.array(tensor.astype(np.float16, copy=False))
    if dtype_name == "bfloat16":
        return mx.array(tensor.astype(np.float32, copy=False), dtype=mx.bfloat16)
    if dtype_name == "float32":
        return mx.array(tensor.astype(np.float32, copy=False))
    raise ValueError(f"Unsupported MLX input dtype: {dtype_name}")


def run_scout_for_video(
    *,
    video_id: str,
    video_path: Path,
    duration_seconds: float,
    model: Any,
    forward: Any | None,
    spec: Any,
    config: RealScoutConfig,
    external_cache_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    assert_again_only_output_path(external_cache_root)
    cache_dir = external_cache_root / f"{spec.name}_cache" / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{scout_cache_key(video_id, config, spec.name, spec.checkpoint_sha256)}.npz"
    stat_prefix = "scout"
    if cache_path.exists():
        loaded = np.load(cache_path)
        return {
            "centers": np.asarray(loaded["centers"], dtype=np.float64),
            "embeddings": np.asarray(loaded["embeddings"], dtype=np.float32),
            "novelty": np.asarray(loaded["novelty"], dtype=np.float64),
        }, {
            f"{stat_prefix}_cache_hit": True,
            f"{stat_prefix}_windows": int(len(loaded["centers"])),
            f"{stat_prefix}_decode_seconds": 0.0,
            f"{stat_prefix}_preprocess_seconds": 0.0,
            f"{stat_prefix}_forward_seconds": 0.0,
            f"{stat_prefix}_cache_write_seconds": 0.0,
        }

    scout_started = time.perf_counter()
    grid_fps = config.scout_frame_count / config.scout_clip_seconds
    decode_started = time.perf_counter()
    grid = decode_video_grid_ffmpeg(video_path, fps=grid_fps, image_size=config.scout_image_size)
    decode_seconds = time.perf_counter() - decode_started
    max_start = max(0.0, duration_seconds - config.scout_clip_seconds)
    starts = np.arange(0.0, max_start + 1e-6, config.scout_stride_seconds)
    if not len(starts):
        starts = np.array([0.0], dtype=np.float64)
    centers = starts + config.scout_clip_seconds / 2.0
    embeddings: list[np.ndarray] = []
    preprocess_seconds = 0.0
    forward_seconds = 0.0
    scout_forward = forward or model

    for batch_start in range(0, len(starts), config.scout_batch_size):
        batch_starts = starts[batch_start : batch_start + config.scout_batch_size]
        windows = []
        for start in batch_starts:
            times = start + np.arange(config.scout_frame_count, dtype=np.float64) / grid_fps
            indices = np.rint(times * grid_fps).astype(np.int64)
            indices = np.clip(indices, 0, len(grid) - 1)
            windows.append(grid[indices])
        preprocess_started = time.perf_counter()
        tensor = _preprocess_video_batch(np.stack(windows, axis=0), image_size=config.scout_image_size)
        preprocess_seconds += time.perf_counter() - preprocess_started
        forward_started = time.perf_counter()
        output = scout_forward(_mlx_input_array(tensor, config.scout_input_dtype))
        mx.eval(output)
        forward_seconds += time.perf_counter() - forward_started
        pooled = pool_scout_tokens(np.asarray(output))
        embeddings.extend(np.asarray(pooled, dtype=np.float32))
    embeddings_arr = np.stack(embeddings, axis=0).astype(np.float32)
    novelty = np.zeros(len(embeddings_arr), dtype=np.float64)
    if len(embeddings_arr) > 1:
        diffs = np.linalg.norm(np.diff(embeddings_arr.astype(np.float64), axis=0), axis=1) / max(1, embeddings_arr.shape[1])
        novelty[1:] = diffs
    cache_started = time.perf_counter()
    np.savez_compressed(
        cache_path,
        centers=centers,
        embeddings=embeddings_arr,
        novelty=novelty,
        video_id=video_id,
        checkpoint_sha256=spec.checkpoint_sha256,
        scout_input_dtype=config.scout_input_dtype,
    )
    cache_write_seconds = time.perf_counter() - cache_started
    total_seconds = time.perf_counter() - scout_started
    return {
        "centers": centers,
        "embeddings": embeddings_arr,
        "novelty": novelty,
    }, {
        f"{stat_prefix}_cache_hit": False,
        f"{stat_prefix}_windows": int(len(centers)),
        f"{stat_prefix}_decode_seconds": round(decode_seconds, 3),
        f"{stat_prefix}_preprocess_seconds": round(preprocess_seconds, 3),
        f"{stat_prefix}_forward_seconds": round(forward_seconds, 3),
        f"{stat_prefix}_cache_write_seconds": round(cache_write_seconds, 3),
        f"{stat_prefix}_total_seconds": round(total_seconds, 3),
    }


def map_scout_to_rows(rows: list[dict[str, Any]], scout: dict[str, np.ndarray], *, scout_model_name: str) -> list[dict[str, Any]]:
    centers = scout["centers"]
    novelty = zscore(scout["novelty"])
    model_prefix = "vjepa_l" if "vitl" in scout_model_name else "vjepa_b"
    out = []
    for row in rows:
        time_s = safe_float(row.get("time_start_seconds"))
        value = float(np.interp(time_s, centers, novelty)) if len(centers) else 0.0
        scout_row = {
            "video_id": row["video_id"],
            "time_start_seconds": time_s,
            "scout_model_name": scout_model_name,
            "scout_novelty_score": value,
            "scout_novelty_z": value,
            f"{model_prefix}_novelty_score": value,
            f"{model_prefix}_novelty_z": value,
        }
        if model_prefix == "vjepa_b":
            scout_row["vjepa_l_novelty_score"] = ""
            scout_row["vjepa_l_novelty_z"] = ""
        else:
            scout_row["vjepa_b_novelty_score"] = ""
            scout_row["vjepa_b_novelty_z"] = ""
        out.append(scout_row)
    return out


def combine_feature_rows(
    manifest_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]],
    cheap_rows: list[dict[str, Any]],
    scout_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for row in manifest_rows:
        key = (row["video_id"], safe_float(row.get("time_start_seconds")))
        by_key[key] = dict(row)
    for source_rows in (telemetry_rows, cheap_rows, scout_rows):
        for row in source_rows:
            key = (row["video_id"], safe_float(row.get("time_start_seconds")))
            by_key.setdefault(key, {}).update(row)
    return [by_key[key] for key in sorted(by_key, key=lambda item: (item[0], item[1]))]


def score_rows(rows: list[dict[str, Any]], weights: dict[str, float] | str, rng: random.Random) -> np.ndarray:
    if weights == "random":
        return np.array([rng.random() for _ in rows], dtype=np.float64)
    if weights == "oracle":
        labels = compute_label_vectors(rows, threshold=0.05)
        spike_delta = np.array([safe_float(row.get("future_spike_1_3s_delta"), 0.0) for row in rows], dtype=np.float64)
        arousal = np.array([safe_float(row.get("arousal"), 0.0) for row in rows], dtype=np.float64)
        lead_score = (
            1.00 * labels["spike"].astype(np.float64)
            + 0.80 * labels["pre_spike_2s"].astype(np.float64)
            + 0.60 * labels["pre_spike_4s"].astype(np.float64)
            + 0.40 * labels["pre_spike_6s"].astype(np.float64)
            + 0.20 * labels["pre_spike_8s"].astype(np.float64)
        )
        return lead_score + 0.05 * zscore(spike_delta) + 0.01 * zscore(arousal)
    scores = np.zeros(len(rows), dtype=np.float64)
    for field, weight in weights.items():
        values = []
        for row in rows:
            value = row.get(field)
            if field == "scout_novelty_z" and (value is None or str(value) == ""):
                value = row.get("vjepa_l_novelty_z") or row.get("vjepa_b_novelty_z", 0.0)
            values.append(safe_float(value, 0.0))
        scores += float(weight) * np.array(values, dtype=np.float64)
    return scores


def select_times_for_budget(rows: list[dict[str, Any]], scores: np.ndarray, budget: dict[str, Any]) -> list[float]:
    if len(rows) == 0:
        return []
    kind = budget["kind"]
    value = budget["value"]
    if kind == "top_percent":
        keep_n = max(1, int(math.ceil(len(rows) * float(value))))
    elif kind == "max_windows":
        keep_n = min(len(rows), int(value))
    else:
        raise ValueError(f"unknown budget kind: {kind}")
    order = np.lexsort((np.array([safe_float(row.get("time_start_seconds")) for row in rows]), -scores))
    return [safe_float(rows[int(index)].get("time_start_seconds")) for index in order[:keep_n]]


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-9:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def selected_intervals(selected_times: list[float], *, duration: float) -> list[tuple[float, float]]:
    return merge_intervals([(max(0.0, time_s - 8.0), min(duration, time_s + 4.0)) for time_s in selected_times])


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return float(sum(end - start for start, end in intervals))


def mask_covered(times: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    covered = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        covered |= (times >= start) & (times <= end)
    return covered


def select_random_times_for_target_coverage(
    rows: list[dict[str, Any]],
    *,
    duration: float,
    target_duration: float,
    rng: random.Random,
) -> list[float]:
    if not rows or target_duration <= 0 or duration <= 0:
        return []
    candidate_times = [safe_float(row.get("time_start_seconds")) for row in rows]
    rng.shuffle(candidate_times)
    selected: list[float] = []
    best_selected: list[float] = []
    best_delta = math.inf
    best_duration = 0.0
    for time_s in candidate_times:
        selected.append(time_s)
        current_duration = interval_duration(selected_intervals(selected, duration=duration))
        delta = abs(current_duration - target_duration)
        if delta < best_delta or (math.isclose(delta, best_delta) and current_duration < best_duration):
            best_delta = delta
            best_duration = current_duration
            best_selected = list(selected)
    return best_selected


def evaluate_selector(
    *,
    selector_name: str,
    budget_name: str,
    rows_by_vid: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> dict[str, Any]:
    budget = BUDGETS[budget_name]
    matched_to_selector = COVERAGE_MATCH_RANDOM_SELECTORS.get(selector_name)
    total_positive: dict[str, int] = {name: 0 for name in ("spike", "pre_spike_2s", "pre_spike_4s", "pre_spike_6s", "pre_spike_8s")}
    total_recalled = dict(total_positive)
    selected_positive = 0
    selected_rows = 0
    total_rows = 0
    selected_duration = 0.0
    total_duration = 0.0
    target_selected_duration = 0.0
    timestamps = 0
    regions = 0
    videos = 0

    for rows in rows_by_vid.values():
        videos += 1
        times = np.array([safe_float(row.get("time_start_seconds")) for row in rows], dtype=np.float64)
        duration = float(max(times) + 1.0) if len(times) else 0.0
        if matched_to_selector:
            target_scores = score_rows(rows, SELECTOR_BASES[matched_to_selector], rng)
            target_times = select_times_for_budget(rows, target_scores, budget)
            target_intervals = selected_intervals(target_times, duration=duration)
            target_duration = interval_duration(target_intervals)
            selected_times = select_random_times_for_target_coverage(
                rows,
                duration=duration,
                target_duration=target_duration,
                rng=rng,
            )
            target_selected_duration += target_duration
        else:
            scores = score_rows(rows, SELECTOR_BASES[selector_name], rng)
            selected_times = select_times_for_budget(rows, scores, budget)
            target_duration = math.nan
        intervals = selected_intervals(selected_times, duration=duration)
        covered = mask_covered(times, intervals)
        labels = compute_label_vectors(rows, threshold=0.05)
        for label_name in total_positive:
            positives = labels[label_name]
            total_positive[label_name] += int(np.sum(positives))
            total_recalled[label_name] += int(np.sum(covered & positives))
        selected_positive += int(np.sum(covered & labels["event_or_pre_8s"]))
        selected_rows += int(np.sum(covered))
        total_rows += len(rows)
        selected_duration += interval_duration(intervals)
        total_duration += duration
        timestamps += len(selected_times)
        regions += len(intervals)

    def recall(label: str) -> float:
        denom = total_positive[label]
        return float(total_recalled[label] / denom) if denom else math.nan

    precision = float(selected_positive / selected_rows) if selected_rows else math.nan
    is_oracle = selector_name == "oracle_spike_upper_bound"
    is_control = selector_name == "random_same_budget" or selector_name in COVERAGE_MATCH_RANDOM_SELECTORS
    return {
        "selector_name": selector_name,
        "budget_name": budget_name,
        "selector_is_oracle": is_oracle,
        "selector_is_control": is_control,
        "deployable_selector": not is_oracle and not is_control,
        "coverage_matched_to_selector": matched_to_selector or "",
        "coverage_match_target_percent_of_video": float(target_selected_duration / total_duration) if matched_to_selector and total_duration else math.nan,
        "videos": videos,
        "selected_timestamps": timestamps,
        "candidate_regions": regions,
        "selected_rows": selected_rows,
        "total_rows": total_rows,
        "selected_percent_of_video": float(selected_duration / total_duration) if total_duration else math.nan,
        "spike_recall": recall("spike"),
        "pre_spike_2s_recall": recall("pre_spike_2s"),
        "pre_spike_4s_recall": recall("pre_spike_4s"),
        "pre_spike_6s_recall": recall("pre_spike_6s"),
        "pre_spike_8s_recall": recall("pre_spike_8s"),
        "selection_precision_event_or_pre8": precision,
        "spike_count": total_positive["spike"],
        "pre_spike_2s_count": total_positive["pre_spike_2s"],
        "pre_spike_4s_count": total_positive["pre_spike_4s"],
        "pre_spike_6s_count": total_positive["pre_spike_6s"],
        "pre_spike_8s_count": total_positive["pre_spike_8s"],
    }


def evaluate_selector_video_rows(
    *,
    rows_by_vid: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selector_name in ALL_SELECTOR_NAMES:
        for budget_name in BUDGETS:
            for video_id, video_rows in rows_by_vid.items():
                result = evaluate_selector(
                    selector_name=selector_name,
                    budget_name=budget_name,
                    rows_by_vid={video_id: video_rows},
                    rng=rng,
                )
                result["video_id"] = video_id
                rows.append(result)
    return rows


def run_validation(
    *,
    output_root: Path,
    external_cache_root: Path,
    again_root: Path,
    manifest_root: Path,
    config: RealScoutConfig,
) -> dict[str, Any]:
    assert_again_only_output_path(output_root)
    assert_again_only_output_path(external_cache_root)
    output_root.mkdir(parents=True, exist_ok=False)
    external_cache_root.mkdir(parents=True, exist_ok=True)
    mlx_memory_config = configure_mlx_memory_from_env(mx, label="again_real_scout_selector_validation")
    manifest_path = manifest_root / "again_boundary_aligned_1hz_manifest.csv"
    all_manifest_rows = load_manifest_rows(manifest_path)
    selected_video_ids = set(select_validation_videos(all_manifest_rows, config.limit_videos))
    manifest_rows = [row for row in all_manifest_rows if row["video_id"] in selected_video_ids]
    clean_data_path = again_root / "annotations" / "clean_data.csv"
    metadata_path = again_root / "metadata" / "cleaned_session_video_metadata.csv"
    annotations_by_video = group_clean_annotations(clean_data_path, metadata_path, limit_videos=None)
    telemetry_rows = telemetry_change_feature_rows(
        manifest_rows=manifest_rows,
        annotations_by_video=annotations_by_video,
    )
    telemetry_by_key = {(row["video_id"], safe_float(row["time_start_seconds"])): row for row in telemetry_rows}

    spec = scout_spec_by_name(config.scout_model_name)
    if spec.name == "vjepa21_vitl_dgrauet_mlx_scout":
        model = load_dgrauet_vitl_model(spec, repo_root=Path(".cache/vjepa21-mlx-repos/dgrauet-vjepa2-mlx"))
    elif spec.name == "vjepa21_vitb_lukasugar_mlx_scout":
        model = load_lukasugar_vitb_model(spec, repo_root=Path(".cache/vjepa21-mlx-repos/vjepa2.1-mlx"))
    else:
        raise ValueError(f"Unsupported real scout model for validation: {spec.name}")
    scout_forward = mx.compile(lambda batch: model(batch)) if config.compile_scout_forward else None
    by_video = rows_by_video(manifest_rows)
    cheap_rows_all: list[dict[str, Any]] = []
    scout_rows_all: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []

    for video_index, (video_id, rows) in enumerate(by_video.items(), start=1):
        video_path = Path(rows[0]["video_path"])
        if config.video_root_override is not None:
            video_path = config.video_root_override / video_path.name
        duration = max(safe_float(row.get("time_start_seconds")) for row in rows) + 1.0
        video_started = time.perf_counter()
        cheap_rows, cheap_stats = compute_cheap_video_audio_features(
            video_path,
            rows,
            image_size=config.cheap_image_size,
        )
        scout, scout_stats = run_scout_for_video(
            video_id=video_id,
            video_path=video_path,
            duration_seconds=duration,
            model=model,
            forward=scout_forward,
            spec=spec,
            config=config,
            external_cache_root=external_cache_root,
        )
        scout_rows = map_scout_to_rows(rows, scout, scout_model_name=spec.name)
        cheap_rows_all.extend(cheap_rows)
        scout_rows_all.extend(scout_rows)
        total_seconds = time.perf_counter() - video_started
        times = np.array([safe_float(row.get("time_start_seconds")) for row in rows], dtype=np.float64)
        labels = compute_label_vectors(rows, threshold=config.threshold)
        throughput_rows.append(
            {
                "video_index": video_index,
                "video_id": video_id,
                "video_path": str(video_path),
                "video_duration_seconds": duration,
                "telemetry_rows": len(rows),
                "spike_rows": int(np.sum(labels["spike"])),
                "pre_spike_2s_rows": int(np.sum(labels["pre_spike_2s"])),
                "scout_model": spec.name,
                "scout_windows_processed": scout_stats["scout_windows"],
                "vjepa_b_windows_processed": scout_stats["scout_windows"],
                "cheap_decode_seconds": cheap_stats["cheap_decode_seconds"],
                "audio_available": cheap_stats["audio_available"],
                "scout_cache_hit": scout_stats["scout_cache_hit"],
                "scout_decode_seconds": scout_stats["scout_decode_seconds"],
                "scout_preprocess_seconds": scout_stats["scout_preprocess_seconds"],
                "scout_forward_seconds": scout_stats["scout_forward_seconds"],
                "scout_cache_write_seconds": scout_stats["scout_cache_write_seconds"],
                "vjepa_b_cache_hit": scout_stats["scout_cache_hit"],
                "vjepa_b_decode_seconds": scout_stats["scout_decode_seconds"],
                "vjepa_b_preprocess_seconds": scout_stats["scout_preprocess_seconds"],
                "vjepa_b_forward_seconds": scout_stats["scout_forward_seconds"],
                "vjepa_b_cache_write_seconds": scout_stats["scout_cache_write_seconds"],
                "total_scout_time_video_seconds": round(total_seconds, 3),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{video_index}/{len(by_video)}",
                    "video_id": video_id,
                    "scout_model": spec.name,
                    "scout_windows": scout_stats["scout_windows"],
                    "total_seconds": round(total_seconds, 3),
                    "cache_hit": scout_stats["scout_cache_hit"],
                }
            ),
            flush=True,
        )

    combined_rows = combine_feature_rows(manifest_rows, telemetry_rows, cheap_rows_all, scout_rows_all)
    grouped_combined = rows_by_video(combined_rows)
    rng = random.Random(config.random_seed)
    selector_rows: list[dict[str, Any]] = []
    for selector_name in ALL_SELECTOR_NAMES:
        for budget_name in BUDGETS:
            selector_rows.append(
                evaluate_selector(
                    selector_name=selector_name,
                    budget_name=budget_name,
                    rows_by_vid=grouped_combined,
                    rng=rng,
                )
            )
    selector_video_rows = evaluate_selector_video_rows(
        rows_by_vid=grouped_combined,
        rng=random.Random(config.random_seed),
    )

    write_csv_rows(output_root / "again_real_scout_throughput.csv", throughput_rows)
    write_csv_rows(output_root / "again_real_scout_selector_metrics.csv", selector_rows)
    write_csv_rows(output_root / "again_real_scout_selector_video_metrics.csv", selector_video_rows)
    write_csv_rows(output_root / "again_real_scout_feature_rows.csv", combined_rows)
    run_manifest = {
        "schema_version": "again_real_scout_selector_validation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": AGAIN_DATASET_NAME,
        "alignment_policy": AGAIN_ALIGNMENT_POLICY,
        "limit_videos": config.limit_videos,
        "videos": len(by_video),
        "output_root": str(output_root),
        "external_cache_root": str(external_cache_root),
        "again_only": True,
        "mlx_only": True,
        "cuda_dependencies_installed": False,
        "dense_vitg_tribe_run": False,
        "models_trained": False,
        "arousal_labels_used_for_deployable_selection": False,
        "arousal_labels_used_for_evaluation": True,
        "scout_model": spec.name,
        "scout_checkpoint_sha256": spec.checkpoint_sha256,
        "scout_stride_seconds": config.scout_stride_seconds,
        "scout_clip_seconds": config.scout_clip_seconds,
        "scout_frame_count": config.scout_frame_count,
        "scout_image_size": config.scout_image_size,
        "scout_input_dtype": config.scout_input_dtype,
        "compile_scout_forward": config.compile_scout_forward,
        "video_root_override": "" if config.video_root_override is None else str(config.video_root_override),
        "mlx_memory_config": mlx_memory_config,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    return {
        "manifest": run_manifest,
        "throughput_rows": throughput_rows,
        "selector_rows": selector_rows,
        "selector_video_rows": selector_video_rows,
        "feature_rows": combined_rows,
    }


def render_report(results: dict[str, Any], *, report_path: Path) -> str:
    selector_rows = results["selector_rows"]
    throughput_rows = results["throughput_rows"]
    manifest = results.get("manifest", {})
    scout_model_name = str(manifest.get("scout_model", "vjepa21_vitb_lukasugar_mlx_scout"))
    scout_label = "V-JEPA-L" if "vitl" in scout_model_name else "V-JEPA-B"

    def row_for(selector: str, budget: str) -> dict[str, Any] | None:
        for row in selector_rows:
            if row["selector_name"] == selector and row["budget_name"] == budget:
                return row
        return None

    hybrid_top5 = row_for("hybrid_telemetry_video_audio_vjepa_b", "top5pct") or {}
    random_top5 = row_for("random_same_budget", "top5pct") or {}
    random_matched_hybrid_top5 = row_for("random_coverage_matched_to_hybrid", "top5pct") or {}
    random_matched_hybrid_top10 = row_for("random_coverage_matched_to_hybrid", "top10pct") or {}
    random_matched_hybrid_max30 = row_for("random_coverage_matched_to_hybrid", "max30") or {}
    random_matched_vjepa_top5 = row_for("random_coverage_matched_to_vjepa_b", "top5pct") or {}
    vjepa_top5 = row_for("vjepa_b_novelty", "top5pct") or {}
    telemetry_top5 = row_for("telemetry_change", "top5pct") or {}
    oracle_top5 = row_for("oracle_spike_upper_bound", "top5pct") or {}
    hybrid_top10 = row_for("hybrid_telemetry_video_audio_vjepa_b", "top10pct") or {}
    hybrid_max30 = row_for("hybrid_telemetry_video_audio_vjepa_b", "max30") or {}
    cache_hits = sum(1 for row in throughput_rows if str(row.get("scout_cache_hit", row.get("vjepa_b_cache_hit"))).lower() == "true")
    uncached_forward = [
        safe_float(row.get("scout_forward_seconds", row.get("vjepa_b_forward_seconds")))
        for row in throughput_rows
        if str(row.get("scout_cache_hit", row.get("vjepa_b_cache_hit"))).lower() != "true"
    ]
    mean_forward = np.mean(uncached_forward) if uncached_forward else math.nan
    mean_total = np.mean([safe_float(row.get("total_scout_time_video_seconds")) for row in throughput_rows])
    total_windows = sum(safe_int(row.get("scout_windows_processed", row.get("vjepa_b_windows_processed"))) for row in throughput_rows)

    def pct(value: Any) -> str:
        val = safe_float(value)
        return "n/a" if not math.isfinite(val) else f"{100 * val:.1f}%"

    lines = [
        "# AGAIN Real Scout Selector Validation",
        "",
        "## Executive Summary",
        f"- Videos validated: {len(throughput_rows)}",
        f"- Scout model: `{scout_model_name}`",
        f"- {scout_label} scout windows processed: {total_windows}",
        f"- {scout_label} cache hits: {cache_hits}/{len(throughput_rows)}",
        f"- Mean uncached {scout_label} forward time/video: {'n/a' if not math.isfinite(mean_forward) else f'{mean_forward:.3f}s'}",
        f"- Mean total scout time/video: {mean_total:.3f}s",
        f"- Hybrid top-5% spike recall: {pct(hybrid_top5.get('spike_recall'))}",
        f"- Same-timestamp-budget random top-5% spike recall: {pct(random_top5.get('spike_recall'))}",
        f"- Coverage-matched random top-5% spike recall: {pct(random_matched_hybrid_top5.get('spike_recall'))}",
        f"- Oracle top-5% spike recall: {pct(oracle_top5.get('spike_recall'))}",
        f"- Hybrid top-10% spike recall: {pct(hybrid_top10.get('spike_recall'))}",
        f"- Hybrid max-30 spike recall: {pct(hybrid_max30.get('spike_recall'))}",
        "",
        "## Guardrails",
        "- AGAIN only.",
        "- MLX only.",
        "- No CUDA.",
        "- No dense ViT-G/TRIBE encoding.",
        "- No training.",
        "- Arousal labels are used only for evaluation and oracle upper bound selectors.",
        "",
        "## Selector Takeaways",
        f"- Telemetry-change top-5% spike recall: {pct(telemetry_top5.get('spike_recall'))}",
        f"- V-JEPA-B novelty top-5% spike recall: {pct(vjepa_top5.get('spike_recall'))}",
        f"- Hybrid top-5% pre-spike 2s/4s/6s/8s recall: "
        f"{pct(hybrid_top5.get('pre_spike_2s_recall'))} / "
        f"{pct(hybrid_top5.get('pre_spike_4s_recall'))} / "
        f"{pct(hybrid_top5.get('pre_spike_6s_recall'))} / "
        f"{pct(hybrid_top5.get('pre_spike_8s_recall'))}",
        f"- Hybrid top-5% selected video coverage: {pct(hybrid_top5.get('selected_percent_of_video'))}",
        f"- Coverage-matched random top-5% selected video coverage: {pct(random_matched_hybrid_top5.get('selected_percent_of_video'))}",
        f"- Hybrid top-10% selected video coverage: {pct(hybrid_top10.get('selected_percent_of_video'))}",
        f"- Coverage-matched random top-10% selected video coverage: {pct(random_matched_hybrid_top10.get('selected_percent_of_video'))}",
        f"- Hybrid max-30 selected video coverage: {pct(hybrid_max30.get('selected_percent_of_video'))}",
        f"- Coverage-matched random max-30 selected video coverage: {pct(random_matched_hybrid_max30.get('selected_percent_of_video'))}",
        "",
        "## Interpretation",
    ]
    hybrid_gain = safe_float(hybrid_top5.get("spike_recall")) - safe_float(random_matched_hybrid_top5.get("spike_recall"))
    vjepa_gain = safe_float(vjepa_top5.get("spike_recall")) - safe_float(random_matched_vjepa_top5.get("spike_recall"))
    if math.isfinite(hybrid_gain) and hybrid_gain > 0:
        lines.append("- Hybrid selection beat coverage-matched random on spike recall in this subset.")
    else:
        lines.append("- Hybrid selection did not beat coverage-matched random on spike recall in this subset.")
    raw_random_gain = safe_float(hybrid_top5.get("spike_recall")) - safe_float(random_top5.get("spike_recall"))
    if math.isfinite(raw_random_gain) and raw_random_gain < 0:
        lines.append("- The earlier same-timestamp random control covered more video after region expansion, so it is retained as a stress test but not the fair primary random control.")
    if math.isfinite(vjepa_gain) and vjepa_gain > 0:
        lines.append("- V-JEPA-B novelty beat its own coverage-matched random control at top-5%.")
    else:
        lines.append("- V-JEPA-B novelty did not beat its own coverage-matched random control at top-5%.")
    lines.extend(
        [
            "",
            "## Recommended Next Budget",
            "- Start sparse ViT-G/TRIBE with 500 windows only if hybrid beats random at a useful coverage level.",
            "- Use 1000 or 2000 windows only after the 500-window teacher subset shows lift over telemetry and V-JEPA-B controls.",
            "",
            "## Files",
            f"- Output root: `{results['manifest']['output_root']}`",
            f"- External cache root: `{results['manifest']['external_cache_root']}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report
