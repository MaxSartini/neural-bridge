"""Run no-new-model non-neuro baselines for OpenLAV.

These baselines use only cached extraction-quality metadata and media container
metadata. They are not a replacement for frozen semantic/audio/video embedding
baselines, but they are stronger than a mean baseline and test whether simple
non-neuro nuisance variables explain the current signal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

from run_openlav_benchmark import aggregate_split_scores, build_feature_masks, fit_predict, score  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ffprobe_json(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def ratio_to_float(value: Any) -> float:
    if not value:
        return 0.0
    text = str(value)
    if "/" not in text:
        return as_float(text)
    numerator, denominator = text.split("/", 1)
    den = as_float(denominator)
    return as_float(numerator) / den if den else 0.0


def media_features(video_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    probe = ffprobe_json(video_path)
    fmt = probe.get("format", {})
    video: dict[str, Any] = {}
    audio: dict[str, Any] = {}
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and not video:
            video = stream
        if stream.get("codec_type") == "audio" and not audio:
            audio = stream
    size_bytes = as_float(fmt.get("size"), video_path.stat().st_size)
    duration = as_float(fmt.get("duration"))
    bitrate = as_float(fmt.get("bit_rate"))
    video_features = {
        "video::duration_seconds": duration,
        "video::size_bytes_log1p": float(np.log1p(size_bytes)),
        "video::container_bitrate_log1p": float(np.log1p(max(bitrate, 0.0))),
        "video::width": as_float(video.get("width")),
        "video::height": as_float(video.get("height")),
        "video::pixels_log1p": float(np.log1p(as_float(video.get("width")) * as_float(video.get("height")))),
        "video::fps": ratio_to_float(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        "video::video_bitrate_log1p": float(np.log1p(max(as_float(video.get("bit_rate")), 0.0))),
        "video::frame_count_log1p": float(np.log1p(max(as_float(video.get("nb_frames")), 0.0))),
    }
    audio_features = {
        "audio::has_audio": 1.0 if audio else 0.0,
        "audio::sample_rate": as_float(audio.get("sample_rate")),
        "audio::channels": as_float(audio.get("channels")),
        "audio::audio_bitrate_log1p": float(np.log1p(max(as_float(audio.get("bit_rate")), 0.0))),
        "audio::audio_duration_seconds": as_float(audio.get("duration"), duration if audio else 0.0),
    }
    return video_features, audio_features


def quality_features(cache_dir: Path) -> dict[str, float]:
    summary = load_json(cache_dir / "tribe_summary.json")
    event_quality = summary.get("event_quality", {})
    segment_quality = summary.get("segment_quality", {})
    features = {
        "text_quality::missing_text": float(bool(event_quality.get("missing_text", True))),
        "text_quality::word_duration_repairs": as_float(event_quality.get("word_duration_repairs")),
        "text_quality::null_word_durations_after_repair": as_float(
            event_quality.get("null_word_durations_after_repair")
        ),
        "text_quality::word_unique_count": as_float(event_quality.get("word_unique_count")),
        "text_quality::top_word_fraction": as_float(event_quality.get("top_word_fraction")),
        "text_quality::zero_duration_word_fraction": as_float(event_quality.get("zero_duration_word_fraction")),
        "text_quality::word_density_per_second": as_float(event_quality.get("word_density_per_second")),
        "text_quality::word_duration_min": as_float(event_quality.get("word_duration_min")),
        "text_quality::word_duration_median": as_float(event_quality.get("word_duration_median")),
        "text_quality::word_duration_max": as_float(event_quality.get("word_duration_max")),
        "text_quality::degenerate_text_dropped": float(bool(event_quality.get("degenerate_text_dropped", False))),
        "quality::retention_ratio": as_float(segment_quality.get("retention_ratio")),
        "quality::kept_segments": as_float(segment_quality.get("kept_segments")),
        "quality::dropped_segments": as_float(segment_quality.get("dropped_segments")),
    }
    return features


def make_matrix(feature_dicts: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.asarray([[features.get(name, 0.0) for name in names] for features in feature_dicts], dtype=np.float32)


def run_condition(
    matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
) -> dict[str, Any]:
    condition_scores = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        predicted = fit_predict(
            matrix[train_idx],
            y[train_idx],
            matrix[test_idx],
            seed + split_index,
        )
        condition_scores.append(score(predicted, y[test_idx], seed + split_index))
    return aggregate_split_scores(condition_scores, baseline_maes)


def split_local_shuffled_additive_control(
    nonneuro_matrix: np.ndarray,
    neuro_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    baseline_maes: list[float],
    seed: int,
) -> dict[str, Any]:
    """Test whether additive lift survives when only the neuro block is fake."""
    condition_scores = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        rng = np.random.default_rng(seed + 40_000 + split_index)
        train_neuro = neuro_matrix[train_idx].copy()
        test_neuro = neuro_matrix[test_idx].copy()
        train_neuro = train_neuro[rng.permutation(train_neuro.shape[0])]
        test_neuro = test_neuro[rng.permutation(test_neuro.shape[0])]
        train_x = np.concatenate([nonneuro_matrix[train_idx], train_neuro], axis=1)
        test_x = np.concatenate([nonneuro_matrix[test_idx], test_neuro], axis=1)
        predicted = fit_predict(train_x, y[train_idx], test_x, seed + split_index)
        condition_scores.append(score(predicted, y[test_idx], seed + split_index))
    return aggregate_split_scores(condition_scores, baseline_maes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--videos-dir", default="/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos")
    parser.add_argument("--output", default="benchmarks/openlav/openlav_nonneuro_baseline_first50.json")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()

    from sklearn.model_selection import GroupShuffleSplit

    manifest_path = Path(args.manifest).expanduser().resolve()
    videos_dir = Path(args.videos_dir).expanduser().resolve()
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = np.asarray([row["group"] for row in rows])
    split_base = np.zeros((len(rows), 1), dtype=np.float32)
    splits = list(
        GroupShuffleSplit(
            n_splits=args.n_splits,
            test_size=args.test_size,
            random_state=args.seed,
        ).split(split_base, groups=groups)
    )

    text_features: list[dict[str, float]] = []
    audio_features: list[dict[str, float]] = []
    video_features: list[dict[str, float]] = []
    for row in rows:
        cache_dir = Path(row["feature_path"]).parent
        video_path = videos_dir / f"{row['stimulus_id']}.webm"
        video_dict, audio_dict = media_features(video_path)
        text_features.append(quality_features(cache_dir))
        audio_features.append(audio_dict)
        video_features.append(video_dict)

    text_names = sorted({name for features in text_features for name in features})
    audio_names = sorted({name for features in audio_features for name in features})
    video_names = sorted({name for features in video_features for name in features})
    combined_names = text_names + audio_names + video_names

    matrices = {
        "text_quality_baseline": (make_matrix(text_features, text_names), text_names),
        "audio_container_baseline": (make_matrix(audio_features, audio_names), audio_names),
        "video_container_baseline": (make_matrix(video_features, video_names), video_names),
        "combined_handcrafted_non_neuro": (
            np.concatenate(
                [
                    make_matrix(text_features, text_names),
                    make_matrix(audio_features, audio_names),
                    make_matrix(video_features, video_names),
                ],
                axis=1,
            ),
            combined_names,
        ),
    }

    neuro_vectors = []
    feature_names = None
    for row in rows:
        feature_path = Path(row["feature_path"])
        with np.load(feature_path) as bundle:
            neuro_vectors.append(np.asarray(bundle["calibration_feature_vector"], dtype=np.float32))
        current_names = load_json(feature_path.with_suffix(".json"))["feature_contract"]["feature_names"]
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError(f"Feature contract mismatch in {feature_path}")
    neuro = np.stack(neuro_vectors)
    masks = build_feature_masks(feature_names or [])
    compact_cortical = neuro[:, masks["cortical_salience"]]
    compact_affect = neuro[:, masks["compact_neuro_affect"]]
    cortical_only = neuro[:, masks["all_cortical"]]
    nonneuro_combined = matrices["combined_handcrafted_non_neuro"][0]
    additive_matrices = {
        "nonneuro_plus_compact_cortical_salience": (
            np.concatenate([nonneuro_combined, compact_cortical], axis=1),
            combined_names + [f"neuro::{name}" for name in np.asarray(feature_names)[masks["cortical_salience"]]],
        ),
        "nonneuro_plus_compact_neuro_affect": (
            np.concatenate([nonneuro_combined, compact_affect], axis=1),
            combined_names + [f"neuro::{name}" for name in np.asarray(feature_names)[masks["compact_neuro_affect"]]],
        ),
        "nonneuro_plus_cortical_only": (
            np.concatenate([nonneuro_combined, cortical_only], axis=1),
            combined_names + [f"neuro::{name}" for name in np.asarray(feature_names)[masks["all_cortical"]]],
        ),
    }
    matrices.update(additive_matrices)

    report: dict[str, Any] = {
        "schema_version": "openlav_handcrafted_nonneuro_baseline_v1",
        "manifest": str(manifest_path),
        "rows": len(rows),
        "groups": int(np.unique(groups).size),
        "split": {
            "method": "repeated_source_url_group_shuffle_holdout",
            "seed": args.seed,
            "n_splits": args.n_splits,
            "test_size": args.test_size,
        },
        "conditions": {
            name: {"feature_count": int(matrix.shape[1]), "feature_names": names}
            for name, (matrix, names) in matrices.items()
        },
        "interpretation": (
            "These are no-new-model non-neuro controls from extraction quality and media metadata. "
            "They are not frozen semantic/audio/video embedding baselines. Additive neuro conditions "
            "test whether current neuro feature blocks add value beyond these handcrafted controls."
        ),
        "targets": {},
    }
    for axis in sorted(rows[0]["targets"]):
        y = np.asarray([row["targets"][axis] for row in rows], dtype=np.float32)
        baseline_scores = []
        for split_index, (train_idx, test_idx) in enumerate(splits):
            baseline_scores.append(
                score(
                    np.full(test_idx.size, np.mean(y[train_idx])),
                    y[test_idx],
                    args.seed + split_index,
                )
            )
        baseline_maes = [entry["mae"] for entry in baseline_scores]
        target_report = {
            "mean_baseline": aggregate_split_scores(baseline_scores, baseline_maes)
        }
        for condition, (matrix, _) in matrices.items():
            target_report[condition] = run_condition(
                matrix,
                y,
                splits,
                baseline_maes,
                args.seed,
            )
        target_report["nonneuro_plus_shuffled_compact_cortical_salience"] = split_local_shuffled_additive_control(
            nonneuro_combined,
            compact_cortical,
            y,
            splits,
            baseline_maes,
            args.seed,
        )
        target_report["nonneuro_plus_shuffled_compact_neuro_affect"] = split_local_shuffled_additive_control(
            nonneuro_combined,
            compact_affect,
            y,
            splits,
            baseline_maes,
            args.seed,
        )
        report["targets"][axis] = target_report

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
