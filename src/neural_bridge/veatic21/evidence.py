"""Hash sealing and independently recomputed VEATIC 2.1 evidence checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(f"{path.name}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, allow_pickle=False, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def row_identity_digest(video_id: np.ndarray, row_index: np.ndarray) -> str:
    if len(video_id) != len(row_index):
        raise ValueError("row identity arrays differ in length")
    digest = hashlib.sha256()
    for video, row in zip(video_id, row_index, strict=True):
        digest.update(f"{video}\0{int(row)}\n".encode())
    return digest.hexdigest()


def create_prediction_seal(
    prediction_path: Path,
    *,
    row_digest: str,
    row_count: int,
    cell_digest: str,
    split_digest: str,
    winner_digest: str,
    substrate_digest: str,
    code_digest: str,
    model_digests: Mapping[str, str],
    lanes: tuple[str, ...],
    promotable: bool = False,
) -> dict[str, Any]:
    return {
        "schema": "veatic21_prediction_seal_v1",
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "row_identity_sha256": row_digest,
        "row_count": row_count,
        "cell_sha256": cell_digest,
        "split_sha256": split_digest,
        "winner_sha256": winner_digest,
        "substrate_sha256": substrate_digest,
        "code_sha256": code_digest,
        "model_sha256": dict(sorted(model_digests.items())),
        "lanes": list(lanes),
        "heldout_labels_opened": False,
        "promotable": promotable,
    }


def verify_prediction_seal(root: Path, seal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    seal = dict(seal or load_json(root / "prediction_seal.json"))
    prediction_path = root / str(seal["prediction_file"])
    failures: list[str] = []
    if seal.get("schema") != "veatic21_prediction_seal_v1":
        failures.append("prediction_seal_schema")
    if not prediction_path.is_file():
        failures.append("prediction_file_missing")
        return {"pass": False, "failures": failures}
    if sha256_file(prediction_path) != seal.get("prediction_sha256"):
        failures.append("prediction_sha256")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        required = {"video_id", "row_index", *map(str, seal.get("lanes", []))}
        if not required.issubset(arrays.files):
            failures.append("prediction_arrays")
        else:
            size = len(arrays["video_id"])
            if size != int(seal.get("row_count", -1)):
                failures.append("prediction_row_count")
            if row_identity_digest(arrays["video_id"], arrays["row_index"]) != seal.get(
                "row_identity_sha256"
            ):
                failures.append("prediction_row_identity")
            if (
                len(
                    set(
                        zip(
                            arrays["video_id"].tolist(),
                            arrays["row_index"].tolist(),
                            strict=True,
                        )
                    )
                )
                != size
            ):
                failures.append("duplicate_prediction_rows")
            for lane in seal.get("lanes", []):
                values = arrays[str(lane)]
                if values.shape != (size,) or not np.isfinite(values).all():
                    failures.append(f"prediction_lane_{lane}")
    if seal.get("heldout_labels_opened") is not False:
        failures.append("labels_opened_before_seal")
    return {"pass": not failures, "failures": failures}


def pooled_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("metric inputs must be aligned and finite")
    if len(np.unique(y_true)) != 2:
        raise ValueError("pooled PR-AUC requires both target classes")
    return float(average_precision_score(y_true, scores))


def average_precision_skill(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Normalize average precision against this VEATIC partition's prevalence."""

    y_true = np.asarray(y_true, dtype=np.int8)
    prevalence = float(np.mean(y_true))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("average-precision skill requires both target classes")
    return (pooled_pr_auc(y_true, scores) - prevalence) / (1.0 - prevalence)


def paired_video_bootstrap_pr_auc_delta(
    video_id: np.ndarray,
    y_true: np.ndarray,
    primary_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    """Paired cluster bootstrap of VEATIC PR-AUC skill differences by video."""

    videos = np.asarray(video_id).astype(str)
    y_true = np.asarray(y_true, dtype=np.int8)
    primary_scores = np.asarray(primary_scores, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    if not (
        videos.shape == y_true.shape == primary_scores.shape == reference_scores.shape
    ):
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not np.isfinite(primary_scores).all() or not np.isfinite(reference_scores).all():
        raise ValueError("paired bootstrap scores must be finite")
    unique_videos = np.unique(videos)
    if len(unique_videos) < 2 or len(np.unique(y_true)) != 2:
        raise ValueError("paired bootstrap requires multiple videos and both target classes")

    indices = {video: np.flatnonzero(videos == video) for video in unique_videos}
    observed = average_precision_skill(y_true, primary_scores) - average_precision_skill(
        y_true, reference_scores
    )
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        rows = np.concatenate([indices[video] for video in sample])
        sampled_y = y_true[rows]
        if len(np.unique(sampled_y)) != 2:
            continue
        deltas.append(
            average_precision_skill(sampled_y, primary_scores[rows])
            - average_precision_skill(sampled_y, reference_scores[rows])
        )
    if len(deltas) < max(100, resamples // 2):
        raise ValueError("too few valid paired bootstrap resamples")
    lower, upper = np.quantile(np.asarray(deltas), (0.025, 0.975))
    return {
        "observed_delta": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "valid_resamples": len(deltas),
        "requested_resamples": resamples,
        "seed": seed,
    }


def paired_video_bootstrap_raw_pr_auc_delta(
    video_id: np.ndarray,
    y_true: np.ndarray,
    primary_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float | int]:
    """Paired video-cluster bootstrap of raw PR-AUC differences."""

    videos = np.asarray(video_id).astype(str)
    y_true = np.asarray(y_true, dtype=np.int8)
    primary_scores = np.asarray(primary_scores, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    if not (
        videos.shape == y_true.shape == primary_scores.shape == reference_scores.shape
    ):
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not np.isfinite(primary_scores).all() or not np.isfinite(reference_scores).all():
        raise ValueError("paired bootstrap scores must be finite")
    unique_videos = np.unique(videos)
    if len(unique_videos) < 2 or len(np.unique(y_true)) != 2:
        raise ValueError("paired bootstrap requires multiple videos and both target classes")
    indices = {video: np.flatnonzero(videos == video) for video in unique_videos}
    observed = pooled_pr_auc(y_true, primary_scores) - pooled_pr_auc(y_true, reference_scores)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(resamples):
        sample = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        rows = np.concatenate([indices[video] for video in sample])
        sampled_y = y_true[rows]
        if len(np.unique(sampled_y)) != 2:
            continue
        deltas.append(
            pooled_pr_auc(sampled_y, primary_scores[rows])
            - pooled_pr_auc(sampled_y, reference_scores[rows])
        )
    if len(deltas) < max(100, resamples // 2):
        raise ValueError("too few valid paired bootstrap resamples")
    lower, upper = np.quantile(np.asarray(deltas), (0.025, 0.975))
    return {
        "observed_delta": float(observed),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "valid_resamples": len(deltas),
        "requested_resamples": resamples,
        "seed": seed,
    }


def per_video_pr_auc(
    video_id: np.ndarray, y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float | None]:
    """Return undefined, never a fabricated zero, for single-class videos."""

    output: dict[str, float | None] = {}
    for video in sorted(set(video_id.astype(str)), key=lambda value: int(value)):
        mask = video_id.astype(str) == video
        output[video] = (
            pooled_pr_auc(y_true[mask], scores[mask])
            if len(np.unique(y_true[mask])) == 2
            else None
        )
    return output
