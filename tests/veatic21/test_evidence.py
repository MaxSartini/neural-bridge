from __future__ import annotations

from pathlib import Path

import numpy as np

from neural_bridge.veatic21.evidence import (
    atomic_save_npz,
    atomic_write_json,
    average_precision_skill,
    create_prediction_seal,
    digest_json,
    paired_video_bootstrap_pr_auc_delta,
    per_video_pr_auc,
    row_identity_digest,
    verify_prediction_seal,
)


def _prediction_arrays(
    *,
    video_id: tuple[str, ...] = ("1", "1", "2"),
    row_index: tuple[int, ...] = (0, 1, 0),
    scores: tuple[float, ...] = (0.1, 0.8, 0.2),
) -> dict[str, np.ndarray]:
    return {
        "video_id": np.asarray(video_id),
        "row_index": np.asarray(row_index, dtype=np.int64),
        "candidate": np.asarray(scores, dtype=np.float64),
    }


def _write_prediction_seal(root: Path, arrays: dict[str, np.ndarray]) -> None:
    prediction_path = root / "heldout_predictions.npz"
    atomic_save_npz(prediction_path, arrays)
    seal = create_prediction_seal(
        prediction_path,
        row_digest=row_identity_digest(arrays["video_id"], arrays["row_index"]),
        row_count=len(arrays["video_id"]),
        cell_digest=digest_json({"target": "arousal_spike", "fold": 0, "seed": 7}),
        split_digest=digest_json({"train": ["2"], "test": ["1"]}),
        winner_digest=digest_json({"candidate": "pca_logistic"}),
        substrate_digest=digest_json({"videos": 124, "rows": 20_657}),
        code_digest=digest_json({"revision": "fixture"}),
        model_digests={"candidate": digest_json({"weights": "fixture"})},
        lanes=("candidate",),
    )
    atomic_write_json(root / "prediction_seal.json", seal)


def test_atomic_prediction_seal_verifies_complete_bundle(tmp_path: Path) -> None:
    _write_prediction_seal(tmp_path, _prediction_arrays())

    assert verify_prediction_seal(tmp_path) == {"pass": True, "failures": []}
    assert {path.name for path in tmp_path.iterdir()} == {
        "heldout_predictions.npz",
        "prediction_seal.json",
    }


def test_prediction_seal_rejects_tampered_prediction_file(tmp_path: Path) -> None:
    arrays = _prediction_arrays()
    _write_prediction_seal(tmp_path, arrays)
    changed = dict(arrays)
    changed["candidate"] = np.asarray((0.9, 0.8, 0.2), dtype=np.float64)
    atomic_save_npz(tmp_path / "heldout_predictions.npz", changed)

    result = verify_prediction_seal(tmp_path)

    assert result["pass"] is False
    assert "prediction_sha256" in result["failures"]


def test_prediction_seal_rejects_duplicate_row_identities(tmp_path: Path) -> None:
    arrays = _prediction_arrays(row_index=(0, 0, 0))
    _write_prediction_seal(tmp_path, arrays)

    assert verify_prediction_seal(tmp_path) == {
        "pass": False,
        "failures": ["duplicate_prediction_rows"],
    }


def test_prediction_seal_rejects_nonfinite_lane(tmp_path: Path) -> None:
    arrays = _prediction_arrays(scores=(0.1, float("nan"), 0.2))
    _write_prediction_seal(tmp_path, arrays)

    assert verify_prediction_seal(tmp_path) == {
        "pass": False,
        "failures": ["prediction_lane_candidate"],
    }


def test_per_video_single_class_is_undefined_not_zero() -> None:
    result = per_video_pr_auc(
        np.asarray(("1", "1", "2", "2", "3", "3")),
        np.asarray((0, 0, 1, 1, 0, 1), dtype=np.int8),
        np.asarray((0.1, 0.2, 0.7, 0.8, 0.2, 0.9), dtype=np.float64),
    )

    assert result == {"1": None, "2": None, "3": 1.0}


def test_prevalence_normalized_skill_and_paired_video_uncertainty() -> None:
    videos = np.repeat(np.asarray(("1", "2", "3", "4", "5", "6")), 4)
    target = np.tile(np.asarray((0, 0, 1, 1), dtype=np.int8), 6)
    primary = np.tile(np.asarray((0.1, 0.2, 0.8, 0.9)), 6)
    reference = np.tile(np.asarray((0.9, 0.8, 0.2, 0.1)), 6)

    assert average_precision_skill(target, primary) == 1.0
    result = paired_video_bootstrap_pr_auc_delta(
        videos,
        target,
        primary,
        reference,
        seed=17,
        resamples=500,
    )

    assert result["observed_delta"] > 0.0
    assert result["ci_lower"] > 0.0
    assert result["valid_resamples"] == 500
