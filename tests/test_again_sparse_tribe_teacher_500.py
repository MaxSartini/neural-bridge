import random

import numpy as np

from backend.scripts.again_sparse_tribe_teacher_500 import (
    build_sparse_teacher_queue,
    fit_predict_mlx_ridge,
    mlx_pca_fit_transform,
    report_lines_results,
)


def _row(video: str, time_s: int, *, spike: bool = False) -> dict:
    return {
        "video_id": video,
        "video_path": f"/tmp/{video}.webm",
        "time_start_seconds": float(time_s),
        "future_spike_1_3s_ge_0.05": str(spike).lower(),
        "future_spike_1_3s_delta": 1.0 if spike else 0.0,
        "arousal": 0.8 if spike else 0.2,
        "telemetry_change_z": 5.0 if spike else 0.0,
        "cheap_video_audio_z": 5.0 if spike else 0.0,
        "vjepa_b_novelty_z": 5.0 if spike else 0.0,
    }


def test_sparse_teacher_queue_caps_unique_actual_windows():
    rows = []
    for video in ("v1", "v2", "v3"):
        for t in range(2, 80):
            rows.append(_row(video, t, spike=t % 20 == 0))

    queue, summary = build_sparse_teacher_queue(
        rows,
        max_actual_windows=50,
        rng=random.Random(1),
        vjepa21_sha256="vjepa",
        tribe_sha256="tribe",
    )

    assert summary["unique_actual_windows"] <= 50
    assert len({row["cache_fingerprint"] for row in queue}) == summary["unique_actual_windows"]
    assert all(row["temporal_role"] in {"T-2", "T-1", "T"} for row in queue)
    assert not summary["future_rows_included"]


def test_sparse_teacher_pca_is_real_mlx_train_only_and_reports_width():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(8, 20)).astype(np.float32)
    x_test = rng.normal(size=(4, 20)).astype(np.float32)

    train_pca, test_pca, width = mlx_pca_fit_transform(
        x_train,
        x_train,
        x_test,
        pca_width=128,
        random_seed=7,
    )

    assert width == 7
    assert train_pca.shape == (8, 7)
    assert test_pca.shape == (4, 7)


def test_sparse_teacher_ridge_uses_mlx_solver():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(8, 20)).astype(np.float32)
    x_test = rng.normal(size=(4, 20)).astype(np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    train_scores, test_scores, info = fit_predict_mlx_ridge(
        x_train,
        y_train,
        x_test,
        rng=rng,
    )

    assert info["ridge_solver"] == "mlx_conjugate_gradient_dual"
    assert train_scores.shape == (8,)
    assert test_scores.shape == (4,)


def test_sparse_teacher_report_declares_50_of_995_scope():
    lines = report_lines_results(
        lane_rows=[],
        gate_rows=[],
        runtime_summary={"successful_windows": 0},
    )
    report = "\n".join(lines)

    assert "50` selected AGAIN videos out of `995`" in report
    assert "5.0%` of the dataset" in report
    assert "must not be read as a 995-video comparison" in report
