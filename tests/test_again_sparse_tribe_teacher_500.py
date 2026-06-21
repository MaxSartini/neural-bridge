import random

import numpy as np

from backend.scripts.again_sparse_tribe_teacher_500 import (
    add_gate_rows,
    build_sparse_teacher_queue,
    fit_predict_mlx_ridge,
    mlx_pca_fit_transform,
    report_lines_results,
    select_pca_width_with_inner_video_validation,
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


def test_sparse_teacher_selects_pca_width_with_inner_train_only_validation():
    rng = np.random.default_rng(2)
    rows_per_video = 4
    video_count = 6
    row_count = rows_per_video * video_count
    causal_roles = rng.normal(size=(row_count, 3, 12)).astype(np.float32)
    y = np.asarray([(idx // rows_per_video + idx) % 2 for idx in range(row_count)], dtype=int)
    groups = np.asarray([f"v{idx // rows_per_video}" for idx in range(row_count)])
    ar_base = rng.normal(size=(row_count, 5)).astype(np.float32)
    outer_train_idx = np.arange(row_count)

    selection = select_pca_width_with_inner_video_validation(
        causal_roles=causal_roles,
        ar_base=ar_base,
        y=y,
        groups=groups,
        outer_train_idx=outer_train_idx,
        candidate_widths=(2, 4, 8),
        random_seed=11,
    )

    assert selection["selected_width"] in {2, 4, 8}
    assert selection["test_labels_used_for_selection"] is False
    assert selection["inner_validation_strategy"] == "grouped_video_inner_validation_train_only"
    assert {row["requested_width"] for row in selection["candidate_scores"]} == {2, 4, 8}


def test_sparse_teacher_gates_selected_pca_against_coverage_random_selected_lane():
    gates = add_gate_rows(
        [
            {
                "selector_arm": "hybrid_top5_selected",
                "model_lane": "AR_plus_sparse_pca_train_selected_causal_past2s_mean",
                "mean_pr_auc": 0.44,
            },
            {
                "selector_arm": "coverage_matched_random_to_hybrid",
                "model_lane": "AR_plus_sparse_pca_train_selected_causal_past2s_mean",
                "mean_pr_auc": 0.24,
            },
        ]
    )

    gate = next(row for row in gates if row["gate"] == "pca_train_selected_vs_coverage_random_selected")
    assert gate["pass"] is True
    assert gate["rhs_selector_arm"] == "coverage_matched_random_to_hybrid"
    assert abs(gate["mean_pr_auc_delta"] - 0.2) < 1e-9


def test_sparse_teacher_report_declares_50_of_995_scope():
    lines = report_lines_results(
        lane_rows=[
            {
                "selector_arm": "hybrid_top5_selected",
                "model_lane": "AR_plus_sparse_pca_train_selected_causal_past2s_mean",
                "mean_pr_auc": 0.12,
                "selected_widths": "16,32",
                "mean_inner_validation_pr_auc": 0.34,
            },
            {
                "selector_arm": "hybrid_top5_selected",
                "model_lane": "AR_plus_sparse_pca16_causal_past2s_mean",
                "mean_pr_auc": 0.13,
                "mean_pca_width_actual": 16,
            },
            {
                "selector_arm": "hybrid_top5_selected",
                "model_lane": "AR_plus_raw_sparse_causal_past2s_mean",
                "mean_pr_auc": 0.11,
            },
        ],
        gate_rows=[],
        runtime_summary={"successful_windows": 0},
    )
    report = "\n".join(lines)

    assert "50` selected AGAIN videos out of `995`" in report
    assert "5.0%` of the dataset" in report
    assert "must not be read as a 995-video comparison" in report
    assert "Smaller PCA Width Re-analysis" in report
    assert "Train-selected PCA widths by grouped outer fold: `16,32`" in report
    assert "Hybrid AR + raw sparse causal mean PR-AUC: `11.00%`" in report
