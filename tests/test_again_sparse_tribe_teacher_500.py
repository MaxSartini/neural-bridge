import random

import numpy as np

from backend.scripts.again_sparse_tribe_teacher_500 import (
    add_gate_rows,
    build_expensive_window_cache_index,
    build_sparse_teacher_queue,
    cache_fingerprint,
    cache_path_for,
    encode_sparse_windows,
    fit_predict_mlx_ridge,
    fingerprint_payload,
    mlx_pca_fit_transform,
    report_lines_results,
    select_pca_width_with_inner_video_validation,
    validate_single_gpu_runtime,
    write_cached_window,
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
        "scout_model_name": "vjepa21_vitl_dgrauet_mlx_scout",
        "scout_novelty_z": 5.0 if spike else 0.0,
        "vjepa_l_novelty_z": 5.0 if spike else 0.0,
        "vjepa_b_novelty_z": "",
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
    assert all(row["scout_model_name"] == "vjepa21_vitl_dgrauet_mlx_scout" for row in queue)
    assert all("scout_novelty_z" in row for row in queue)
    assert not summary["future_rows_included"]


def test_sparse_teacher_queue_accepts_true_same_budget_fixed_random_control():
    rows = []
    for video in ("v1", "v2", "v3"):
        for t in range(2, 80):
            rows.append(_row(video, t, spike=t % 20 == 0))

    queue, summary = build_sparse_teacher_queue(
        rows,
        max_actual_windows=60,
        rng=random.Random(2),
        vjepa21_sha256="vjepa",
        tribe_sha256="tribe",
        arm_window_budgets={
            "hybrid_top5_selected": 30,
            "fixed_random_same_budget": 30,
        },
    )

    assert summary["unique_actual_windows"] <= 60
    assert set(summary["arm_unique_window_counts"]) == {
        "hybrid_top5_selected",
        "fixed_random_same_budget",
    }
    assert summary["arm_unique_window_counts"]["fixed_random_same_budget"] <= summary["arm_window_budgets"]["fixed_random_same_budget"]
    assert {row["selector_arm"] for row in queue} == {"hybrid_top5_selected", "fixed_random_same_budget"}


def test_sparse_teacher_cache_index_ignores_selector_hash_for_expensive_window_identity(tmp_path):
    payload_old_selector = fingerprint_payload(
        video_id="v1",
        video_path="/tmp/v1.webm",
        actual_clip_timestamp=12.0,
        vjepa21_sha256="vjepa",
        tribe_sha256="tribe",
        selector_arm="old_arm",
        selector_config_hash="old_hash",
        strict_selector_fingerprint=True,
    )
    old_fp = cache_fingerprint(payload_old_selector)
    old_path = cache_path_for(tmp_path, old_fp)
    write_cached_window(
        old_path,
        np.ones((2, 3), dtype=np.float32),
        np.ones((3, 4), dtype=np.float32),
        payload_old_selector,
    )

    payload_new_selector = fingerprint_payload(
        video_id="v1",
        video_path="/tmp/v1.webm",
        actual_clip_timestamp=12.0,
        vjepa21_sha256="vjepa",
        tribe_sha256="tribe",
        selector_arm="new_arm",
        selector_config_hash="new_hash",
        strict_selector_fingerprint=True,
    )

    index = build_expensive_window_cache_index(tmp_path)
    key = next(iter(index))

    assert index[key] == old_path
    assert "old_hash" not in key
    assert "new_hash" not in key
    assert "selector_arm" not in key
    assert cache_fingerprint(payload_new_selector) != old_fp


def test_sparse_teacher_queue_fingerprint_includes_teacher_dtype():
    rows = [_row("v1", t, spike=t % 4 == 0) for t in range(2, 16)]
    common = {
        "max_actual_windows": 9,
        "rng": random.Random(1),
        "vjepa21_sha256": "vjepa",
        "tribe_sha256": "tribe",
        "arm_window_budgets": {"hybrid_top5_selected": 9},
    }

    queue_f16, summary_f16 = build_sparse_teacher_queue(rows, teacher_dtype="float16", **common)
    queue_bf16, summary_bf16 = build_sparse_teacher_queue(rows, teacher_dtype="bfloat16", **common)

    assert summary_f16["teacher_dtype"] == "float16"
    assert summary_bf16["teacher_dtype"] == "bfloat16"
    assert {row["cache_fingerprint"] for row in queue_f16}.isdisjoint(
        {row["cache_fingerprint"] for row in queue_bf16}
    )


def test_sparse_teacher_rejects_multiple_gpu_workers():
    try:
        validate_single_gpu_runtime(
            gpu_workers=2,
            preprocess_workers=0,
            ready_queue_max_size=4,
            writer_queue_max_size=4,
            microbatch_size=1,
        )
    except ValueError as exc:
        assert "exactly one GPU owner" in str(exc)
    else:
        raise AssertionError("expected multiple GPU workers to be rejected")


def test_sparse_teacher_microbatches_uncached_windows(monkeypatch, tmp_path):
    rows = [_row("v1", t, spike=False) for t in (4, 5)]
    queue = []
    for row in rows:
        payload = fingerprint_payload(
            video_id=row["video_id"],
            video_path=row["video_path"],
            actual_clip_timestamp=row["time_start_seconds"],
            vjepa21_sha256="vjepa",
            tribe_sha256="tribe",
            selector_arm="hybrid_top5_selected",
            selector_config_hash="selector",
        )
        queue.append(
            {
                **row,
                "selector_arm": "hybrid_top5_selected",
                "actual_clip_timestamp": row["time_start_seconds"],
                "cache_fingerprint": cache_fingerprint(payload),
                "selector_config_hash": "selector",
            }
        )

    calls = {"forward_batches": []}

    def fake_decode(*_args, **_kwargs):
        return np.zeros((128, 8, 8, 3), dtype=np.uint8)

    def fake_sample(_grid, *, fps, times):
        assert fps > 0
        return np.zeros((len(times), 8, 8, 3), dtype=np.uint8)

    class FakeModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def predict_hidden_states(self, images, selected_indices):
            calls["forward_batches"].append(int(images.shape[0]))
            return np.ones((images.shape[0], len(selected_indices), 1, 4), dtype=np.float32)

    class FakeTribe:
        def __init__(self, *_args, **_kwargs):
            pass

        def predict(self, features):
            batch = features["video"].shape[0]
            return np.ones((batch, 6, 3), dtype=np.float32)

    monkeypatch.setattr("backend.scripts.again_sparse_tribe_teacher_500.IMAGE_SIZE", 8)
    monkeypatch.setattr("backend.scripts.again_sparse_tribe_teacher_500._decode_video_grid_ffmpeg", fake_decode)
    monkeypatch.setattr("backend.scripts.again_sparse_tribe_teacher_500._sample_decoded_grid", fake_sample)
    monkeypatch.setattr("backend.scripts.again_sparse_tribe_teacher_500.MlxVjepa21FeatureModel", FakeModel)
    monkeypatch.setattr("backend.scripts.again_sparse_tribe_teacher_500.MlxTribeEncoder", FakeTribe)

    features, runtime_rows, summary = encode_sparse_windows(
        queue,
        external_cache_root=tmp_path,
        vjepa_weights_dir=tmp_path / "vjepa",
        tribe_model_dir=tmp_path / "tribe",
        vjepa_sha256="vjepa",
        tribe_sha256="tribe",
        microbatch_size=2,
        ready_queue_max_size=4,
    )

    assert calls["forward_batches"] == [2]
    assert len(features) == 2
    assert summary["microbatch_size"] == 2
    assert {row["microbatch_size"] for row in runtime_rows} == {2}
    assert all(float(row["forward_batch_seconds"]) >= float(row["forward_seconds"]) for row in runtime_rows)


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


def test_sparse_teacher_delta_over_ar_gate_compares_arm_local_lifts():
    gates = add_gate_rows(
        [
            {
                "selector_arm": "hybrid_top5_selected",
                "model_lane": "AR_only",
                "mean_pr_auc": 0.20,
            },
            {
                "selector_arm": "hybrid_top5_selected",
                "model_lane": "AR_plus_sparse_pca32_causal_past2s_mean",
                "mean_pr_auc": 0.30,
            },
            {
                "selector_arm": "coverage_matched_random_to_hybrid",
                "model_lane": "AR_only",
                "mean_pr_auc": 0.50,
            },
            {
                "selector_arm": "coverage_matched_random_to_hybrid",
                "model_lane": "AR_plus_sparse_pca32_causal_past2s_mean",
                "mean_pr_auc": 0.55,
            },
        ]
    )

    gate = next(row for row in gates if row["gate"] == "pca32_locked_delta_over_ar_vs_coverage_random_delta_over_ar")
    assert gate["pass"] is True
    assert abs(gate["lhs_delta_over_ar"] - 0.10) < 1e-9
    assert abs(gate["rhs_delta_over_ar"] - 0.05) < 1e-9
    assert abs(gate["mean_pr_auc_delta"] - 0.05) < 1e-9


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
    assert "Reported PCA-width rows are AR + sparse PCA lanes, not PCA-only lanes." in report
    assert "Train-selected PCA widths by grouped outer fold: `16,32`" in report
    assert "Hybrid AR + raw sparse causal mean PR-AUC: `11.00%`" in report
    assert "AR + sparse PCA16: PR-AUC `13.00%`" in report
