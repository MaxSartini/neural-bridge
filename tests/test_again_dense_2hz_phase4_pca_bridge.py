from __future__ import annotations

import numpy as np
import pandas as pd

from backend.scripts.again_dense_2hz_benchmark import (
    AR_FEATURE_COLUMNS,
    QUALITY_FEATURE_COLUMNS,
    TIME_FEATURE_COLUMNS,
    TARGET_SPECS,
)
from backend.scripts.again_dense_2hz_phase4_pca_bridge import (
    CORTICAL_WIDTH,
    run_phase4,
)


def _fake_dense_root(tmp_path):
    root = tmp_path / "again_dense_cache"
    per_video = root / "per_video"
    per_video.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(7)
    for vid in range(6):
        video_id = f"video_{vid:02d}"
        video_dir = per_video / video_id
        video_dir.mkdir()
        n = 12
        times = np.arange(n, dtype=np.float32) * 0.5
        cortical = rng.normal(size=(n, CORTICAL_WIDTH)).astype(np.float16)
        np.savez(
            video_dir / "tribe_v2_cortical_predictions.npz",
            cortical_prediction=cortical,
            time_seconds=times,
        )
        np.savez(
            video_dir / "vjepa_temporal_diagnostics.npz",
            temporal_std_global=np.linspace(0.1, 0.2, n, dtype=np.float32),
            temporal_std_by_state=rng.normal(size=(n, 20)).astype(np.float16),
            temporal_std_by_state_token=rng.normal(size=(n, 20, 32)).astype(np.float16),
        )
        for row_index, time_seconds in enumerate(times):
            arousal = float((row_index % 6) / 6.0 + vid * 0.01)
            row = {
                "video_id": video_id,
                "row_index": row_index,
                "time_seconds": float(time_seconds),
                "row_rate_hz": 2.0,
                "label_available": True,
                "ar_context_available": row_index >= 4,
                "future_arousal_max_delta_rows_2_6": float(row_index >= 8) + 0.05 * vid,
                "future_arousal_delta_p2rows": float(row_index >= 7) + 0.05 * vid,
                "future_arousal_delta_p4rows": float(row_index >= 6) + 0.05 * vid,
                "target_mask_arousal_spike_rows_2_6": row_index >= 4,
                "target_mask_arousal_delta_p2rows": row_index >= 4,
                "target_mask_arousal_delta_p4rows": row_index >= 4,
                "arousal": arousal,
                "video_time_fraction": row_index / max(1, n - 1),
                "sin_time_10s": np.sin(float(time_seconds) / 10.0),
                "cos_time_10s": np.cos(float(time_seconds) / 10.0),
                "sin_time_30s": np.sin(float(time_seconds) / 30.0),
                "cos_time_30s": np.cos(float(time_seconds) / 30.0),
                "black_frame_fraction": 0.0,
                "duplicate_frame_fraction": 0.0,
                "quality_black_frame_flag": 0,
                "quality_duplicate_frame_flag": 0,
                "quality_exclusion_flag": 0,
                "quality_weight_suggested": 1.0,
                "motion_absdiff_mean": 0.1 + 0.01 * row_index,
                "luma_mean": 0.4,
                "luma_std": 0.05,
                "frame_luma_std_mean": 0.02,
            }
            for lag in (1, 2, 4):
                row[f"arousal_lag_{lag}row"] = arousal - 0.01 * lag
                row[f"arousal_delta_prev_{lag}row"] = 0.01 * lag
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in set(AR_FEATURE_COLUMNS + QUALITY_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS):
        assert col in df.columns
    for spec in TARGET_SPECS:
        assert spec.value_column in df.columns
        assert spec.mask_column in df.columns
    df.to_parquet(root / "labels_aligned_2hz.parquet", index=False)
    df[["video_id", "row_index", "time_seconds"]].to_parquet(root / "row_index.parquet", index=False)
    (root / "global_run_metadata.json").write_text('{"cache_only": true, "forbid_vjepa": true}\n')
    return root


def test_phase4_smoke_uses_train_only_pca_and_writes_required_outputs(tmp_path):
    dense_root = _fake_dense_root(tmp_path)
    output_root = tmp_path / "again_phase4_output"
    manifest = run_phase4(
        dense_root=dense_root,
        output_root=output_root,
        widths=(4,),
        feature_families=("current",),
        validation_protocols=("grouped_video",),
        n_splits=2,
        batch_size=8,
        oversampling=2,
        power_iterations=0,
        max_fit_count=1,
        report_dir=tmp_path / "reports",
    )
    assert manifest["pca_run"] is True
    assert manifest["vjepa_encoding_run"] is False
    assert (output_root / "run_manifest.json").exists()
    assert (output_root / "pca_feature_manifest.json").exists()
    assert (output_root / "metrics" / "phase4_fold_metrics.csv").exists()
    assert (output_root / "promotion" / "promotion_gates.json").exists()
    pca_manifest = pd.read_csv(output_root / "pca_feature_manifest.csv")
    assert set(pca_manifest["pca_width"]) == {4}
    assert pca_manifest["training_row_count"].min() > 0
