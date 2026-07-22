from __future__ import annotations

from pathlib import Path

import numpy as np

from neural_bridge.veatic21.contracts import FeatureRows
from neural_bridge.veatic21.evidence import digest_json
from neural_bridge.veatic21.pca_cache import fit_event_pca_cache


def test_pca_cache_fits_once_and_reuses_verified_payload(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    videos = np.repeat(np.asarray(["0", "1", "2", "3", "4"]), 6)
    rows = np.tile(np.arange(6, dtype=np.int32), 5)
    features = FeatureRows(
        video_id=videos,
        row_index=rows,
        time_seconds=rows.astype(np.float32) / 2,
        quality_eligible=np.ones(30, dtype=np.bool_),
        representations={"tribe_cortical": rng.normal(size=(30, 8)).astype(np.float32)},
    )
    split = {
        "video_ids": ["0", "1", "2", "3", "4"],
        "per_video_boundaries": [
            {
                "video_id": video,
                "last_train_row_index": 5,
                "first_test_row_index": 6,
            }
            for video in ["0", "1", "2", "3", "4"]
        ],
        "inner_grouped_video_folds": [[video] for video in ["0", "1", "2", "3", "4"]],
        "split_sha256": "split",
    }
    preregistration = {
        "schema": "veatic21_event_preregistration_v12",
        "preregistration_sha256": "preregistration",
        "substrate": {"identity": "fixture"},
        "split": split,
        "representations": {
            "pca": {
                "maximum_components": 3,
                "batch_rows": "2_x_maximum_components_balanced_without_short_final_batch",
                "scaler": "featurewise_standard_scaler_fit_on_exact_training_rows",
                "solver": "deterministic_incremental_pca",
                "fixed_width_candidates": [1, 2, 3],
                "variance_targets": [0.5, 0.9],
            }
        },
    }
    preregistration["preregistration_sha256"] = digest_json(preregistration)

    first = fit_event_pca_cache(features, preregistration, tmp_path, folds=[0])
    second = fit_event_pca_cache(features, preregistration, tmp_path, folds=[0])
    preregistration["unrelated_head_change"] = True
    preregistration["preregistration_sha256"] = digest_json(preregistration)
    third = fit_event_pca_cache(features, preregistration, tmp_path, folds=[0])

    assert first["label_values_accessed"] is False
    assert first["benchmark_rows"] == 30
    assert first["folds"][0]["candidate_widths"][:3] == [1, 2, 3]
    assert first["folds"][0]["cache_hit"] is False
    assert second["folds"][0]["cache_hit"] is True
    assert third["folds"][0]["cache_hit"] is True
    assert third["folds"][0]["directory"] == first["folds"][0]["directory"]
    projection = np.load(
        tmp_path / second["folds"][0]["directory"] / "benchmark-train-projection.npy",
        mmap_mode="r",
    )
    assert projection.shape == (30, 3)
    assert np.isfinite(projection).all()
