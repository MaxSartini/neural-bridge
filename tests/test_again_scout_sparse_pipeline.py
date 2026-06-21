import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.scripts.again_scout_sparse_pipeline import (
    SelectorConfig,
    ScoutModelSpec,
    build_sparse_teacher_queue,
    control_contract_rows,
    default_selector_configs,
    event_mask_contract_rows,
    group_clean_annotations,
    merge_candidate_regions,
    model_registry,
    select_candidate_timestamps,
    split_contract_rows,
    telemetry_change_feature_rows,
)
from backend.scripts.again_vjepa21_scout import (
    embedding_delta,
    pool_scout_tokens,
    require_vjepa21_scout,
    scout_window_fingerprint,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_telemetry_change_features_exclude_label_fields(tmp_path):
    metadata = tmp_path / "metadata.csv"
    clean = tmp_path / "clean.csv"
    write_csv(
        metadata,
        [
            {
                "[control]player_id": "p1",
                "[control]game": "Endless",
                "[control]session_id": "s1",
                "video_name": "p1_endless_s1.webm",
            }
        ],
    )
    write_csv(
        clean,
        [
            {
                "[control]player_id": "p1",
                "[control]game": "Endless",
                "[control]session_id": "s1",
                "[control]time_stamp": "0",
                "[control]genre": "Platformer",
                "[output]arousal": "0.1",
                "[general]input_intensity": "0.0",
                "[general]input_diversity": "0.0",
                "[general]activity": "0.0",
                "[general]score": "0.0",
                "key_press_count": "0",
            },
            {
                "[control]player_id": "p1",
                "[control]game": "Endless",
                "[control]session_id": "s1",
                "[control]time_stamp": "1",
                "[control]genre": "Platformer",
                "[output]arousal": "0.9",
                "[general]input_intensity": "1.0",
                "[general]input_diversity": "1.0",
                "[general]activity": "1.0",
                "[general]score": "5.0",
                "key_press_count": "3",
            },
        ],
    )
    manifest_rows = [
        {
            "video_id": "p1_endless_s1",
            "video_path": str(tmp_path / "p1_endless_s1.webm"),
            "participant_id": "p1",
            "session_id": "s1",
            "game": "Endless",
            "genre": "Platformer",
            "time_start_seconds": str(time_s),
            "alignment_policy": "use_annotation_covered_video_time_only",
            "arousal": "0.5",
            "future_spike_1_3s_ge_0.05": "true",
        }
        for time_s in (0, 1)
    ]

    grouped = group_clean_annotations(clean, metadata)
    rows = telemetry_change_feature_rows(manifest_rows=manifest_rows, annotations_by_video=grouped)

    assert len(rows) == 2
    assert rows[1]["idle_to_active_transition"] == 1
    assert rows[1]["telemetry_change_score"] > rows[0]["telemetry_change_score"]
    assert "arousal" not in rows[0]
    assert "future_spike_1_3s_ge_0.05" not in rows[0]
    assert rows[0]["label_fields_excluded_from_selector"] is True


def test_deployable_selector_rejects_label_fields():
    rows = [{"video_id": "v1", "time_start_seconds": 0, "arousal": 0.7}]
    config = SelectorConfig("bad", {"arousal": 1.0}, top_percent=1.0, oracle=False)

    with pytest.raises(ValueError, match="uses label fields"):
        select_candidate_timestamps(rows, [config])


def test_candidate_selection_adds_controls_and_merges_regions():
    rows = [
        {
            "video_id": "v1",
            "video_path": "/tmp/v1.webm",
            "time_start_seconds": i,
            "telemetry_change_z": float(i),
            "cheap_video_audio_z": 0.0,
            "vjepa_b_novelty_z": 0.0,
            "game": "g",
        }
        for i in range(10)
    ]
    selected = select_candidate_timestamps(
        rows,
        [SelectorConfig("telemetry_top20", {"telemetry_change_z": 1.0}, top_percent=0.2, include_random_negatives=2)],
        seed=1,
    )
    sources = {row["selection_source"] for row in selected}
    regions = merge_candidate_regions(selected)

    assert {"candidate", "random_low_salience_negative", "sparse_anchor"} <= sources
    assert regions
    assert all(row["clip_start_seconds"] >= 0 for row in regions)


def test_sparse_teacher_queue_uses_strict_fingerprint_and_again_cache(tmp_path):
    teacher = ScoutModelSpec(
        name="vjepa21_vitg_tribe_sparse_teacher",
        vjepa_version="2.1",
        backend_repo="lukasugar/vjepa2.1-mlx",
        model_role="sparse_expensive_teacher",
        model_name="vjepa2_1_vitg_384",
        mlx_path="/models/vjepa21_mlx/vitg",
        source_checkpoint_path="",
        checkpoint_sha256="abc123",
        hidden_size=1408,
        num_layers=40,
        num_heads=22,
        image_size=256,
        frames_per_clip=64,
        status="ready",
    )
    regions = [
        {
            "selector_name": "telemetry_video_audio_vjepa_b_top5",
            "video_id": "v1",
            "video_path": "/tmp/v1.webm",
            "clip_start_seconds": 1.0,
            "clip_end_seconds": 5.0,
            "max_selector_score": 2.0,
            "game": "g",
        }
    ]
    queue = build_sparse_teacher_queue(
        regions,
        [],
        teacher_model=teacher,
        output_external_root=tmp_path / "benchmarks" / "again" / "sparse",
    )
    assert queue == []

    from backend.scripts.again_scout_sparse_pipeline import default_sparse_teacher_budgets

    queue = build_sparse_teacher_queue(
        regions,
        default_sparse_teacher_budgets()[:1],
        teacher_model=teacher,
        output_external_root=tmp_path / "benchmarks" / "again" / "sparse",
    )

    assert len(queue) == 1
    assert queue[0]["strict_fingerprint"]
    assert "/again/" in queue[0]["cache_dir"]
    assert queue[0]["force_required_to_recompute"] is True


def test_control_and_split_contracts_include_veatic_grade_controls():
    controls = {row["control_name"] for row in control_contract_rows()}
    splits = {row["split_name"] for row in split_contract_rows()}
    masks = {row["event_mask"] for row in event_mask_contract_rows()}

    assert "shuffled_tribe_rows" in controls
    assert "random_gaussian_same_dim" in controls
    assert "telemetry_change_only" in controls
    assert "oracle_spike_selector_upper_bound" in controls
    assert "leave_video_out_grouped_kfold" in splits
    assert "leave_game_out" in splits
    assert "balanced_event_vs_stable_1_to_5" in masks


def test_model_registry_is_vjepa21_only(tmp_path):
    registry = model_registry(tmp_path)
    payload = [row.__dict__ for row in registry]

    assert all(row["vjepa_version"] == "2.1" for row in payload)
    assert all("V-JEPA2-vitl-fpc64-256" not in json.dumps(row) for row in payload)
    assert {row["model_role"] for row in payload} == {
        "primary_scout",
        "fallback_scout",
        "sparse_expensive_teacher",
    }


def test_scout_helpers_pool_embeddings_and_fingerprint():
    spec = ScoutModelSpec(
        name="vjepa21_vitb_lukasugar_mlx_scout",
        vjepa_version="2.1",
        backend_repo="lukasugar/vjepa2.1-mlx",
        model_role="primary_scout",
        model_name="vjepa2_1_vit_base_384",
        mlx_path="/tmp/model.safetensors",
        source_checkpoint_path="",
        checkpoint_sha256="abc",
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        image_size=384,
        frames_per_clip=64,
        status="ready",
    )
    pooled = pool_scout_tokens(np.ones((2, 4, 3), dtype=np.float32))
    fp = scout_window_fingerprint(
        spec=spec,
        video_id="v1",
        clip_start_seconds=1.0,
        clip_end_seconds=5.0,
        frame_count=16,
        resolution=384,
        stride_seconds=4.0,
    )

    assert pooled.shape == (2, 3)
    assert embedding_delta(pooled[0], pooled[1]) == 0.0
    assert len(fp.digest()) == 64


def test_scout_helper_rejects_non_vjepa21():
    bad = ScoutModelSpec(
        name="old",
        vjepa_version="2.0",
        backend_repo="mlx-community",
        model_role="primary_scout",
        model_name="vjepa2",
        mlx_path="/tmp/old",
        source_checkpoint_path="",
        checkpoint_sha256="",
        hidden_size=1,
        num_layers=1,
        num_heads=1,
        image_size=256,
        frames_per_clip=64,
        status="ready",
    )

    with pytest.raises(ValueError, match="Only V-JEPA 2.1"):
        require_vjepa21_scout(bad)
