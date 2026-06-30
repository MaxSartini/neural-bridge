import numpy as np

from backend.scripts import run_veatic_raw_representation_audit as audit
from backend.scripts import veatic_representation_builders as reps


def _rows(video_count=2, rows_per_video=4):
    rows = []
    for video_id in range(video_count):
        for second in range(rows_per_video):
            rows.append(
                {
                    "video_id": str(video_id),
                    "time_start_seconds": float(second),
                    "frame_index": second,
                    "targets": {"arousal": float(second) / 10.0},
                    "splits": {"blocked_temporal_gap": "train" if video_id == 0 else "test"},
                }
            )
    return rows


def test_audit_plan_keeps_v2_guardrails():
    parser = audit.build_parser()
    args = parser.parse_args(["--dry-run", "--primary-audit"])
    plan = audit.audit_plan(args)

    assert plan["schema_version"] == "veatic_raw_representation_audit_plan_v1"
    assert plan["no_reencode"] is True
    assert "cortical_pca64_delta" in {row["name"] for row in plan["candidates"]}
    assert "raw_current_ridge" in {row["name"] for row in plan["candidates"]}
    assert "grouped_video_5_fold" in plan["splits"]
    assert "fit representations on train rows only" in plan["leakage_rules"]
    assert plan["sensitivity"] == ["all_videos"]
    assert all(not row["uses_future_features"] for row in plan["candidates"])


def test_audit_plan_makes_video_83_sensitivity_opt_in():
    parser = audit.build_parser()
    default_plan = audit.audit_plan(parser.parse_args(["--dry-run", "--primary-audit"]))
    sensitivity_plan = audit.audit_plan(parser.parse_args(["--dry-run", "--primary-audit", "--with-sensitivity"]))

    assert default_plan["sensitivity"] == ["all_videos"]
    assert sensitivity_plan["sensitivity"] == ["all_videos", "exclude_video_83"]


def test_checkpoint_state_round_trips_completed_jobs(tmp_path):
    config = {"mode": "smoke", "skip_sensitivity": True}
    state = audit.empty_checkpoint_state(config)
    key = audit.job_key(
        scope="all_videos",
        split="blocked",
        candidate="cortical_pca64_delta",
        head="ridge_score",
        target="arousal__future_spike_1_3s",
        threshold=0.05,
        task_type="binary",
    )
    state["job_status"][key] = "complete"
    state["result_rows"].append({"scope": "all_videos", "split": "blocked", "candidate": "cortical_pca64_delta"})

    audit.save_checkpoint_state(tmp_path, state)
    loaded = audit.load_checkpoint_state(tmp_path)

    assert loaded["config"] == config
    assert loaded["job_status"][key] == "complete"
    assert loaded["result_rows"][0]["scope"] == "all_videos"


def test_job_seed_is_stable_by_job_key():
    key = audit.job_key(
        scope="all_videos",
        split="grouped_0",
        candidate="pca_delta_128",
        head="ridge_score",
        target="arousal__future_spike_1_3s",
        threshold=0.075,
        task_type="binary",
    )

    assert audit.job_seed(43, key) == audit.job_seed(43, key)
    assert audit.job_seed(44, key) != audit.job_seed(43, key)


def test_pca_current_builder_records_train_only_fit_scope(monkeypatch):
    rows = _rows(video_count=2, rows_per_video=3)
    raw = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    raw[3:] += 1000.0
    base = {"cortical_raw": raw, "cortical_global": raw[:, :2]}

    def fake_pca(train_x, apply_x, components):
        assert train_x.shape == (3, 4)
        assert np.max(train_x) < 1000.0
        return apply_x[:, :components].astype(np.float32), {
            "backend": "test",
            "requested_components": components,
            "actual_components": components,
        }

    monkeypatch.setattr(reps.bench, "pca_fit_transform", fake_pca)
    fitted = reps.PcaCurrentBuilder(2).fit(rows[:3], np.arange(3), rows, base)
    transformed = fitted.transform(rows[3:], np.arange(3, 6))

    assert transformed.values.shape == (3, 2)
    meta = fitted.metadata()
    assert meta["fit_scope"] == "train_rows_only"
    assert meta["train_row_count"] == 3
    assert meta["leakage_contract"]["fit_on_train_rows_only"] is True
    assert meta["leakage_contract"]["test_labels_used_for_fit"] is False


def test_pca_projection_is_reused_across_same_split_candidates(monkeypatch):
    rows = _rows(video_count=2, rows_per_video=4)
    raw = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    base = {"cortical_raw": raw, "cortical_global": raw[:, :2]}
    calls = {"count": 0}

    def fake_pca(train_x, apply_x, components):
        calls["count"] += 1
        return apply_x[:, :components].astype(np.float32), {
            "backend": "test",
            "requested_components": components,
            "actual_components": components,
        }

    monkeypatch.setattr(reps.bench, "pca_fit_transform", fake_pca)
    shared = {"fit_cache": {}}
    train_idx = np.arange(4)

    current = reps.PcaCurrentBuilder(2).fit(rows[:4], train_idx, rows, base, inner_validation=shared)
    delta = reps.PcaDeltaBuilder(2).fit(rows[:4], train_idx, rows, base, inner_validation=shared)
    sequence = reps.PcaSequenceBuilder(2, 2.0, "mean").fit(rows[:4], train_idx, rows, base, inner_validation=shared)

    assert calls["count"] == 1
    assert current.metadata()["pca"]["cache_hit"] is False
    assert delta.metadata()["pca"]["cache_hit"] is True
    assert sequence.metadata()["pca"]["cache_hit"] is True


def test_pca_projection_disk_cache_reuses_after_memory_cache_is_gone(monkeypatch, tmp_path):
    rows = _rows(video_count=2, rows_per_video=4)
    raw = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
    base = {"cortical_raw": raw, "cortical_global": raw[:, :2]}
    calls = {"count": 0}

    def fake_pca(train_x, apply_x, components):
        calls["count"] += 1
        return apply_x[:, :components].astype(np.float32), {
            "backend": "test",
            "requested_components": components,
            "actual_components": components,
        }

    monkeypatch.setattr(reps.bench, "pca_fit_transform", fake_pca)
    train_idx = np.arange(4)
    first_context = {"fit_cache": {}, "fit_cache_dir": tmp_path}
    second_context = {"fit_cache": {}, "fit_cache_dir": tmp_path}

    first = reps.PcaCurrentBuilder(2).fit(rows[:4], train_idx, rows, base, inner_validation=first_context)
    second = reps.PcaDeltaBuilder(2).fit(rows[:4], train_idx, rows, base, inner_validation=second_context)

    assert calls["count"] == 1
    assert first.metadata()["pca"]["cache_hit"] is False
    assert second.metadata()["pca"]["cache_hit"] is True
    assert second.metadata()["pca"]["disk_cache_hit"] is True


def test_causal_window_drops_first_rows_without_crossing_videos():
    rows = _rows(video_count=2, rows_per_video=3)
    raw = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    base = {"cortical_raw": raw, "cortical_global": raw}
    fitted = reps.RawCausalMeanBuilder(2.0).fit(rows[:3], np.arange(3), rows, base)
    transformed = fitted.transform(rows, np.arange(6))

    kept_keys = [(row["video_id"], row["time_start_seconds"]) for row in transformed.rows]
    assert ("0", 0.0) not in kept_keys
    assert ("1", 0.0) not in kept_keys
    assert kept_keys == [("0", 2.0), ("1", 2.0)]
    assert transformed.values.shape == (2, 2)
