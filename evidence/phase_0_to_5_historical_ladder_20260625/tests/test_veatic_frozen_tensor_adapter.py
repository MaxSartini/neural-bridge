import csv
import json
from pathlib import Path

import numpy as np
import pytest

from backend.scripts import veatic_frozen_tensor_adapter as adapter
from backend.scripts import run_veatic_frozen_tensor_incremental_benchmark as wrapper


TARGETS = (
    ("arousal__future_spike_1_3s", 0.05),
    ("arousal__future_spike_1_3s", 0.075),
    ("arousal__future_change_p3s_movement", 0.05),
)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(video_id: str, index: int, split: str, role: str) -> dict:
    return {
        "video_id": video_id,
        "time_start_seconds": float(index),
        "frame_index": index,
        "global_row_index": index,
        "is_video_83": video_id == "83",
        "split_name": split,
        "split_role": role,
        "used_future_feature_rows": False,
        "y_value": 1.0 if index % 2 else 0.0,
    }


def _make_contract_tree(tmp_path: Path, *, omit_video_83_for: tuple[str, str, str] | None = None):
    tensor_root = tmp_path / "external" / "tensors" / "veatic_124_raw_representation_v1"
    summary_root = tmp_path / "repo" / "outputs" / "veatic_124_raw_representation_tensor_export_v1"
    summary = {
        "verification_status": "pass",
        "representations_exported": [
            "pca_sequence_128_causal_past_2s_mean",
            "roi_parcel_features",
            "topk_vertices_512",
            "cortical_pca64_delta_frozen_baseline",
        ],
        "safe_for_primary_training": [
            "pca_sequence_128_causal_past_2s_mean",
            "roi_parcel_features",
            "cortical_pca64_delta_frozen_baseline",
        ],
        "total_contracts_expected": 84,
        "total_contracts_exported": 84,
        "total_npy_files_exported": 420,
        "video_83_included": True,
        "exclude_video_83_sensitivity_exported": False,
        "targets_exported": [
            {"target_name": name, "task_type": "binary", "threshold": threshold}
            for name, threshold in TARGETS
        ],
        "splits_exported": list(adapter.SPLITS),
    }
    _write_json(summary_root / "tensor_export_summary.json", summary)
    _write_json(summary_root / "tensor_export_verification.json", {"status": "pass", "checks": [], "failures": [], "warnings": []})
    with (summary_root / "tensor_export_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename"])
        writer.writeheader()
        writer.writerow({"filename": "placeholder"})

    for representation in adapter.REQUIRED_REPRESENTATIONS:
        for split in adapter.SPLITS:
            for target_name, threshold in TARGETS:
                folder = adapter.target_dir_name(target_name, threshold)
                rel = Path(representation) / split / folder
                external = tensor_root / rel
                tracked = summary_root / rel
                external.mkdir(parents=True, exist_ok=True)
                tracked.mkdir(parents=True, exist_ok=True)

                np.save(external / "X_train.npy", np.arange(6, dtype=np.float32).reshape(2, 3))
                np.save(external / "X_test.npy", np.arange(6, 12, dtype=np.float32).reshape(2, 3))
                np.save(external / "y_train.npy", np.asarray([0.0, 1.0], dtype=np.float32))
                np.save(external / "y_test.npy", np.asarray([1.0, 0.0], dtype=np.float32))
                if representation == adapter.PRIMARY_CANDIDATE:
                    np.save(external / "X_sequence_train.npy", np.zeros((2, 3, 3), dtype=np.float32))
                    np.save(external / "X_sequence_test.npy", np.zeros((2, 3, 3), dtype=np.float32))
                    np.save(external / "sequence_mask_train.npy", np.ones((2, 3), dtype=np.float32))
                    np.save(external / "sequence_mask_test.npy", np.ones((2, 3), dtype=np.float32))

                omit = omit_video_83_for == (representation, split, folder)
                train_rows = [_row("1", 1, split, "train"), _row("3", 3, split, "train")]
                test_rows = [_row("2", 2, split, "test")]
                test_rows.append(_row("4" if omit else "83", 4 if omit else 83, split, "test"))
                _write_jsonl(external / "row_metadata_train.jsonl", train_rows)
                _write_jsonl(external / "row_metadata_test.jsonl", test_rows)
                _write_json(
                    tracked / "representation_metadata.json",
                    {
                        "representation_name": representation,
                        "video_83_included": True,
                        "uses_future_features": False,
                        "fit_scope": "train_rows_only",
                        "transform_scope": "train_fit_applied_to_test",
                        "pca": {
                            "pca_fit_scope": "train_rows_only",
                            "cache_rebuilt": False,
                        },
                        "dropped_rows": {"reason_counts": {"missing_full_causal_history": 1}},
                    },
                )
                _write_json(tracked / "split_metadata.json", {"split_name": split, "train_test_video_overlap": []})
                _write_json(tracked / "target_metadata.json", {"target_name": target_name, "threshold": threshold, "task_type": "binary"})
                _write_json(tracked / "checksum_manifest.json", {"files": []})
                _write_json(tracked / "leakage_contract.json", {"no_future_rows_in_primary": True})
    return tensor_root, summary_root


def test_dry_run_plan_uses_existing_suite_with_frozen_tensor_adapter(tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)

    plan = provider.dry_run_plan(fresh_run_id="test-fresh-run")

    assert plan["preflight_status"] == "pass"
    assert plan["benchmark_mode"] == "existing_suite_with_frozen_tensor_adapter"
    assert plan["reused_existing_benchmark_structure"] is True
    assert plan["reused_frozen_tensors"] is True
    assert plan["reused_canonical_controls"] is True
    assert plan["reused_benchmark_result_rows"] is False
    assert plan["computed_ar_fresh"] is True
    assert plan["computed_controls_fresh"] is True
    assert "autoregressive_features" in plan["canonical_helpers"]
    assert "event_metrics" in plan["event_metric_helpers_found"]
    assert "topk_recall" in plan["event_metric_helpers_found"]
    assert "autoregressive_plus_PCA128" in plan["existing_suite_prediction_keys"]["PCA128"]
    assert "residualized_autoregressive_plus_PCA64_delta" in plan["existing_suite_prediction_keys"]["PCA64_delta"]
    ledger = plan["freshness_ledger"]
    assert ledger["fresh_run_id"] == "test-fresh-run"
    assert ledger["benchmark_contract_version"] == adapter.BENCHMARK_CONTRACT_VERSION
    assert ledger["computed_scores"] is False
    assert ledger["fit_models"] is False
    assert ledger["wrote_result_csvs"] is False


def test_provider_loads_contract_as_existing_benchmark_inputs(tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)
    contract = provider.load_contract(
        representation=adapter.PRIMARY_CANDIDATE,
        split="grouped_0",
        target_name="arousal__future_spike_1_3s",
        threshold=0.05,
    )
    all_rows = [
        {"video_id": str(index), "time_start_seconds": float(index), "frame_index": index}
        for index in range(84)
    ]

    inputs = contract.as_existing_benchmark_inputs(all_rows)

    assert inputs.feature_name == "PCA128"
    assert inputs.train_x.shape == (2, 3)
    assert inputs.test_x.shape == (2, 3)
    assert inputs.train_y.tolist() == [0.0, 1.0]
    assert inputs.test_y.tolist() == [1.0, 0.0]
    assert [row["video_id"] for row in inputs.train_rows] == ["1", "3"]
    assert [row["video_id"] for row in inputs.test_rows] == ["2", "83"]
    assert "shuffled_PCA128" in inputs.prediction_keys
    assert "autoregressive_plus_random_gaussian_PCA128" in inputs.prediction_keys
    assert "residualized_autoregressive_plus_shuffled_PCA128" in inputs.prediction_keys


def test_same_row_order_assertion_stops_on_mismatch_without_sorting():
    ar_rows = [
        {"video_id": "2", "time_start_seconds": 2.0, "frame_index": 2},
        {"video_id": "1", "time_start_seconds": 1.0, "frame_index": 1},
    ]
    tensor_rows = [
        {"video_id": "1", "time_start_seconds": 1.0, "frame_index": 1},
        {"video_id": "2", "time_start_seconds": 2.0, "frame_index": 2},
    ]

    with pytest.raises(AssertionError, match="row order mismatch"):
        adapter.assert_same_row_order(ar_rows, tensor_rows, role="train")


def test_rows_from_metadata_order_uses_tensor_order_key_lookup_without_sorting():
    all_rows = [
        {"video_id": "2", "time_start_seconds": 2.0, "frame_index": 2},
        {"video_id": "1", "time_start_seconds": 1.0, "frame_index": 1},
    ]
    tensor_rows = [
        {"global_row_index": 99, "video_id": "1", "time_start_seconds": 1.0, "frame_index": 1},
        {"global_row_index": 98, "video_id": "2", "time_start_seconds": 2.0, "frame_index": 2},
    ]

    rows = adapter.rows_from_metadata_order(all_rows, tensor_rows, role="test")

    assert [row["video_id"] for row in rows] == ["1", "2"]


def test_rows_from_metadata_order_rejects_missing_manifest_key():
    all_rows = [
        {"video_id": "2", "time_start_seconds": 2.0, "frame_index": 2},
    ]
    tensor_rows = [
        {"video_id": "1", "time_start_seconds": 1.0, "frame_index": 1},
    ]

    with pytest.raises(AssertionError, match="tensor row key missing"):
        adapter.rows_from_metadata_order(all_rows, tensor_rows, role="test")


def test_video_83_absence_is_recorded_not_silently_ignored(tmp_path):
    omit = (
        adapter.PRIMARY_CANDIDATE,
        "grouped_0",
        adapter.target_dir_name("arousal__future_spike_1_3s", 0.05),
    )
    tensor_root, summary_root = _make_contract_tree(tmp_path, omit_video_83_for=omit)
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)

    plan = provider.dry_run_plan()
    recorded = [
        check
        for check in plan["preflight_checks"]
        if check["check_name"] == "video_83_not_deliberately_excluded"
        and check["details"].get("contract") == str(Path(*omit))
    ]

    assert recorded
    assert recorded[0]["status"] == "recorded_absent_after_target_horizon_trimming"
    assert "No eligible video 83 rows" in recorded[0]["details"]["reason"]
    assert plan["preflight_status"] == "pass"


def test_exclude_83_paths_are_rejected(tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)

    with pytest.raises(ValueError, match="exclude-video-83"):
        adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root / "exclude_video_83", summary_root=summary_root)


def test_wrapper_is_thin_and_exposes_ridge_only_args():
    args = wrapper.build_parser().parse_args([])
    assert args.dry_run is False
    assert set(vars(args)) == {"manifest", "tensor_root", "summary_root", "output_dir", "fresh_run_id", "seed", "dry_run"}


def _all_rows():
    return [
        {
            "video_id": str(index),
            "time_start_seconds": float(index),
            "frame_index": index,
            "targets": {"arousal": float(index) / 100.0},
        }
        for index in range(84)
    ]


def test_ridge_scoring_calls_existing_semantics_and_computes_required_lanes(monkeypatch, tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)
    contract = provider.load_contract(
        representation=adapter.PRIMARY_CANDIDATE,
        split="grouped_0",
        target_name="arousal__future_spike_1_3s",
        threshold=0.05,
    )
    original_ridge = adapter.bench.ridge_fit_predict
    original_ar = adapter.bench.autoregressive_features
    ridge_calls = []
    ar_calls = []

    def counting_ridge(train_x, train_y, test_x, alpha=1.0):
        ridge_calls.append((train_x.shape, test_x.shape))
        return original_ridge(train_x, train_y, test_x, alpha=alpha)

    def counting_ar(all_rows, rows, target, *, include_current):
        ar_calls.append({"rows": len(rows), "target": target, "include_current": include_current})
        return original_ar(all_rows, rows, target, include_current=include_current)

    monkeypatch.setattr(adapter.bench, "ridge_fit_predict", counting_ridge)
    monkeypatch.setattr(adapter.bench, "autoregressive_features", counting_ar)

    lane_rows, leakage_checks = adapter.score_contract_ridge_existing_semantics(
        contract,
        all_rows=_all_rows(),
        seed=7,
    )

    lanes = {row["lane"] for row in lane_rows}
    assert "AR_only" in lanes
    assert "PCA128_only" in lanes
    assert "AR_plus_PCA128" in lanes
    assert "residualized_AR_plus_PCA128" in lanes
    assert "shuffled_PCA128" in lanes
    assert "AR_plus_random_PCA128" in lanes
    assert "residualized_AR_plus_shuffled_PCA128" in lanes
    assert any(check["check_name"] == "same_row_order_before_ar_scoring" for check in leakage_checks)
    assert len(ar_calls) == 2
    assert all(call["target"] == "arousal" and call["include_current"] is True for call in ar_calls)
    assert ridge_calls
    assert any(train_shape[1] > contract.train_x.shape[1] for train_shape, _test_shape in ridge_calls)


def test_score_ridge_only_writes_requested_artifacts_and_freshness_ledger(tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)
    output_dir = tmp_path / "ridge_outputs"

    manifest = provider.score_ridge_only(
        all_rows=_all_rows(),
        output_dir=output_dir,
        fresh_run_id="unit-ridge",
        seed=11,
    )

    expected_files = {
        "lane_results.csv",
        "fold_results.csv",
        "control_results.csv",
        "freshness_ledger.json",
        "leakage_checks.json",
        "gate_checks.json",
        "run_manifest.json",
        "ridge_only_report.md",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    assert manifest["benchmark_mode"] == "existing_suite_with_frozen_tensor_adapter"
    assert manifest["computed_scores"] is True
    assert manifest["reused_benchmark_result_rows"] is False
    assert manifest["computed_ar_fresh"] is True
    assert manifest["computed_controls_fresh"] is True
    assert manifest["full_veatic_124"] is True
    assert manifest["video_83_included"] is True
    assert manifest["exclude_video_83_run"] is False
    lane_rows = (output_dir / "lane_results.csv").read_text(encoding="utf-8")
    assert "AR_plus_PCA128" in lane_rows
    assert "residualized_AR_plus_PCA128" in lane_rows
    assert "AR_plus_PCA64_delta" in lane_rows
    assert "residualized_AR_plus_PCA64_delta" in lane_rows
    ledger = json.loads((output_dir / "freshness_ledger.json").read_text(encoding="utf-8"))
    assert ledger["computed_scores"] is True
    assert ledger["computed_ar_fresh"] is True
    assert ledger["computed_controls_fresh"] is True


def test_prior_result_csvs_are_denied_before_scoring(tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)
    (summary_root / "control_results.csv").write_text("old,row\n", encoding="utf-8")
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)

    with pytest.raises(ValueError, match="Prior benchmark result inputs are forbidden"):
        provider.score_ridge_only(
            all_rows=_all_rows(),
            output_dir=tmp_path / "out",
            fresh_run_id="blocked",
        )


def test_full_veatic_policy_is_enforced_before_scoring(tmp_path):
    tensor_root, summary_root = _make_contract_tree(tmp_path)
    summary_path = summary_root / "tensor_export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["exclude_video_83_sensitivity_exported"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    provider = adapter.FrozenTensorFeatureProvider(tensor_root=tensor_root, summary_root=summary_root)

    with pytest.raises(ValueError, match="Frozen tensor preflight failed"):
        provider.score_ridge_only(
            all_rows=_all_rows(),
            output_dir=tmp_path / "out",
            fresh_run_id="blocked",
        )
