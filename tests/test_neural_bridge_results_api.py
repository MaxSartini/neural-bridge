from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.app import create_app


SEEDS = (20260721, 20260722, 20260723)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    results_root: Path,
    media_path: Path,
    *,
    analysis_id: str = "beat_test_bundle",
    input_video: str | None = None,
    row_count_override: int | None = None,
    ensemble_offset: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    bundle = results_root / analysis_id
    bundle.mkdir(parents=True)
    count = 20
    timestamps = np.arange(1, count + 1, dtype=np.float64) / 2.0
    scores = np.linspace(0.01, 0.20, count, dtype=np.float64)
    scores[9] = 0.50
    rank_order = np.argsort(scores, kind="stable")
    percentiles = np.empty(count, dtype=np.float64)
    percentiles[rank_order] = np.arange(1, count + 1, dtype=np.float64) / count
    candidates = np.zeros(count, dtype=bool)
    candidates[9] = True
    members = {
        str(SEEDS[0]): scores - 0.03,
        str(SEEDS[1]): scores,
        str(SEEDS[2]): scores + 0.03 + ensemble_offset,
    }
    predictions = bundle / "neural_bridge_predictions.csv"
    fieldnames = [
        "time_seconds",
        "future_arousal_movement_score",
        "within_video_percentile",
        "relative_top_5pct_spike_candidate",
        *[f"member_seed_{seed}" for seed in SEEDS],
    ]
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    "time_seconds": f"{timestamps[index]:.7f}",
                    "future_arousal_movement_score": f"{scores[index]:.9f}",
                    "within_video_percentile": f"{percentiles[index]:.7f}",
                    "relative_top_5pct_spike_candidate": str(bool(candidates[index])),
                    **{
                        f"member_seed_{seed}": f"{values[index]:.9f}"
                        for seed, values in members.items()
                    },
                }
            )
    manifest = {
        "schema_version": "beat_single_video_neural_bridge_mlx_v1",
        "created_at": "2026-07-15T16:06:56+00:00",
        "status": "complete",
        "scope": "experimental_cross_domain_unlabeled_video_inference",
        "validation_status": "not_external_validation_without_labels",
        "input_video": input_video or str(media_path),
        "input_sha256": _sha256(media_path),
        "duration_seconds": 10.25,
        "row_hz": 2.0,
        "row_count": row_count_override if row_count_override is not None else count,
        "modalities": ["video"],
        "vjepa_backend": "mlx_vjepa_2_1_vitg",
        "vjepa_weights": "/private/model/path/that/must/not/leak",
        "vjepa_weights_sha256": "d" * 64,
        "tribe_backend": "mlx_tribe_v2",
        "tribe_weights": "/private/tribe/path/that/must/not/leak",
        "neural_bridge_lane": "video_supervised_temporal",
        "neural_bridge_target": "future_arousal_max_delta_rows_4_10",
        "neural_bridge_interpretation": "relative future arousal-movement ranking 2-5 seconds ahead",
        "checkpoint_hashes": {str(seed): f"{index + 1}" * 64 for index, seed in enumerate(SEEDS)},
        "locked_prediction_reproduction": {
            "feature_width": 1340,
            "locked_rows": 73093,
            "train_rows": 170482,
            "passed": True,
            "per_seed": {
                str(seed): {"max_abs_error": 0.0, "mean_abs_error": 0.0, "passed": True}
                for seed in SEEDS
            },
        },
        "relative_spike_policy": (
            "within-video top 5 percent; provisional ranking marker, not calibrated threshold"
        ),
        "runtime_seconds": {"total": 12.5},
        "artifacts": {
            "predictions": str(predictions),
            "vjepa": "/private/cache/path/that/must/not/leak",
        },
    }
    (bundle / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    expected = {
        "timestamps": timestamps,
        "scores": scores,
        "percentiles": percentiles,
        "candidate_time": 5.0,
        "predictions": predictions,
    }
    return bundle, expected


@pytest.fixture()
def bridge_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    external_root = tmp_path / "external"
    default_outputs = external_root / "outputs"
    default_outputs.mkdir(parents=True)
    results_root = tmp_path / "registered-results"
    results_root.mkdir()
    media_root = tmp_path / "registered-media"
    media_root.mkdir()
    media_path = media_root / "clip.mp4"
    media_path.write_bytes(bytes(range(256)) * 24)
    bundle, expected = _write_bundle(results_root, media_path)

    monkeypatch.setenv("NEURAL_BRIDGE_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("NEURAL_BRIDGE_RESULTS_ROOTS", str(results_root))
    monkeypatch.setenv("NEURAL_BRIDGE_MEDIA_ROOTS", str(media_root))

    class TestConfig:
        TESTING = True
        DEBUG = False

    app = create_app(TestConfig)
    # Config loads the project .env with override=True; restore the isolated
    # request-time registry values after application creation.
    monkeypatch.setenv("NEURAL_BRIDGE_EXTERNAL_ROOT", str(external_root))
    monkeypatch.setenv("NEURAL_BRIDGE_RESULTS_ROOTS", str(results_root))
    monkeypatch.setenv("NEURAL_BRIDGE_MEDIA_ROOTS", str(media_root))
    return {
        "client": app.test_client(),
        "results_root": results_root,
        "media_path": media_path,
        "bundle": bundle,
        "expected": expected,
        "tmp_path": tmp_path,
    }


def _data(response) -> dict[str, Any]:
    payload = response.get_json()
    assert payload["success"] is True
    return payload["data"]


def test_catalog_lists_only_complete_bridge_bundles(bridge_api) -> None:
    client = bridge_api["client"]
    incomplete = bridge_api["results_root"] / "incomplete_bundle"
    incomplete.mkdir()
    (incomplete / "run_manifest.json").write_text("{}\n", encoding="utf-8")

    response = client.get("/api/neural-bridge/v1/analyses")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "neural_bridge.analysis_list.v1"
    items = payload["data"]["items"]
    assert [item["analysis_id"] for item in items] == ["beat_test_bundle"]
    item = items[0]
    assert item["source"] == {
        "kind": "external_experiment",
        "dataset_id": "BEAT",
        "labels_available": False,
    }
    assert item["grid"] == {
        "row_hz": 2.0,
        "row_count": 20,
        "start_seconds": 0.5,
        "end_seconds": 10.0,
    }
    assert item["inference"]["lane"] == "video_supervised_temporal"
    assert item["inference"]["observed_arousal_used_at_inference"] is False
    assert item["inference"]["modalities_used"] == ["video"]
    assert item["outputs"]["candidate_count"] == 1
    assert item["outputs"]["raw_cortical_diagnostics"] is False
    assert item["evidence"]["external_validity"] == "not_established"
    assert item["evidence"]["exact_value_calibrated"] is False


def test_timeline_is_exact_bridge_csv_with_true_2hz_grid(bridge_api) -> None:
    response = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/beat_test_bundle/timeline?include_members=true"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "neural_bridge.timeline.v1"
    timeline = payload["data"]
    np.testing.assert_allclose(
        timeline["grid"]["timestamps_seconds"], bridge_api["expected"]["timestamps"], atol=0.0
    )
    np.testing.assert_allclose(
        timeline["series"]["future_arousal_movement_score"]["values"],
        bridge_api["expected"]["scores"],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        timeline["series"]["within_video_percentile"]["values"],
        bridge_api["expected"]["percentiles"],
        atol=1e-7,
    )
    assert timeline["target"]["forecast_window_offset_seconds"] == [2.0, 5.0]
    assert set(timeline["diagnostics"]["member_scores"]) == {str(seed) for seed in SEEDS}
    member_matrix = np.column_stack(list(timeline["diagnostics"]["member_scores"].values()))
    np.testing.assert_allclose(
        member_matrix.mean(axis=1),
        timeline["series"]["future_arousal_movement_score"]["values"],
        atol=1e-6,
    )


def test_timeline_preserves_candidate_policy_and_context_quality(bridge_api) -> None:
    timeline = _data(
        bridge_api["client"].get(
            "/api/neural-bridge/v1/analyses/beat_test_bundle/timeline"
        )
    )

    assert "diagnostics" not in timeline
    assert timeline["events"]["policy"] == {
        "id": "artifact_within_video_top_5pct_v1",
        "percentile": 0.95,
        "provisional": True,
        "calibrated": False,
        "source_field": "relative_top_5pct_spike_candidate",
    }
    assert len(timeline["events"]["items"]) == 1
    event = timeline["events"]["items"][0]
    assert event["anchor_time_seconds"] == 5.0
    assert event["forecast_window_start_seconds"] == 7.0
    assert event["forecast_window_end_seconds"] == 10.0
    assert event["full_bridge_history_context"] is True
    assert event["full_upstream_window_context"] is True
    assert event["cold_start_context"] is False
    assert event["full_forecast_window_in_media"] is True
    quality = timeline["row_quality"]
    assert quality["bridge_history_rows_available"][:6] == [1, 2, 3, 4, 5, 5]
    assert quality["full_bridge_history_context"][:5] == [False, False, False, False, True]
    assert quality["full_upstream_window_context"][:8] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert timeline["context_windows"] == {
        "bridge_history_rows": 5,
        "bridge_history_span_seconds_including_current": 2.5,
        "upstream_vjepa_window_seconds": 4.0,
    }
    assert quality["full_forecast_window_in_media"][9] is True
    assert quality["full_forecast_window_in_media"][10] is False
    assert timeline["unsupported_outputs"]["arousal_dropoff"]["available"] is False
    assert timeline["unsupported_outputs"]["valence"]["available"] is False
    assert timeline["reference"]["available"] is False
    assert timeline["reference"]["channels"] == []


def test_upstream_window_controls_event_and_report_cold_start_flag(bridge_api) -> None:
    predictions = bridge_api["expected"]["predictions"]
    rows = list(csv.DictReader(predictions.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row["future_arousal_movement_score"] = "0.900000000" if row["time_seconds"] == "2.5000000" else row[
            "future_arousal_movement_score"
        ]
        row["relative_top_5pct_spike_candidate"] = str(row["time_seconds"] == "2.5000000")
        score = float(row["future_arousal_movement_score"])
        row[f"member_seed_{SEEDS[0]}"] = f"{score - 0.03:.9f}"
        row[f"member_seed_{SEEDS[1]}"] = f"{score:.9f}"
        row[f"member_seed_{SEEDS[2]}"] = f"{score + 0.03:.9f}"
    updated_scores = np.asarray(
        [float(row["future_arousal_movement_score"]) for row in rows], dtype=np.float64
    )
    order = np.argsort(updated_scores, kind="stable")
    updated_percentiles = np.empty(len(rows), dtype=np.float64)
    updated_percentiles[order] = np.arange(1, len(rows) + 1, dtype=np.float64) / len(rows)
    for row, percentile in zip(rows, updated_percentiles, strict=True):
        row["within_video_percentile"] = f"{percentile:.7f}"
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    timeline = _data(
        bridge_api["client"].get(
            "/api/neural-bridge/v1/analyses/beat_test_bundle/timeline"
        )
    )
    event = timeline["events"]["items"][0]
    assert event["anchor_time_seconds"] == 2.5
    assert event["full_bridge_history_context"] is True
    assert event["full_upstream_window_context"] is False
    assert event["cold_start_context"] is True

    report = _data(
        bridge_api["client"].get(
            "/api/neural-bridge/v1/analyses/beat_test_bundle/report"
        )
    )
    assert "cold_start_context" in report["findings"][0]["quality_flags"]


def test_evidence_scopes_do_not_promote_external_validity_or_upstream_parity(bridge_api) -> None:
    timeline = _data(
        bridge_api["client"].get(
            "/api/neural-bridge/v1/analyses/beat_test_bundle/timeline"
        )
    )
    evidence = timeline["evidence_scopes"]

    assert evidence["implementation_reproduction"]["passed"] is True
    assert evidence["implementation_reproduction"]["per_seed"][str(SEEDS[0])]["max_abs_error"] == 0.0
    assert evidence["model_validation_reference"]["dataset_id"] == "AGAIN"
    assert evidence["model_validation_reference"]["applies_to_this_analysis"] is False
    assert evidence["model_validation_reference"]["metrics"]["spearman"] == {
        "real": 0.1785132961,
        "strongest_control": 0.1004882655,
        "delta": 0.0780250306,
    }
    assert evidence["run_level_validation"] == {
        "evidence_scope": "this_analysis",
        "labels_available": False,
        "controls_run": False,
        "validation_status": "not_external_validation_without_labels",
        "external_validity": "not_established",
    }
    assert evidence["upstream_backend_parity"]["status"] == "not_evaluated"


def test_report_is_deterministic_bounded_and_path_free(bridge_api) -> None:
    first = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/beat_test_bundle/report"
    )
    second = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/beat_test_bundle/report?format=json"
    )

    assert first.status_code == 200
    assert first.get_json() == second.get_json()
    report = _data(first)
    assert report["report_schema"] == "neural_bridge.report.v1"
    assert report["executive_summary"]["candidate_count"] == 1
    assert report["findings"][0]["anchor_time_seconds"] == 5.0
    assert report["findings"][0]["wording_status"] == "bounded"
    assert "potential relative response increase" in report["findings"][0]["review_guidance"]
    assert report["controls_and_validation"]["model_validation_reference"][
        "applies_to_this_analysis"
    ] is False
    assert report["claim_boundaries"]["external_validity_established"] is False
    assert report["claim_boundaries"]["calibrated_exact_arousal"] is False
    assert report["claim_boundaries"]["arousal_dropoff_available"] is False
    serialized = json.dumps(first.get_json(), sort_keys=True)
    assert str(bridge_api["tmp_path"]) not in serialized
    assert "/private/" not in serialized

    unsupported = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/beat_test_bundle/report?format=pdf"
    )
    assert unsupported.status_code == 406
    assert unsupported.get_json()["error"]["code"] == "unsupported_report_format"


def test_media_supports_head_and_byte_ranges_without_path_disclosure(bridge_api) -> None:
    client = bridge_api["client"]
    expected = bridge_api["media_path"].read_bytes()[10:30]

    ranged = client.get(
        "/api/neural-bridge/v1/analyses/beat_test_bundle/media",
        headers={"Range": "bytes=10-29"},
    )
    assert ranged.status_code == 206
    assert ranged.data == expected
    assert ranged.headers["Content-Range"].startswith("bytes 10-29/")
    assert ranged.headers["Accept-Ranges"] == "bytes"
    assert ranged.headers["Content-Type"].startswith("video/mp4")
    assert str(bridge_api["tmp_path"]) not in json.dumps(dict(ranged.headers))

    head = client.head("/api/neural-bridge/v1/analyses/beat_test_bundle/media")
    assert head.status_code == 200
    assert head.data == b""
    assert int(head.headers["Content-Length"]) == bridge_api["media_path"].stat().st_size


def test_predictions_csv_download_is_fixed_to_registered_bundle(bridge_api) -> None:
    response = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/beat_test_bundle/predictions.csv"
    )

    assert response.status_code == 200
    assert response.data == bridge_api["expected"]["predictions"].read_bytes()
    assert response.headers["Content-Type"].startswith("text/csv")
    assert "attachment" in response.headers["Content-Disposition"]
    assert "beat_test_bundle-neural-bridge-predictions.csv" in response.headers[
        "Content-Disposition"
    ]


def test_invalid_and_incomplete_bundles_fail_closed(bridge_api) -> None:
    client = bridge_api["client"]
    incomplete = bridge_api["results_root"] / "missing_predictions"
    incomplete.mkdir()
    (incomplete / "run_manifest.json").write_text(
        json.dumps({"status": "complete"}) + "\n", encoding="utf-8"
    )

    missing = client.get(
        "/api/neural-bridge/v1/analyses/missing_predictions/timeline"
    )
    assert missing.status_code == 409
    assert missing.get_json()["error"]["code"] == "bundle_incomplete"

    invalid_id = client.get("/api/neural-bridge/v1/analyses/bad$id/timeline")
    assert invalid_id.status_code == 400
    assert invalid_id.get_json()["error"]["code"] == "invalid_analysis_id"

    unknown = client.get("/api/neural-bridge/v1/analyses/unknown/timeline")
    assert unknown.status_code == 404
    assert unknown.get_json()["error"]["code"] == "analysis_not_found"


def test_schema_mismatch_and_untrusted_media_path_are_rejected(bridge_api) -> None:
    invalid_bundle, _ = _write_bundle(
        bridge_api["results_root"],
        bridge_api["media_path"],
        analysis_id="row_count_mismatch",
        row_count_override=21,
    )
    response = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/row_count_mismatch/timeline"
    )
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "row_count_mismatch"

    _write_bundle(
        bridge_api["results_root"],
        bridge_api["media_path"],
        analysis_id="unsafe_media",
        input_video="/etc/passwd",
    )
    media = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/unsafe_media/media"
    )
    assert media.status_code == 404
    assert media.get_json()["error"]["code"] == "media_not_found"
    assert invalid_bundle.is_dir()


def test_ensemble_mismatch_fails_closed(bridge_api) -> None:
    _write_bundle(
        bridge_api["results_root"],
        bridge_api["media_path"],
        analysis_id="invalid_ensemble",
        ensemble_offset=0.003,
    )

    response = bridge_api["client"].get(
        "/api/neural-bridge/v1/analyses/invalid_ensemble/timeline"
    )

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "invalid_ensemble"
