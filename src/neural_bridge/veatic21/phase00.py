"""Publish the compact Phase 00 protected-input foundation audit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_bridge.veatic21.bundle import (
    DEFAULT_BUNDLE_ROOT,
    EXPECTED_VIDEO_IDS,
    FORBIDDEN_NAME,
    PROTECTED_ROOTS,
    BundleError,
    _seal_tree,
    assert_safe_delete_target,
    verify_bundle,
)

DEFAULT_PHASE00_ROOT = Path(
    "/Volumes/onn. Drive/Neural Bridge Artifacts/runs/veatic-2.1/"
    "phase-00-protected-input-foundation"
)

ARRAY_ROLES = {
    "cortical_prediction": "primary_real_predicted_cortical_representation",
    "temporal_diagnostics53": "explicit_diagnostic_control_or_fusion_candidate",
    "tribe_grouped_video_feature": "excluded_upstream_intermediate",
    "time_seconds": "row_identity_audit",
    "source_frame_position": "row_identity_audit",
    "source_floor_frame_index": "row_identity_audit",
    "source_ceil_frame_index": "row_identity_audit",
    "source_interp_alpha": "row_identity_audit",
    "arousal": "rows_csv_equality_check_only",
    "valence": "rows_csv_equality_check_only",
    "source_arousal": "audit_only",
    "source_valence": "audit_only",
    "luma_mean": "nuisance_control_or_audit",
    "luma_std": "nuisance_control_or_audit",
    "frame_luma_std_mean": "nuisance_control_or_audit",
    "motion_absdiff_mean": "nuisance_control_or_audit",
    "black_frame_fraction": "quality_audit",
    "duplicate_frame_fraction": "quality_audit",
    "quality_black_frame_flag": "quality_audit",
    "quality_duplicate_frame_flag": "quality_audit",
    "quality_exclusion_flag": "quality_audit_no_automatic_row_deletion",
    "quality_weight_suggested": "quality_audit_no_automatic_weighting",
    "sample_frame_indices": "sampling_provenance",
    "sample_time_seconds": "sampling_provenance",
    "selected_state_indices": "upstream_provenance",
}


def _sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifact_manifest(root: Path, names: Sequence[str]) -> dict[str, Any]:
    files = []
    for name in names:
        path = root / name
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": "veatic21_phase00_artifact_manifest_v1",
        "file_count": len(files),
        "files": files,
    }


def run_phase00(
    *,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
    output_root: Path = DEFAULT_PHASE00_ROOT,
    expected_ids: Sequence[str] = EXPECTED_VIDEO_IDS,
    workers: int = 12,
    registration_path: Path | None = None,
    protected_roots: Sequence[Path] = PROTECTED_ROOTS,
) -> dict[str, Any]:
    """Audit the sealed bundle and atomically publish a compact, model-free record."""

    bundle_root = bundle_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise BundleError(f"refusing to overwrite existing Phase 00 root: {output_root}")
    assert_safe_delete_target(output_root)
    verification = verify_bundle(
        output_root=bundle_root,
        expected_ids=expected_ids,
        workers=workers,
    )
    manifest_path = bundle_root / "bundle-manifest.json"
    bundle_manifest = json.loads(manifest_path.read_text())
    records = bundle_manifest["per_video"]
    schemas = {tuple(record["npz_keys"]) for record in records}
    if len(schemas) != 1:
        raise BundleError("Phase 00 found inconsistent per-video schemas")
    schema = next(iter(schemas))
    missing_roles = sorted(set(schema).difference(ARRAY_ROLES))
    if missing_roles:
        raise BundleError(f"Phase 00 found arrays without frozen roles: {missing_roles}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        protected_root_audit = {
            "all_protected_roots_exist": all(path.exists() for path in protected_roots),
            "protected_roots": [str(path) for path in protected_roots],
            "phase00_output_outside_protected_roots": True,
            "source_hashes_unchanged": verification["protected_source_hashes_unchanged"],
            "source_roots_modified": False,
        }
        if not protected_root_audit["all_protected_roots_exist"]:
            raise BundleError("one or more protected roots are missing")
        forbidden_audit = {
            "forbidden_name": FORBIDDEN_NAME,
            "bundle_occurrences": [
                str(path.relative_to(bundle_root)) for path in bundle_root.rglob(FORBIDDEN_NAME)
            ],
            "forbidden_file_opened_mapped_hashed_or_copied": False,
            "hidden_state_transport_receipts_copied": False,
        }
        if forbidden_audit["bundle_occurrences"]:
            raise BundleError("forbidden hidden-state payload is present in bundle")
        schema_report = {
            "video_count": len(records),
            "total_rows": sum(int(record["row_count"]) for record in records),
            "npz_schema": list(schema),
            "array_roles": {name: ARRAY_ROLES[name] for name in schema},
            "cortical_width": 20_484,
            "temporal_diagnostics_width": 53,
            "row_hz": 2.0,
            "identity_key": ["video_id", "row_index"],
        }
        result = {
            "schema_version": "veatic21_phase00_result_v1",
            "status": "pass",
            "created_at": datetime.now(UTC).isoformat(),
            "scientific_claim": "input_foundation_only_no_predictive_claim",
            "video_ids": list(expected_ids),
            "video_count": verification["video_count"],
            "total_rows": verification["total_rows"],
            "mismatch_count": 0,
            "nonfinite_cortical_count": 0,
            "forbidden_read_count": 0,
            "bundle_manifest_sha256": verification["bundle_manifest_sha256"],
            "workers": workers,
        }
        provenance = {
            "bundle_root": str(bundle_root),
            "bundle_manifest_sha256": verification["bundle_manifest_sha256"],
            "registration_path": str(registration_path.resolve()) if registration_path else None,
            "registration_sha256": _sha256(registration_path.resolve())
            if registration_path
            else None,
            "vjepa_or_tribe_rerun": False,
            "predictive_model_fit": False,
            "pca_fit": False,
            "target_derived": False,
        }
        derivation_ledger = {
            "phase": "00",
            "derived_predictive_quantities": [],
            "fixed_input_facts": {
                "video_ids": list(expected_ids),
                "row_hz": 2.0,
                "identity_key": ["video_id", "row_index"],
            },
            "next_phase_values_authorized": [],
        }
        documents = {
            "result.json": result,
            "schema-report.json": schema_report,
            "mismatch-ledger.json": {"mismatch_count": 0, "mismatches": []},
            "protected-root-audit.json": protected_root_audit,
            "forbidden-input-audit.json": forbidden_audit,
            "derivation-ledger.json": derivation_ledger,
            "provenance.json": provenance,
        }
        for name, value in documents.items():
            _write_json(temporary / name, value)
        _write_json(
            temporary / "artifact-manifest.json",
            _artifact_manifest(temporary, tuple(documents)),
        )
        os.replace(temporary, output_root)
        _seal_tree(output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_phase00(output_root=output_root)


def verify_phase00(*, output_root: Path = DEFAULT_PHASE00_ROOT) -> dict[str, Any]:
    """Verify the compact Phase 00 record and every declared artifact digest."""

    root = output_root.resolve()
    result = json.loads((root / "result.json").read_text())
    artifact_manifest = json.loads((root / "artifact-manifest.json").read_text())
    for record in artifact_manifest.get("files", []):
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise BundleError(f"Phase 00 artifact mismatch: {path}")
    if result.get("status") != "pass" or result.get("mismatch_count") != 0:
        raise BundleError("Phase 00 result is not a clean pass")
    return {
        "status": "pass",
        "phase00_root": str(root),
        "video_count": result["video_count"],
        "total_rows": result["total_rows"],
        "result_sha256": _sha256(root / "result.json"),
        "artifact_manifest_sha256": _sha256(root / "artifact-manifest.json"),
    }
