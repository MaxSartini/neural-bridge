"""Execute the control-complete VEATIC 2.1 Phase 00 foundation audit."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neural_bridge.veatic21.contracts import (
    CURRENT_STATE,
    EXPECTED_CORTICAL_DTYPE,
    EXPECTED_CORTICAL_WIDTH,
    EXPECTED_QUALITY_COUNTS,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_SOURCE_MATCH_COUNTS,
    EXPECTED_VIDEO_IDS,
    LIFECYCLE_ROOT,
    MANDATORY_CHECKS,
    MASTER_SPECIFICATION,
    PHASE00_REQUIRED_ARRAYS,
    PHASE00_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    TRIBE_KEY_SCHEMA,
    TRIBE_ROOT,
    TRIBE_TREE_FILES,
    TRIBE_TREE_SHA256,
    TRIBE_TREE_SIZE_BYTES,
    VJEPA_ALLOWED_FILENAMES,
    VJEPA_ALLOWED_TREE_FILES,
    VJEPA_ALLOWED_TREE_SHA256,
    VJEPA_ALLOWED_TREE_SIZE_BYTES,
    VJEPA_PAYLOAD_VERIFIED_FILENAMES,
    VJEPA_ROOT,
    reject_forbidden_runtime_path,
    validate_runtime_manifest_paths,
)
from neural_bridge.veatic21.data import (
    allowlisted_tree_identity,
    discover_numeric_video_ids,
    load_json,
    load_phase00_tribe_arrays,
    read_row_identity,
    safe_sha256_file,
    validate_allowed_vjepa_filenames,
    validate_cortical_array,
    validate_quality_arrays,
    validate_row_count_identity,
    validate_video_inventory,
    verify_allowed_payload_record,
)
from neural_bridge.veatic21.evidence import canonical_json_bytes, source_tree_digest


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty row inventory")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _expected_tribe_paths() -> tuple[Path, ...]:
    paths = [Path("per_video/run_status.json")]
    for video_id in EXPECTED_VIDEO_IDS:
        base = Path("per_video") / video_id
        paths.extend(
            (
                base / "manifest.json",
                base / "status.json",
                base / "tribe_v2_cortical_predictions.npz",
            )
        )
    return tuple(paths)


def _expected_vjepa_paths() -> tuple[Path, ...]:
    return tuple(
        Path(video_id) / filename
        for video_id in EXPECTED_VIDEO_IDS
        for filename in VJEPA_ALLOWED_FILENAMES
    )


def _validate_exact_tribe_layout() -> None:
    ignored_names = {".DS_Store", ".pytest_cache", "__pycache__"}
    root_names = {
        path.name for path in TRIBE_ROOT.iterdir() if path.name not in ignored_names
    }
    if root_names != {"per_video"}:
        raise ValueError(f"unexpected TRIBE root entries: {sorted(root_names)}")
    per_video = TRIBE_ROOT / "per_video"
    expected_names = {"run_status.json", *EXPECTED_VIDEO_IDS}
    actual_names = {
        path.name for path in per_video.iterdir() if path.name not in ignored_names
    }
    if actual_names != expected_names:
        raise ValueError("TRIBE per-video root layout mismatch")
    expected_video_files = {
        "manifest.json",
        "status.json",
        "tribe_v2_cortical_predictions.npz",
    }
    for video_id in EXPECTED_VIDEO_IDS:
        actual = {
            path.name
            for path in (per_video / video_id).iterdir()
            if path.name not in ignored_names
        }
        if actual != expected_video_files:
            raise ValueError(f"TRIBE file layout mismatch for video {video_id}: {actual}")


def _assert_tree(
    actual: dict[str, str | int], *, sha256: str, files: int, size_bytes: int, name: str
) -> None:
    expected = {"sha256_tree": sha256, "files": files, "size_bytes": size_bytes}
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{name} tree identity mismatch: {mismatches}")


def _assert_source_firewall(package_root: Path) -> dict[str, object]:
    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "neural_bridge.again" or name.startswith("neural_bridge.again."):
                    violations.append(f"{path}:{node.lineno}:{name}")
    if violations:
        raise ValueError(f"VEATIC source imports AGAIN: {violations}")
    imported = sorted(name for name in sys.modules if name.startswith("neural_bridge.again"))
    if imported:
        raise ValueError(f"AGAIN modules were executed in the Phase 00 process: {imported}")
    return {
        "source_import_violations": violations,
        "again_modules_imported_or_executed": imported,
        "pass": True,
    }


def phase01_authorized(checks: dict[str, bool]) -> bool:
    return set(checks) == set(MANDATORY_CHECKS) and all(checks.values())


def _write_artifact_manifest(output_root: Path, filenames: tuple[str, ...]) -> dict[str, Any]:
    artifacts = []
    for filename in filenames:
        path = output_root / filename
        artifacts.append(
            {"path": filename, "bytes": path.stat().st_size, "sha256": safe_sha256_file(path)}
        )
    manifest = {
        "schema": "veatic21_phase00_artifact_manifest_v1",
        "created_at": _utc_now(),
        "root": str(output_root),
        "artifacts": artifacts,
    }
    _atomic_write_json(output_root / "artifact-manifest.json", manifest)
    return manifest


def _write_checksum_file(output_root: Path, filenames: tuple[str, ...]) -> None:
    lines = [f"{safe_sha256_file(output_root / name)}  {name}" for name in filenames]
    _atomic_write_text(output_root / "checksums.sha256", "\n".join(lines) + "\n")


def run_phase00(output_root: Path = PHASE00_ROOT) -> dict[str, Any]:
    """Run the sole authorized audit and emit a sealed external evidence bundle."""

    output_root = reject_forbidden_runtime_path(output_root)
    if output_root != PHASE00_ROOT:
        raise ValueError(f"Phase 00 output root must be exactly {PHASE00_ROOT}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty Phase 00 root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    code_sha256 = source_tree_digest(package_root)
    started_at = _utc_now()
    validate_runtime_manifest_paths(
        (TRIBE_ROOT, VJEPA_ROOT, LIFECYCLE_ROOT, output_root, package_root)
    )
    request = {
        "schema": "veatic21_phase00_request_v1",
        "started_at": started_at,
        "phase": "phase-00-dense-foundation",
        "authorized_action": "immutable input-boundary audit only",
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "inputs": {"tribe_root": str(TRIBE_ROOT), "vjepa_root": str(VJEPA_ROOT)},
        "output_root": str(output_root),
        "code_sha256": code_sha256,
        "operations": {
            "pca": False,
            "ar_fit": False,
            "target_threshold": False,
            "dataset_split": False,
            "model_training": False,
        },
    }
    _atomic_write_json(output_root / "request.json", request)

    checks: dict[str, bool] = {}
    tribe_ids = discover_numeric_video_ids(TRIBE_ROOT / "per_video")
    vjepa_ids = discover_numeric_video_ids(VJEPA_ROOT)
    validate_video_inventory(tribe_ids)
    checks["exact_tribe_video_inventory"] = True
    validate_video_inventory(vjepa_ids)
    checks["exact_vjepa_video_inventory"] = True
    if tribe_ids != vjepa_ids:
        raise ValueError("TRIBE and V-JEPA video IDs differ")
    checks["cross_root_video_identity"] = True
    _validate_exact_tribe_layout()
    checks["tribe_per_video_files"] = True

    run_status = load_json(TRIBE_ROOT / "per_video/run_status.json")
    if not (
        run_status.get("expected_videos") == 124
        and run_status.get("completed_videos") == 124
        and run_status.get("failures") == {}
    ):
        raise ValueError("TRIBE run status is not complete")
    checks["run_status_complete"] = True

    tribe_tree = allowlisted_tree_identity(TRIBE_ROOT, _expected_tribe_paths())
    _assert_tree(
        tribe_tree.compact(),
        sha256=TRIBE_TREE_SHA256,
        files=TRIBE_TREE_FILES,
        size_bytes=TRIBE_TREE_SIZE_BYTES,
        name="TRIBE",
    )
    checks["tribe_tree_digest"] = True
    vjepa_tree = allowlisted_tree_identity(VJEPA_ROOT, _expected_vjepa_paths())
    _assert_tree(
        vjepa_tree.compact(),
        sha256=VJEPA_ALLOWED_TREE_SHA256,
        files=VJEPA_ALLOWED_TREE_FILES,
        size_bytes=VJEPA_ALLOWED_TREE_SIZE_BYTES,
        name="V-JEPA allowlisted metadata",
    )
    checks["vjepa_allowed_tree_digest"] = True

    row_inventory: list[dict[str, object]] = []
    quality_total = Counter[str]()
    match_quality_total = Counter[str]()
    total_rows = 0
    observed_schemas: set[tuple[str, ...]] = set()
    for video_id in EXPECTED_VIDEO_IDS:
        tribe_video_root = TRIBE_ROOT / "per_video" / video_id
        tribe_paths = {
            "manifest": tribe_video_root / "manifest.json",
            "status": tribe_video_root / "status.json",
            "payload": tribe_video_root / "tribe_v2_cortical_predictions.npz",
        }
        if not all(path.is_file() and not path.is_symlink() for path in tribe_paths.values()):
            raise ValueError(f"incomplete TRIBE file set for video {video_id}")
        tribe_manifest = load_json(tribe_paths["manifest"])
        tribe_status = load_json(tribe_paths["status"])
        if tribe_manifest != tribe_status:
            raise ValueError(f"TRIBE manifest/status mismatch for video {video_id}")
        if tribe_manifest.get("status") != "complete" or tribe_manifest.get("video_id") != video_id:
            raise ValueError(f"TRIBE manifest is incomplete for video {video_id}")
        checks["tribe_manifest_status"] = True

        vjepa_video_root = VJEPA_ROOT / video_id
        validate_allowed_vjepa_filenames(vjepa_video_root)
        checks["vjepa_allowed_files"] = True
        vjepa_manifest = load_json(vjepa_video_root / "manifest.json")
        vjepa_status = load_json(vjepa_video_root / "status.json")
        if vjepa_manifest != vjepa_status:
            raise ValueError(f"V-JEPA manifest/status mismatch for video {video_id}")
        if vjepa_manifest.get("status") != "complete" or vjepa_manifest.get("video_id") != video_id:
            raise ValueError(f"V-JEPA manifest is incomplete for video {video_id}")
        checks["vjepa_manifest_status"] = True

        payload_manifest_path = vjepa_video_root / "_PAYLOAD_SHA256.json"
        upload_marker = load_json(vjepa_video_root / "_UPLOAD_COMPLETE.json")
        payload_manifest = load_json(payload_manifest_path)
        payload_manifest_sha256 = safe_sha256_file(payload_manifest_path)
        if not (
            upload_marker.get("status") == "complete"
            and upload_marker.get("video_id") == video_id
            and upload_marker.get("payload_manifest") == "_PAYLOAD_SHA256.json"
            and upload_marker.get("payload_manifest_sha256") == payload_manifest_sha256
            and upload_marker.get("payload_manifest_bytes") == payload_manifest_path.stat().st_size
        ):
            raise ValueError(f"upload-marker mismatch for video {video_id}")
        checks["upload_marker_payload_manifest"] = True
        payload_records = {
            str(record["path"]): record
            for record in payload_manifest.get("files", [])
            if str(record.get("path")) in VJEPA_PAYLOAD_VERIFIED_FILENAMES
        }
        if set(payload_records) != set(VJEPA_PAYLOAD_VERIFIED_FILENAMES):
            raise ValueError(f"allowed payload records mismatch for video {video_id}")
        for filename, record in payload_records.items():
            verify_allowed_payload_record(vjepa_video_root / filename, record)
        checks["vjepa_allowed_file_hashes"] = True

        identity = read_row_identity(vjepa_video_root / "rows.csv", video_id)
        row_count = identity.row_count
        validate_row_count_identity(
            tribe_manifest.get("row_count"), vjepa_manifest.get("row_count"), row_count
        )
        checks["per_video_row_count_identity"] = True
        checks["sequential_row_identity"] = True
        checks["native_two_hz_time_grid"] = True
        checks["csv_video_identity"] = True
        checks["csv_schema_and_encode_policy"] = True

        schema, arrays = load_phase00_tribe_arrays(tribe_paths["payload"], PHASE00_REQUIRED_ARRAYS)
        observed_schemas.add(schema)
        if schema != TRIBE_KEY_SCHEMA:
            raise ValueError(f"TRIBE key schema mismatch for video {video_id}")
        validate_cortical_array(arrays["cortical_prediction"], row_count)
        checks["cortical_layout_and_dtype"] = True
        checks["cortical_finite"] = True
        tribe_time = arrays["time_seconds"]
        if tribe_time.shape != (row_count,) or not np.array_equal(
            tribe_time.astype(np.float64), identity.time_seconds
        ):
            raise ValueError(f"TRIBE/CSV time mismatch for video {video_id}")
        checks["tribe_csv_time_identity"] = True
        quality = validate_quality_arrays(arrays, row_count)
        checks["quality_flag_layout_and_union"] = True
        quality_total.update(quality)
        match_counts = Counter(identity.source_match_quality)
        match_quality_total.update(match_counts)
        total_rows += row_count
        row_inventory.append(
            {
                "video_id": video_id,
                "row_count": row_count,
                "time_start_seconds": float(identity.time_seconds[0]),
                "time_end_seconds": float(identity.time_seconds[-1]),
                "row_hz": EXPECTED_ROW_HZ,
                "cortical_width": EXPECTED_CORTICAL_WIDTH,
                "cortical_dtype": EXPECTED_CORTICAL_DTYPE,
                "black_rows": quality["black_rows"],
                "duplicate_rows": quality["duplicate_rows"],
                "both_rows": quality["both_rows"],
                "union_rows": quality["union_rows"],
                "native_exact_rows": match_counts["native_exact"],
                "linear_native_frames_rows": match_counts["linear_native_frames"],
            }
        )

    if total_rows != EXPECTED_ROW_COUNT:
        raise ValueError(f"total row count mismatch: {total_rows}")
    checks["total_row_count"] = True
    if observed_schemas != {TRIBE_KEY_SCHEMA}:
        raise ValueError("TRIBE key schema is not uniform")
    checks["uniform_tribe_key_schema"] = True
    if dict(quality_total) != EXPECTED_QUALITY_COUNTS:
        raise ValueError(
            "quality counts mismatch: "
            f"expected={EXPECTED_QUALITY_COUNTS}, actual={dict(quality_total)}"
        )
    checks["quality_counts_and_all_rows_retained"] = True
    if dict(match_quality_total) != EXPECTED_SOURCE_MATCH_COUNTS:
        raise ValueError(
            "source-match provenance counts mismatch: "
            f"expected={EXPECTED_SOURCE_MATCH_COUNTS}, actual={dict(match_quality_total)}"
        )
    checks["forbidden_hidden_state_not_read_or_hashed"] = True
    checks["no_model_or_selection_work"] = all(
        not value for value in request["operations"].values()
    )
    source_audit = _assert_source_firewall(package_root)
    checks["again_source_and_runtime_firewall"] = source_audit["pass"] is True
    if not phase01_authorized(checks):
        raise ValueError(f"Phase 00 mandatory check matrix is incomplete: {checks}")

    allowed_input_manifest = {
        "schema": "veatic21_phase00_allowed_input_manifest_v1",
        "tribe": {**tribe_tree.compact(), "entries": list(tribe_tree.entries)},
        "vjepa_allowed_metadata": {
            **vjepa_tree.compact(),
            "allowlist": sorted(VJEPA_ALLOWED_FILENAMES),
            "entries": list(vjepa_tree.entries),
        },
        "forbidden_input": {
            "filename": "vjepa21_hidden_states.npz",
            "loaded": False,
            "hashed": False,
            "copied": False,
            "keys_inspected": False,
        },
        "again_runtime_firewall": source_audit,
    }
    _atomic_write_json(output_root / "allowed-input-manifest.json", allowed_input_manifest)
    _atomic_write_csv(output_root / "row-inventory.csv", row_inventory)

    quality_summary = {
        "schema": "veatic21_phase00_quality_summary_v1",
        **dict(quality_total),
        "all_source_rows_retained": True,
        "rows_filtered": 0,
        "quality_flags_are_metadata_only": True,
        "source_match_quality": dict(sorted(match_quality_total.items())),
    }
    _atomic_write_json(output_root / "quality-summary.json", quality_summary)

    input_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "tribe": tribe_tree.compact(),
                "vjepa_allowed_metadata": vjepa_tree.compact(),
                "rows": total_rows,
                "videos": len(EXPECTED_VIDEO_IDS),
            }
        )
    ).hexdigest()
    derivations = []
    fixed_choices = (
        ("expected_video_count", 124, "exact canonical video inventory"),
        ("expected_total_rows", EXPECTED_ROW_COUNT, "exact canonical row inventory"),
        ("row_hz", EXPECTED_ROW_HZ, "native canonical row grid"),
        ("cortical_width", EXPECTED_CORTICAL_WIDTH, "final TRIBE cortical layout"),
        ("black_fraction_threshold", 0.50, "canonical upstream quality semantics"),
        ("duplicate_fraction_threshold", 0.95, "canonical upstream quality semantics"),
    )
    for name, value, rule in fixed_choices:
        derivations.append(
            {
                "choice": name,
                "value": value,
                "authority": str(MASTER_SPECIFICATION),
                "derivation_rule": rule,
                "owned_rows": "all 20,657 canonical source rows; no selection",
                "code_sha256": code_sha256,
                "artifact_sha256": input_identity_sha256,
            }
        )
    ledger = {
        "schema": "veatic21_derivation_ledger_v1",
        "phase": "phase-00-dense-foundation",
        "fitted_choices": [],
        "numeric_choices": derivations,
        "method_only_transfer": True,
        "again_numeric_choices_inherited": False,
        "input_identity_sha256": input_identity_sha256,
        "code_sha256": code_sha256,
    }
    _atomic_write_json(output_root / "veatic-derivation-ledger.json", ledger)

    completed_at = _utc_now()
    result = {
        "schema": "veatic21_phase00_result_v1",
        "phase": "phase-00-dense-foundation",
        "status": "pass",
        "started_at": started_at,
        "completed_at": completed_at,
        "code_sha256": code_sha256,
        "input_identity_sha256": input_identity_sha256,
        "input_digests": {
            "tribe_tree_sha256": tribe_tree.sha256_tree,
            "vjepa_allowed_metadata_tree_sha256": vjepa_tree.sha256_tree,
        },
        "videos": len(EXPECTED_VIDEO_IDS),
        "rows": total_rows,
        "cortical_layout": ["rows", EXPECTED_CORTICAL_WIDTH],
        "cortical_dtype": EXPECTED_CORTICAL_DTYPE,
        "quality": dict(quality_total),
        "checks": checks,
        "forbidden_input_audit": allowed_input_manifest["forbidden_input"],
        "again_firewall_audit": source_audit,
        "operations": request["operations"],
        "phase01_authorized": True,
        "single_next_authorized_action": (
            "Phase 01 label alignment and VEATIC target-substrate implementation"
        ),
    }
    _atomic_write_json(output_root / "result.json", result)
    report = f"""# VEATIC 2.1 Phase 00 Dense-Foundation Audit

Status: **PASS**

The audit verified all 124 final-TRIBE and matching V-JEPA video identities and all 20,657
canonical 2 Hz rows. Every cortical payload had finite float16 shape `[rows, 20,484]`, the
registered TRIBE and allowlisted V-JEPA tree digests matched, and quality flags retained all
source rows as metadata.

The forbidden `vjepa21_hidden_states.npz` payload was not opened, loaded, inspected, copied,
or hashed. No AGAIN runtime dependency was imported or executed. No PCA, AR, target
threshold, dataset split, or model training occurred.

All {len(MANDATORY_CHECKS)} mandatory checks passed. Phase 01 label alignment and target-substrate
implementation is the single next authorized action after this transition is reviewed,
committed, and pushed to `origin/main`.

Code SHA-256: `{code_sha256}`  
Input identity SHA-256: `{input_identity_sha256}`  
TRIBE tree SHA-256: `{tribe_tree.sha256_tree}`  
V-JEPA allowlisted metadata tree SHA-256: `{vjepa_tree.sha256_tree}`
"""
    _atomic_write_text(output_root / "report.md", report)
    required_outputs = (
        "request.json",
        "allowed-input-manifest.json",
        "row-inventory.csv",
        "quality-summary.json",
        "veatic-derivation-ledger.json",
        "result.json",
        "report.md",
    )
    _write_artifact_manifest(output_root, required_outputs)
    _write_checksum_file(output_root, (*required_outputs, "artifact-manifest.json"))
    return result
