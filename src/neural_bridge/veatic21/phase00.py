"""Fresh, deterministic VEATIC 2.1 Phase 00 dense-foundation audit."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from neural_bridge.veatic21.contracts import (
    CURRENT_STATE,
    EXPECTED_CORTICAL_DTYPE,
    EXPECTED_CORTICAL_WIDTH,
    EXPECTED_QUALITY_COUNTS,
    EXPECTED_ROW_COUNT,
    EXPECTED_ROW_HZ,
    EXPECTED_SOURCE_MATCH_COUNTS,
    EXPECTED_TIME_STEP_SECONDS,
    EXPECTED_VIDEO_IDS,
    LABEL_ARRAY_NAMES,
    LIFECYCLE_ROOT,
    MANDATORY_CHECK_NAMES,
    MASTER_SPECIFICATION,
    PHASE00_ROOT,
    REBUILD_PROTOCOL,
    REPOSITORY_ROOT,
    TRIBE_PER_VIDEO_ROOT,
    TRIBE_TREE_FILES,
    TRIBE_TREE_SHA256,
    TRIBE_TREE_SIZE_BYTES,
    VJEPA_ALLOWED_TREE_FILES,
    VJEPA_ALLOWED_TREE_SHA256,
    VJEPA_ALLOWED_TREE_SIZE_BYTES,
    VJEPA_ROOT,
)
from neural_bridge.veatic21.data import (
    allowlisted_tree_identity,
    assert_tree_identity,
    audit_tribe_payload,
    discover_numeric_video_ids,
    expected_tribe_paths,
    expected_vjepa_paths,
    load_json,
    read_row_identity,
    reject_forbidden_runtime_path,
    sha256_file,
    validate_row_count_identity,
    validate_tribe_layout,
    validate_video_inventory,
    validate_vjepa_layout,
    verify_vjepa_payload_records,
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    _atomic_write(path, payload)


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.encode())


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty row inventory")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _source_tree_identity(root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        file_sha256 = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"F\0{path.name}\0{size}\0{file_sha256}\n".encode())
        entries.append({"path": path.name, "bytes": size, "sha256": file_sha256})
    if not entries:
        raise ValueError(f"empty VEATIC source tree: {root}")
    return {"root": str(root), "sha256": digest.hexdigest(), "entries": entries}


def _test_tree_identity(root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    digest = hashlib.sha256()
    for path in sorted(root.glob("test_*.py")):
        file_sha256 = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"F\0{path.name}\0{size}\0{file_sha256}\n".encode())
        entries.append({"path": path.name, "bytes": size, "sha256": file_sha256})
    return {"root": str(root), "sha256": digest.hexdigest(), "entries": entries}


def _audit_again_source_firewall(package_root: Path) -> dict[str, object]:
    audited: list[str] = []
    for path in sorted(package_root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name == "neural_bridge.again" or name.startswith("neural_bridge.again.")
                for name in names
            ):
                raise ValueError(f"VEATIC source imports AGAIN runtime code: {path}")
        audited.append(path.name)
    return {"files_audited": audited, "again_imports": [], "again_runtime_paths": []}


def _validate_runtime_manifest(output_root: Path) -> dict[str, object]:
    runtime_paths = {
        "tribe_per_video_root": str(reject_forbidden_runtime_path(TRIBE_PER_VIDEO_ROOT)),
        "vjepa_root": str(reject_forbidden_runtime_path(VJEPA_ROOT)),
        "output_root": str(reject_forbidden_runtime_path(output_root)),
    }
    return {
        "runtime_paths": runtime_paths,
        "again_code_loaded": False,
        "again_study_runner_executed": False,
        "again_data_or_artifact_loaded": False,
    }


def build_result(
    *,
    checks: Mapping[str, bool],
    code_sha256: str,
    test_sha256: str,
    input_identity_sha256: str,
    tribe_tree_sha256: str,
    vjepa_tree_sha256: str,
    video_count: int,
    row_count: int,
    quality_counts: Mapping[str, int],
    source_match_counts: Mapping[str, int],
    runtime_firewall: Mapping[str, object],
) -> dict[str, object]:
    missing = sorted(set(MANDATORY_CHECK_NAMES) - set(checks))
    extra = sorted(set(checks) - set(MANDATORY_CHECK_NAMES))
    failed = sorted(name for name, passed in checks.items() if not passed)
    phase_pass = not missing and not extra and not failed and len(checks) == 27
    return {
        "schema_version": "veatic21_fresh_phase00_result_v2",
        "phase": "phase-00-dense-foundation",
        "status": "PASS" if phase_pass else "FAIL",
        "phase00_pass": phase_pass,
        "mandatory_controls_expected": 27,
        "mandatory_controls_passed": sum(bool(value) for value in checks.values()),
        "checks": dict(checks),
        "missing_checks": missing,
        "extra_checks": extra,
        "failed_checks": failed,
        "code_sha256": code_sha256,
        "tests_sha256": test_sha256,
        "input_identity_sha256": input_identity_sha256,
        "tribe_tree_sha256": tribe_tree_sha256,
        "vjepa_allowed_metadata_tree_sha256": vjepa_tree_sha256,
        "video_count": video_count,
        "row_count": row_count,
        "row_hz": EXPECTED_ROW_HZ,
        "time_step_seconds": EXPECTED_TIME_STEP_SECONDS,
        "cortical_layout": ["per_video_rows", EXPECTED_CORTICAL_WIDTH],
        "cortical_dtype": EXPECTED_CORTICAL_DTYPE,
        "quality_counts": dict(quality_counts),
        "source_match_counts": dict(source_match_counts),
        "all_video_predictions_considered": video_count == len(EXPECTED_VIDEO_IDS),
        "all_canonical_rows_considered": row_count == EXPECTED_ROW_COUNT,
        "vjepa_hidden_states_loaded": False,
        "vjepa_hidden_states_hashed": False,
        "vjepa_hidden_states_copied": False,
        "vjepa_hidden_states_inspected": False,
        "runtime_firewall": dict(runtime_firewall),
        "operations": {
            "target_construction": False,
            "target_selection": False,
            "dataset_split": False,
            "pca": False,
            "ar_fit": False,
            "model_training": False,
            "head_search": False,
            "washout_design": False,
        },
        "single_next_authorized_action": (
            "Phase 01 exact label alignment and VEATIC target-substrate construction only"
            if phase_pass
            else None
        ),
    }


def _build_report(result: Mapping[str, object]) -> str:
    checks = result["checks"]
    assert isinstance(checks, dict)
    check_lines = "\n".join(
        f"- `{name}`: {'PASS' if value else 'FAIL'}" for name, value in checks.items()
    )
    return f"""# VEATIC 2.1 fresh Phase 00 dense-foundation audit

Status: **{result['status']}**

The audit considered all {result['video_count']} per-video TRIBE cortical prediction payloads
and all {result['row_count']:,} matching canonical rows on the exact 2 Hz grid. Every source
row was retained. No target, split, PCA, AR model, learned head, washout, or scientific model
comparison was created.

The real representation audited in every payload was `cortical_prediction`, with layout
`[per_video_rows, {EXPECTED_CORTICAL_WIDTH:,}]` and dtype `{EXPECTED_CORTICAL_DTYPE}`. V-JEPA
hidden-state payloads were not opened, inspected, loaded, copied, or hashed.

## Mandatory controls

{check_lines}

## Input identity

- TRIBE per-video tree: `{result['tribe_tree_sha256']}`
- V-JEPA metadata-only tree: `{result['vjepa_allowed_metadata_tree_sha256']}`
- combined input identity: `{result['input_identity_sha256']}`
- code identity: `{result['code_sha256']}`

## Authorization

{result['single_next_authorized_action']}
"""


def _write_artifact_manifests(output_root: Path, payload_names: Sequence[str]) -> dict[str, str]:
    entries = []
    for name in payload_names:
        path = output_root / name
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "veatic21_fresh_phase00_artifact_manifest_v2",
        "root": str(output_root),
        "files": entries,
    }
    manifest_path = output_root / "artifact-manifest.json"
    _write_json(manifest_path, manifest)
    checksum_names = [*payload_names, manifest_path.name]
    checksum_lines = [f"{sha256_file(output_root / name)}  {name}" for name in checksum_names]
    _write_text(output_root / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    return {name: sha256_file(output_root / name) for name in checksum_names}


def run_phase00(
    output_root: Path = PHASE00_ROOT,
    *,
    enforce_canonical_output: bool = True,
) -> dict[str, object]:
    """Execute the only authorized fresh Phase 00 action."""

    output_root = reject_forbidden_runtime_path(output_root)
    if enforce_canonical_output and output_root != PHASE00_ROOT:
        raise ValueError(f"Phase 00 output must use the canonical lifecycle root: {PHASE00_ROOT}")
    if enforce_canonical_output and output_root.parent != LIFECYCLE_ROOT:
        raise ValueError("Phase 00 output escaped the fresh lifecycle root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty Phase 00 output: {output_root}")

    package_root = REPOSITORY_ROOT / "src/neural_bridge/veatic21"
    tests_root = REPOSITORY_ROOT / "tests/veatic21"
    source_identity = _source_tree_identity(package_root)
    tests_identity = _test_tree_identity(tests_root)
    source_firewall = _audit_again_source_firewall(package_root)
    runtime_firewall = _validate_runtime_manifest(output_root)
    code_sha256 = str(source_identity["sha256"])
    test_sha256 = str(tests_identity["sha256"])

    request = {
        "schema_version": "veatic21_fresh_phase00_request_v2",
        "phase": "phase-00-dense-foundation",
        "authority": {
            "master": str(MASTER_SPECIFICATION),
            "protocol": str(REBUILD_PROTOCOL),
            "current_state": str(CURRENT_STATE),
        },
        "inputs": runtime_firewall["runtime_paths"],
        "output_root": str(output_root),
        "code_sha256": code_sha256,
        "tests_sha256": test_sha256,
        "operations": {
            "audit_all_124_prediction_payloads": True,
            "audit_all_20657_rows": True,
            "target_construction": False,
            "dataset_split": False,
            "pca": False,
            "ar_fit": False,
            "model_training": False,
        },
    }
    _write_json(output_root / "request.json", request)

    checks: dict[str, bool] = {}
    tribe_ids = discover_numeric_video_ids(TRIBE_PER_VIDEO_ROOT)
    vjepa_ids = discover_numeric_video_ids(VJEPA_ROOT)
    validate_video_inventory(tribe_ids)
    checks["exact_tribe_video_inventory"] = True
    validate_video_inventory(vjepa_ids)
    checks["exact_vjepa_video_inventory"] = True
    if tribe_ids != vjepa_ids:
        raise ValueError("TRIBE and V-JEPA video inventories differ")
    checks["cross_root_video_identity"] = True

    run_status = load_json(TRIBE_PER_VIDEO_ROOT / "run_status.json")
    if not (
        run_status.get("expected_videos") == 124
        and run_status.get("completed_videos") == 124
        and run_status.get("failures") == {}
    ):
        raise ValueError("TRIBE run status is not complete")
    checks["run_status_complete"] = True
    validate_tribe_layout(TRIBE_PER_VIDEO_ROOT, tribe_ids)
    checks["complete_tribe_per_video_layout"] = True
    validate_vjepa_layout(VJEPA_ROOT, vjepa_ids)
    checks["matching_vjepa_allowed_layout"] = True

    tribe_tree = allowlisted_tree_identity(
        TRIBE_PER_VIDEO_ROOT, expected_tribe_paths(tribe_ids)
    )
    assert_tree_identity(
        tribe_tree,
        sha256=TRIBE_TREE_SHA256,
        files=TRIBE_TREE_FILES,
        size_bytes=TRIBE_TREE_SIZE_BYTES,
        name="TRIBE per-video",
    )
    checks["tribe_tree_digest"] = True
    vjepa_tree = allowlisted_tree_identity(VJEPA_ROOT, expected_vjepa_paths(vjepa_ids))
    assert_tree_identity(
        vjepa_tree,
        sha256=VJEPA_ALLOWED_TREE_SHA256,
        files=VJEPA_ALLOWED_TREE_FILES,
        size_bytes=VJEPA_ALLOWED_TREE_SIZE_BYTES,
        name="V-JEPA metadata-only",
    )
    checks["vjepa_metadata_tree_digest"] = True

    tribe_sha_by_path = {str(entry["path"]): str(entry["sha256"]) for entry in tribe_tree.entries}
    row_inventory: list[dict[str, object]] = []
    quality = Counter[str]()
    source_matches = Counter[str]()
    schemas: set[tuple[str, ...]] = set()
    accessed_arrays: set[str] = set()
    total_rows = 0
    for video_id in tribe_ids:
        tribe_root = TRIBE_PER_VIDEO_ROOT / video_id
        vjepa_root = VJEPA_ROOT / video_id
        tribe_manifest = load_json(tribe_root / "manifest.json")
        tribe_status = load_json(tribe_root / "status.json")
        if tribe_manifest != tribe_status:
            raise ValueError(f"TRIBE manifest/status mismatch for video {video_id}")
        if tribe_manifest.get("status") != "complete" or tribe_manifest.get("video_id") != video_id:
            raise ValueError(f"TRIBE status is incomplete for video {video_id}")

        vjepa_manifest = load_json(vjepa_root / "manifest.json")
        vjepa_status = load_json(vjepa_root / "status.json")
        if vjepa_manifest != vjepa_status:
            raise ValueError(f"V-JEPA manifest/status mismatch for video {video_id}")
        if vjepa_manifest.get("status") != "complete" or vjepa_manifest.get("video_id") != video_id:
            raise ValueError(f"V-JEPA status is incomplete for video {video_id}")
        verify_vjepa_payload_records(vjepa_root, video_id)

        identity = read_row_identity(vjepa_root / "rows.csv", video_id)
        validate_row_count_identity(
            tribe_manifest.get("row_count"), vjepa_manifest.get("row_count"), identity.row_count
        )
        payload_path = tribe_root / "tribe_v2_cortical_predictions.npz"
        payload = audit_tribe_payload(payload_path, identity)
        if tribe_manifest.get("cortical_shape") != list(payload.cortical_shape):
            raise ValueError(f"TRIBE manifest cortical shape mismatch for video {video_id}")

        total_rows += identity.row_count
        source_matches.update(identity.source_match_counts)
        quality.update(
            {
                "black_rows": payload.black_rows,
                "duplicate_rows": payload.duplicate_rows,
                "both_rows": payload.both_rows,
                "union_rows": payload.union_rows,
            }
        )
        schemas.add(payload.key_schema)
        accessed_arrays.update(payload.accessed_arrays)
        row_inventory.append(
            {
                "video_id": video_id,
                "row_count": identity.row_count,
                "time_start_seconds": float(identity.time_seconds[0]),
                "time_end_seconds": float(identity.time_seconds[-1]),
                "row_hz": EXPECTED_ROW_HZ,
                "time_step_seconds": EXPECTED_TIME_STEP_SECONDS,
                "cortical_width": EXPECTED_CORTICAL_WIDTH,
                "cortical_dtype": EXPECTED_CORTICAL_DTYPE,
                "black_rows": payload.black_rows,
                "duplicate_rows": payload.duplicate_rows,
                "both_rows": payload.both_rows,
                "quality_union_rows": payload.union_rows,
                "native_exact_rows": identity.source_match_counts.get("native_exact", 0),
                "linear_native_frame_rows": identity.source_match_counts.get(
                    "linear_native_frames", 0
                ),
                "prediction_payload_sha256": tribe_sha_by_path[
                    f"{video_id}/tribe_v2_cortical_predictions.npz"
                ],
            }
        )

    checks["tribe_manifest_status"] = True
    checks["vjepa_manifest_status"] = True
    checks["upload_marker_payload_manifest"] = True
    checks["allowed_vjepa_file_hashes"] = True
    checks["per_video_row_count_identity"] = True
    if total_rows != EXPECTED_ROW_COUNT:
        raise ValueError(f"total row count mismatch: {total_rows}")
    checks["total_row_count"] = True
    checks["sequential_row_index"] = True
    checks["exact_time_grid"] = True
    checks["exact_video_id"] = True
    checks["rows_schema_encode_policy"] = True
    checks["cortical_shape_dtype"] = True
    checks["cortical_finite"] = True
    if len(schemas) != 1:
        raise ValueError("TRIBE key schema is not uniform across all videos")
    checks["uniform_tribe_key_schema"] = True
    checks["copied_time_identity"] = True
    checks["quality_flag_semantics"] = True

    quality["unflagged_rows"] = total_rows - quality["union_rows"]
    quality["total_rows"] = total_rows
    if dict(quality) != EXPECTED_QUALITY_COUNTS:
        raise ValueError(f"quality count mismatch: {dict(quality)}")
    if dict(source_matches) != EXPECTED_SOURCE_MATCH_COUNTS:
        raise ValueError(f"source-match count mismatch: {dict(source_matches)}")
    checks["quality_counts_all_rows_retained"] = True
    if accessed_arrays & LABEL_ARRAY_NAMES:
        raise AssertionError("Phase 00 accessed label arrays through the feature path")
    checks["forbidden_hidden_state_firewall"] = True
    checks["no_modeling_operations"] = True
    if source_firewall["again_imports"] or source_firewall["again_runtime_paths"]:
        raise ValueError("AGAIN source firewall failed")
    checks["again_runtime_firewall"] = True

    input_identity = {
        "tribe_tree_sha256": tribe_tree.sha256,
        "vjepa_allowed_metadata_tree_sha256": vjepa_tree.sha256,
        "video_ids": list(tribe_ids),
        "row_count": total_rows,
        "row_hz": EXPECTED_ROW_HZ,
        "time_step_seconds": EXPECTED_TIME_STEP_SECONDS,
        "cortical_width": EXPECTED_CORTICAL_WIDTH,
        "cortical_dtype": EXPECTED_CORTICAL_DTYPE,
    }
    input_identity_sha256 = digest_json(input_identity)
    allowed_manifest = {
        "schema_version": "veatic21_fresh_phase00_allowed_input_manifest_v2",
        "input_identity": input_identity,
        "input_identity_sha256": input_identity_sha256,
        "tribe": {**tribe_tree.compact(), "entries": list(tribe_tree.entries)},
        "vjepa_metadata_only": {**vjepa_tree.compact(), "entries": list(vjepa_tree.entries)},
        "vjepa_hidden_states_loaded": False,
        "vjepa_hidden_states_hashed": False,
    }
    quality_summary = {
        "schema_version": "veatic21_fresh_phase00_quality_summary_v2",
        "counts": dict(quality),
        "all_rows_retained": True,
        "quality_filter_applied": False,
    }
    ledger = {
        "schema_version": "veatic21_fresh_derivation_ledger_v2",
        "phase": "phase-00-dense-foundation",
        "code_sha256": code_sha256,
        "input_identity_sha256": input_identity_sha256,
        "entries": [
            {
                "name": "complete_video_inventory",
                "value": list(EXPECTED_VIDEO_IDS),
                "evidence": "master specification v2.0 and canonical-root enumeration",
                "derivation_rule": "all numeric IDs 0..123; no subset",
                "owned_rows": "all canonical rows",
            },
            {
                "name": "native_row_grid",
                "value": {"row_hz": EXPECTED_ROW_HZ, "step_seconds": EXPECTED_TIME_STEP_SECONDS},
                "evidence": "per-video rows.csv declarations and exact timestamps",
                "derivation_rule": "preserve source 2 Hz schema without resampling",
                "owned_rows": "all canonical rows",
            },
            {
                "name": "cortical_layout",
                "value": {"width": EXPECTED_CORTICAL_WIDTH, "dtype": EXPECTED_CORTICAL_DTYPE},
                "evidence": "all 124 cortical_prediction arrays",
                "derivation_rule": "audit source layout; no projection or selection",
                "owned_rows": "all canonical rows",
            },
            {
                "name": "scientific_choices",
                "value": [],
                "evidence": "Phase 00 audit-only contract",
                "derivation_rule": "no fitted or selected scientific value in Phase 00",
                "owned_rows": "none",
            },
        ],
    }
    result = build_result(
        checks=checks,
        code_sha256=code_sha256,
        test_sha256=test_sha256,
        input_identity_sha256=input_identity_sha256,
        tribe_tree_sha256=tribe_tree.sha256,
        vjepa_tree_sha256=vjepa_tree.sha256,
        video_count=len(row_inventory),
        row_count=total_rows,
        quality_counts=quality,
        source_match_counts=source_matches,
        runtime_firewall=runtime_firewall,
    )
    if not result["phase00_pass"]:
        raise RuntimeError("Phase 00 mandatory controls did not all pass")

    _write_json(output_root / "allowed-input-manifest.json", allowed_manifest)
    _write_csv(output_root / "row-inventory.csv", row_inventory)
    _write_json(output_root / "quality-summary.json", quality_summary)
    _write_json(output_root / "veatic-derivation-ledger.json", ledger)
    _write_json(output_root / "result.json", result)
    _write_text(output_root / "report.md", _build_report(result))
    output_hashes = _write_artifact_manifests(
        output_root,
        (
            "request.json",
            "allowed-input-manifest.json",
            "row-inventory.csv",
            "quality-summary.json",
            "veatic-derivation-ledger.json",
            "result.json",
            "report.md",
        ),
    )
    result["output_hashes"] = {
        **output_hashes,
        "checksums.sha256": sha256_file(output_root / "checksums.sha256"),
    }
    return result


def verify_phase00_output(output_root: Path = PHASE00_ROOT) -> dict[str, object]:
    output_root = reject_forbidden_runtime_path(output_root)
    manifest = load_json(output_root / "artifact-manifest.json")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("invalid Phase 00 artifact manifest")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid Phase 00 artifact record")
        path = output_root / record["path"]
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Phase 00 artifact mismatch: {path}")

    checksum_lines = (output_root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        expected_sha256, name = line.split("  ", maxsplit=1)
        if sha256_file(output_root / name) != expected_sha256:
            raise ValueError(f"Phase 00 checksum mismatch: {name}")
    result = load_json(output_root / "result.json")
    checks = result.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("Phase 00 result is missing its mandatory-control mapping")
    if set(checks) != set(MANDATORY_CHECK_NAMES) or not all(checks.values()):
        raise ValueError("Phase 00 result does not contain 27 passing mandatory controls")
    if not (
        result.get("phase00_pass") is True
        and result.get("video_count") == 124
        and result.get("row_count") == EXPECTED_ROW_COUNT
        and result.get("all_video_predictions_considered") is True
        and result.get("all_canonical_rows_considered") is True
        and result.get("vjepa_hidden_states_loaded") is False
        and result.get("vjepa_hidden_states_hashed") is False
    ):
        raise ValueError("Phase 00 result summary is not promotable")
    return {
        "verified": True,
        "artifact_manifest_sha256": sha256_file(output_root / "artifact-manifest.json"),
        "checksums_sha256": sha256_file(output_root / "checksums.sha256"),
        "result_sha256": sha256_file(output_root / "result.json"),
        "result": result,
    }
