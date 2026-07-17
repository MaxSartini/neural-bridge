#!/usr/bin/env python3
"""Run the locked VEATIC 2.1 nested discovery, confirmation, and final refit.

This is the sole orchestration entry point for the VEATIC 2.1 end-state
programme.  The scientific contract lives in :mod:`veatic21_endstate_contract`;
this file turns that contract into auditable work items and owns durable,
restart-safe stage manifests.  It deliberately keeps model execution serial:
two MLX jobs are never launched concurrently.

The runner has four useful modes:

``--stage discovery``
    Run the exact six-recipe, three-seed nested selection matrix.  Every score
    comes from one of the three inner validation folds and the outer test fold
    is sealed out of PCA, normalizer, AR, threshold, and model selection.

``--stage confirmation``
    Lock the per-outer-fold winner and run the exact 1,680 privileged
    continuous, 1,680 true-BCE privileged binary, and 560 zero-label rows.
    Zero-label event PR-AUC is derived from the continuous prediction; there is
    no separately trained zero-label binary head.

``--stage final``
    Freeze one deterministic global recipe per target/protocol, derive fixed
    epochs from completed training provenance, then refit PCA, normalizers, AR,
    thresholds, and models from scratch on all 124 videos.  No in-sample score
    is presented as confirmation evidence.

``--stage all``
    Execute the three stages in the above order.

``--dry-run`` writes only the immutable plan and work manifests.  ``--smoke``
is explicitly non-promotable and reduces execution to one target/fold/seed
while keeping every matched lane.  Source cache files are opened read-only and
are never replaced or deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts import again_dense_2hz_phase4_pca_bridge as phase4  # noqa: E402
from backend.scripts import veatic21_compact_cache as compact  # noqa: E402
from backend.scripts import veatic21_controls as controls  # noqa: E402
from backend.scripts import veatic21_discovery as discovery  # noqa: E402
from backend.scripts import veatic21_endstate_contract as endstate  # noqa: E402
from backend.scripts import veatic21_evaluation as evaluation  # noqa: E402
from backend.scripts import veatic21_features as feature_contract  # noqa: E402
from backend.scripts import veatic21_modeling as modeling  # noqa: E402
from backend.scripts import veatic21_pca as pca  # noqa: E402
from backend.scripts import veatic21_targets as targets  # noqa: E402
from backend.scripts import run_veatic21_distilled_development as distilled  # noqa: E402


RUN_SCHEMA_VERSION = "veatic21_endstate_runner_v1"
PREDICTION_SCHEMA_VERSION = "veatic21_endstate_prediction_shard_v1"
FINAL_EXPORT_SCHEMA_VERSION = "veatic21_endstate_all124_export_v1"
SMOKE_SELECTION_SCHEMA_VERSION = "veatic21_nonpromotable_smoke_selection_v1"
STAGES = ("discovery", "confirmation", "final", "all")
DISCOVERY_PROTOCOLS = (
    discovery.PRIVILEGED_CONTINUOUS,
    discovery.PRIVILEGED_BINARY,
    discovery.ZERO_LABEL_CONTINUOUS,
)
PRIVILEGED_ENDPOINTS = ("privileged_continuous", "privileged_binary")
ZERO_ENDPOINT = "zero_label_continuous"
MEMBER_KIND = "member"
ENSEMBLE_KIND = "ensemble"
MAX_PCA_WIDTH = 256
INNER_FOLD_COUNT = 3
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
DEFAULT_MAX_EPOCHS = 80
DEFAULT_PATIENCE = 12
DEFAULT_BATCH_SIZE = 8192
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 1e-4


class EndStateRunError(RuntimeError):
    """Raised when a stage cannot preserve the locked execution contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|")
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise EndStateRunError(f"refusing to write an empty CSV: {path}")
    fieldnames = tuple(rows[0])
    if any(tuple(row) != fieldnames for row in rows):
        raise EndStateRunError("CSV rows have inconsistent field order")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shared-derived-root", type=Path, default=None)
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        default=compact.DEFAULT_IDENTITY_MANIFEST,
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.max_epochs < 1:
        raise EndStateRunError("--max-epochs must be positive")
    if args.patience < 1 or args.patience > args.max_epochs:
        raise EndStateRunError("--patience must be in [1, max-epochs]")
    if args.batch_size < 1:
        raise EndStateRunError("--batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise EndStateRunError("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise EndStateRunError("--weight-decay must be finite and non-negative")
    if args.bootstrap_resamples < 100:
        raise EndStateRunError("--bootstrap-resamples must be at least 100")
    if args.audit_only and args.dry_run:
        raise EndStateRunError("--audit-only and --dry-run are mutually exclusive")


@dataclass(frozen=True)
class DatasetSeal:
    video_count: int
    total_rows: int
    row_hz: float
    row_counts: tuple[tuple[str, int], ...]
    cache_fingerprint: str
    row_plan_sha256: str
    model_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "row_counts": [[video, count] for video, count in self.row_counts],
        }


def dataset_seal_from_report(report: compact.Veatic21ValidationReport) -> DatasetSeal:
    seal = DatasetSeal(
        video_count=int(report.video_count),
        total_rows=int(report.total_rows),
        row_hz=float(report.row_hz),
        row_counts=tuple((str(video), int(report.row_counts[video])) for video in report.video_ids),
        cache_fingerprint=str(report.dataset_fingerprint_sha256),
        row_plan_sha256=str(report.row_plan_sha256),
        model_sha256=str(report.model_sha256),
    )
    validate_dataset_seal(seal)
    return seal


@dataclass(frozen=True)
class DenseDataset:
    """Read-only row-aligned substrate used by every numerical stage."""

    row_idx: np.ndarray
    local_row_idx: np.ndarray
    video_id: np.ndarray
    time_seconds: np.ndarray
    arousal: np.ndarray
    valence: np.ndarray
    quality_valid: np.ndarray
    diagnostics: np.ndarray
    cortical: np.ndarray
    target_values: Mapping[str, np.ndarray]
    target_valid: Mapping[str, np.ndarray]
    dataset_seal_digest: str
    artifact_digest: str

    @property
    def rows(self) -> int:
        return int(len(self.row_idx))


def _dense_paths(shared_root: Path) -> tuple[Path, Path, Path]:
    root = Path(shared_root) / "veatic21_endstate_dense"
    return root / "cortical_prediction_fp16.npy", root / "rows.npz", root / "manifest.json"


def _target_rows(
    *,
    video_id: np.ndarray,
    local_row_idx: np.ndarray,
    time_seconds: np.ndarray,
    arousal: np.ndarray,
    valence: np.ndarray,
) -> targets.Veatic21TargetResult:
    rows = [
        {
            "video_id": str(video_id[index]),
            "row_index": int(local_row_idx[index]),
            "time_start_seconds": float(time_seconds[index]),
            "sampling_frequency_hz": 2.0,
            "targets": {
                "arousal": float(arousal[index]),
                "valence": float(valence[index]),
            },
        }
        for index in range(len(video_id))
    ]
    return targets.build_veatic21_targets(rows, build_events=False)


def _extract_target_arrays(
    result: targets.Veatic21TargetResult,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for target in endstate.PRIMARY_TARGETS:
        value = np.full(len(result.rows), np.nan, dtype=np.float32)
        mask = np.zeros(len(result.rows), dtype=bool)
        mask_name = f"target_mask_{target.name}"
        for index, row in enumerate(result.rows):
            if bool(row["target_masks"].get(mask_name, False)):
                raw = row["targets"].get(target.name)
                if raw is not None and math.isfinite(float(raw)):
                    value[index] = float(raw)
                    mask[index] = True
        if not np.any(mask) or not np.isfinite(value[mask]).all():
            raise EndStateRunError(f"target construction failed for {target.name}")
        values[target.name] = value
        masks[target.name] = mask
    return values, masks


def materialize_dense_dataset(
    *,
    cache: compact.Veatic21CompactCache,
    seal: DatasetSeal,
    shared_root: Path,
) -> DenseDataset:
    """Materialize one sealed memmap; never alter a source cache file."""

    # Reuse the already exercised cache-to-row adapter and Phase-4 memmap
    # builder.  The old Phase-4 schema is not scientific authority here: the
    # new compact-cache seal, target contract, PCA contract, and runner sidecar
    # below remain authoritative.
    manifest_rows, frame, diagnostics = distilled._dense_rows_and_diagnostics(cache)
    cortical = phase4.load_or_build_cortical_memmap(
        cache.cache_root,
        frame,
        output_root=Path(shared_root) / "cortical_memmap",
    )
    if cortical.shape != (seal.total_rows, compact.PREDICTION_WIDTH) or cortical.dtype != np.float16:
        raise EndStateRunError("validated cortical memmap shape/dtype drift")
    video_id = frame["video_id"].astype(str).to_numpy()
    local_row_idx = frame["row_index"].to_numpy(dtype=np.int32)
    time_seconds = frame["time_seconds"].to_numpy(dtype=np.float32)
    arousal = frame["arousal"].to_numpy(dtype=np.float32)
    valence = frame["valence"].to_numpy(dtype=np.float32)
    quality_valid = ~frame["quality_exclusion_flag"].to_numpy(dtype=bool)
    target_result = targets.build_veatic21_targets(manifest_rows, build_events=False)
    target_values, target_valid = _extract_target_arrays(target_result)
    sidecar_payload = {
        "schema_version": "veatic21_endstate_dense_adapter_v1",
        "dataset_seal_digest": canonical_digest(seal.manifest()),
        "row_count": seal.total_rows,
        "video_count": seal.video_count,
        "row_hz": 2.0,
        "cortical_path": str(Path(cortical.filename).resolve()),
        "cortical_shape": list(cortical.shape),
        "cortical_dtype": str(cortical.dtype),
        "row_identity_digest": canonical_digest(
            {
                "video": controls.array_digest(video_id),
                "local_row": controls.array_digest(local_row_idx),
                "time": controls.array_digest(time_seconds),
            }
        ),
        "target_contract": target_result.contract,
        "old_phase4_schema_authoritative": False,
        "source_cache_mutated": False,
    }
    sidecar = Path(shared_root) / "veatic21_endstate_dense_adapter.json"
    if sidecar.exists():
        observed = json.loads(sidecar.read_text(encoding="utf-8"))
        if observed != {**sidecar_payload, "artifact_digest": canonical_digest(sidecar_payload)}:
            raise EndStateRunError("shared dense adapter sidecar identity mismatch")
    else:
        atomic_json(
            sidecar,
            {**sidecar_payload, "artifact_digest": canonical_digest(sidecar_payload)},
        )
    return DenseDataset(
        row_idx=np.arange(seal.total_rows, dtype=np.int64),
        local_row_idx=local_row_idx,
        video_id=video_id,
        time_seconds=time_seconds,
        arousal=arousal,
        valence=valence,
        quality_valid=quality_valid,
        diagnostics=np.asarray(diagnostics, dtype=np.float32),
        cortical=cortical,
        target_values=target_values,
        target_valid=target_valid,
        dataset_seal_digest=canonical_digest(seal.manifest()),
        artifact_digest=canonical_digest(sidecar_payload),
    )

def synthetic_dataset_seal(row_counts: Mapping[str, int]) -> DatasetSeal:
    """Build a plan-only seal for tests and explicit dry-run construction."""

    ordered = tuple(sorted(((str(k), int(v)) for k, v in row_counts.items()), key=lambda x: int(x[0])))
    seal = DatasetSeal(
        video_count=len(ordered),
        total_rows=sum(value for _, value in ordered),
        row_hz=2.0,
        row_counts=ordered,
        cache_fingerprint=canonical_digest(ordered),
        row_plan_sha256=compact.ROW_PLAN_SHA256,
        model_sha256=compact.MODEL_SHA256,
    )
    validate_dataset_seal(seal, require_canonical_rows=False)
    return seal


def validate_dataset_seal(
    seal: DatasetSeal, *, require_canonical_rows: bool = True
) -> None:
    if seal.video_count != endstate.VIDEO_COUNT or len(seal.row_counts) != endstate.VIDEO_COUNT:
        raise EndStateRunError("VEATIC 2.1 requires all 124 videos")
    if len({video for video, _ in seal.row_counts}) != endstate.VIDEO_COUNT:
        raise EndStateRunError("dataset seal contains duplicate video identifiers")
    if any(count <= 0 for _, count in seal.row_counts):
        raise EndStateRunError("dataset seal contains a non-positive row count")
    if not math.isclose(seal.row_hz, 2.0, rel_tol=0.0, abs_tol=1e-9):
        raise EndStateRunError("VEATIC 2.1 requires exact 2 Hz rows")
    if require_canonical_rows and seal.total_rows != compact.EXPECTED_TOTAL_ROWS:
        raise EndStateRunError("canonical VEATIC 2.1 row count mismatch")
    if seal.total_rows != sum(count for _, count in seal.row_counts):
        raise EndStateRunError("dataset seal row total is inconsistent")
    if seal.row_plan_sha256 != compact.ROW_PLAN_SHA256:
        raise EndStateRunError("row-plan provenance mismatch")
    if seal.model_sha256 != compact.MODEL_SHA256:
        raise EndStateRunError("V-JEPA 2.1 model provenance mismatch")


def build_plan(seal: DatasetSeal) -> discovery.NestedDiscoveryPlan:
    validate_dataset_seal(
        seal, require_canonical_rows=(seal.total_rows == compact.EXPECTED_TOTAL_ROWS)
    )
    plan = discovery.build_nested_discovery_plan(
        dict(seal.row_counts),
        targets=tuple(target.name for target in endstate.PRIMARY_TARGETS),
        protocols=DISCOVERY_PROTOCOLS,
        discovery_seeds=endstate.DISCOVERY_SEEDS,
        confirmation_seeds=tuple(
            endstate.PRIVILEGED_CONFIRMATION_SEEDS
            + endstate.ZERO_LABEL_CONFIRMATION_SEEDS
        ),
        inner_fold_count=INNER_FOLD_COUNT,
    )
    ownership = discovery.audit_nested_ownership(plan)
    if not ownership.passed:
        raise EndStateRunError(
            f"nested ownership failed: {list(ownership.failed_checks)}"
        )
    return plan


def scientific_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "huber_delta": 1.0,
        "huber_delta_source": "veatic21_modeling_locked_default",
        "residual_do_no_harm_gate": True,
        "residual_alpha_cap": 0.12,
        "model_hidden_dim": 64,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "serial_gpu_execution": True,
        "pca_max_width_fit_once": MAX_PCA_WIDTH,
        "pca_slice_policy": "sealed_leading_component_slice",
        "zero_label_binary_head_trained": False,
        "binary_objective": "true_bce_with_frozen_ar_logit_offset",
        "continuous_objective": "huber_residual_over_frozen_ar_score",
        "event_quantile": endstate.EVENT_QUANTILE,
    }


@dataclass(frozen=True)
class RunIdentity:
    schema_version: str
    dataset_seal_digest: str
    plan_digest: str
    endstate_contract_digest: str
    settings_digest: str
    cache_root: str
    upstream_root: str
    identity_manifest: str
    shared_derived_root: str
    smoke: bool
    digest: str

    def manifest(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_digest:
            payload.pop("digest")
        return payload


def build_run_identity(
    *,
    args: argparse.Namespace,
    seal: DatasetSeal,
    plan: discovery.NestedDiscoveryPlan,
) -> RunIdentity:
    shared = Path(args.shared_derived_root or (Path(args.output_root) / "derived"))
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "dataset_seal_digest": canonical_digest(seal.manifest()),
        "plan_digest": plan.digest,
        "endstate_contract_digest": endstate.contract_manifest()["contract_digest"],
        "settings_digest": canonical_digest(scientific_settings(args)),
        "cache_root": str(Path(args.cache_root).expanduser().resolve()),
        "upstream_root": str(Path(args.upstream_root).expanduser().resolve()),
        "identity_manifest": str(Path(args.identity_manifest).expanduser().resolve()),
        "shared_derived_root": str(shared.expanduser().resolve()),
        "smoke": bool(args.smoke),
    }
    return RunIdentity(**payload, digest=canonical_digest(payload))


def write_or_verify_run_identity(output_root: Path, identity: RunIdentity) -> Path:
    path = Path(output_root) / "run_identity.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != identity.manifest():
            raise EndStateRunError(
                "resume identity mismatch; use a new output root instead of mixing runs"
            )
    else:
        atomic_json(path, identity.manifest())
    return path


@dataclass(frozen=True)
class PcaRequest:
    scope: str
    outer_fold: int
    inner_fold: int | None
    feature_family: str
    parent_width: int
    requested_widths: tuple[int, ...]
    fit_videos: tuple[str, ...]
    held_out_videos: tuple[str, ...]
    parent_identity: str

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "fit_videos": list(self.fit_videos),
            "held_out_videos": list(self.held_out_videos),
            "requested_widths": list(self.requested_widths),
        }


def pca_requests_for_plan(plan: discovery.NestedDiscoveryPlan) -> tuple[PcaRequest, ...]:
    """Plan max-width PCA fits; 64-wide recipes use a sealed leading slice."""

    families: dict[str, set[int]] = defaultdict(set)
    for recipe in plan.recipes:
        families[recipe.feature_family].add(int(recipe.pca_width))
    requests: list[PcaRequest] = []
    for outer in plan.outer_folds:
        for inner in outer.inner_folds:
            for family, widths in sorted(families.items()):
                requested = tuple(sorted(widths))
                parent_width = max(requested)
                payload = {
                    "scope": "inner_discovery",
                    "outer_fold": outer.outer_fold,
                    "inner_fold": inner.fold,
                    "feature_family": family,
                    "parent_width": parent_width,
                    "requested_widths": list(requested),
                    "fit_videos": list(inner.train_videos),
                    "held_out_videos": list(
                        tuple(inner.validation_videos) + tuple(outer.test_videos)
                    ),
                    "slice_policy": "leading_components_only",
                }
                requests.append(
                    PcaRequest(
                        scope="inner_discovery",
                        outer_fold=outer.outer_fold,
                        inner_fold=inner.fold,
                        feature_family=family,
                        parent_width=parent_width,
                        requested_widths=requested,
                        fit_videos=inner.train_videos,
                        held_out_videos=tuple(inner.validation_videos)
                        + tuple(outer.test_videos),
                        parent_identity=canonical_digest(payload),
                    )
                )
        for family, widths in sorted(families.items()):
            requested = tuple(sorted(widths))
            payload = {
                "scope": "outer_confirmation",
                "outer_fold": outer.outer_fold,
                "inner_fold": None,
                "feature_family": family,
                "parent_width": max(requested),
                "requested_widths": list(requested),
                "fit_videos": list(outer.train_videos),
                "held_out_videos": list(outer.test_videos),
                "slice_policy": "leading_components_only",
            }
            requests.append(
                PcaRequest(
                    scope="outer_confirmation",
                    outer_fold=outer.outer_fold,
                    inner_fold=None,
                    feature_family=family,
                    parent_width=max(requested),
                    requested_widths=requested,
                    fit_videos=outer.train_videos,
                    held_out_videos=outer.test_videos,
                    parent_identity=canonical_digest(payload),
                )
            )
    return tuple(requests)


def pca_slice_manifest(
    *, parent_metadata: Mapping[str, Any], requested_width: int
) -> dict[str, Any]:
    identity = parent_metadata.get("identity")
    if not isinstance(identity, Mapping):
        raise EndStateRunError("parent PCA metadata has no immutable identity")
    parent_width = int(identity.get("pca_width", 0))
    width = int(requested_width)
    if width not in feature_contract.ALLOWED_PCA_WIDTHS or width > parent_width:
        raise EndStateRunError("invalid leading PCA slice width")
    payload = {
        "schema_version": "veatic21_pca_leading_slice_v1",
        "parent_pca_identity_sha256": str(parent_metadata.get("identity_sha256", "")),
        "parent_width": parent_width,
        "slice_width": width,
        "component_range": [0, width],
        "policy": "leading_components_only_no_refit",
    }
    if not payload["parent_pca_identity_sha256"]:
        raise EndStateRunError("parent PCA identity digest is missing")
    return {**payload, "slice_digest": canonical_digest(payload)}


@dataclass(frozen=True)
class MatrixCell:
    endpoint: str
    target: str
    outer_fold: int
    lane: str
    row_kind: str
    seed: int | None
    ensemble_group: int | None
    member_seeds: tuple[int, ...]
    recipe: str
    objective: str
    response_free: bool
    descriptive_only: bool
    event_from_continuous_prediction: bool
    key: str

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["member_seeds"] = list(self.member_seeds)
        return payload


def _selection_recipe(
    selections: discovery.DiscoverySelectionArtifact | SmokeSelectionArtifact,
    *,
    target: str,
    endpoint: str,
    outer_fold: int,
) -> str:
    protocol = {
        "privileged_continuous": discovery.PRIVILEGED_CONTINUOUS,
        "privileged_binary": discovery.PRIVILEGED_BINARY,
        "zero_label_continuous": discovery.ZERO_LABEL_CONTINUOUS,
    }[endpoint]
    return selections.selection(target, protocol, outer_fold).selected_recipe


def _cell(
    *,
    endpoint: str,
    target: str,
    outer_fold: int,
    lane: str,
    row_kind: str,
    seed: int | None,
    ensemble_group: int | None,
    member_seeds: tuple[int, ...],
    recipe: str,
) -> MatrixCell:
    response_free = endpoint == ZERO_ENDPOINT and lane != "privileged_teacher_ceiling"
    descriptive = endpoint == ZERO_ENDPOINT and lane == "privileged_teacher_ceiling"
    objective = "bce_with_frozen_ar_logit_offset" if endpoint == "privileged_binary" else (
        "direct_huber_continuous" if endpoint == ZERO_ENDPOINT else "huber_frozen_ar_residual"
    )
    payload = {
        "endpoint": endpoint,
        "target": target,
        "outer_fold": int(outer_fold),
        "lane": lane,
        "row_kind": row_kind,
        "seed": seed,
        "ensemble_group": ensemble_group,
        "member_seeds": list(member_seeds),
        "recipe": recipe,
        "objective": objective,
        "response_free": response_free,
        "descriptive_only": descriptive,
        "event_from_continuous_prediction": endpoint == ZERO_ENDPOINT,
    }
    return MatrixCell(**payload, key=canonical_digest(payload))


def build_confirmation_matrix(
    selections: discovery.DiscoverySelectionArtifact | SmokeSelectionArtifact,
    plan: discovery.NestedDiscoveryPlan,
    *,
    smoke: bool = False,
) -> tuple[MatrixCell, ...]:
    if isinstance(selections, SmokeSelectionArtifact):
        if not smoke:
            raise EndStateRunError("smoke selection cannot expand a canonical matrix")
        verify_smoke_selection_artifact(selections, plan)
    else:
        discovery.verify_selection_artifact(selections, plan=plan)
    targets_to_run = plan.targets[:1] if smoke else plan.targets
    outer_to_run = plan.outer_folds[:1] if smoke else plan.outer_folds
    privileged_seeds = (
        endstate.PRIVILEGED_CONFIRMATION_SEEDS[:1]
        if smoke
        else endstate.PRIVILEGED_CONFIRMATION_SEEDS
    )
    privileged_groups = (
        (privileged_seeds,)
        if smoke
        else endstate.PRIVILEGED_CONFIRMATION_GROUPS
    )
    zero_seeds = (
        endstate.ZERO_LABEL_CONFIRMATION_SEEDS[:1]
        if smoke
        else endstate.ZERO_LABEL_CONFIRMATION_SEEDS
    )
    zero_groups = (zero_seeds,) if smoke else endstate.ZERO_LABEL_CONFIRMATION_GROUPS
    cells: list[MatrixCell] = []
    for target in targets_to_run:
        for outer in outer_to_run:
            for endpoint in PRIVILEGED_ENDPOINTS:
                recipe = _selection_recipe(
                    selections,
                    target=target,
                    endpoint=endpoint,
                    outer_fold=outer.outer_fold,
                )
                for lane in endstate.PRIVILEGED_LANES:
                    for seed in privileged_seeds:
                        cells.append(
                            _cell(
                                endpoint=endpoint,
                                target=target,
                                outer_fold=outer.outer_fold,
                                lane=lane,
                                row_kind=MEMBER_KIND,
                                seed=seed,
                                ensemble_group=None,
                                member_seeds=(seed,),
                                recipe=recipe,
                            )
                        )
                    for group_index, group in enumerate(privileged_groups, start=1):
                        cells.append(
                            _cell(
                                endpoint=endpoint,
                                target=target,
                                outer_fold=outer.outer_fold,
                                lane=lane,
                                row_kind=ENSEMBLE_KIND,
                                seed=None,
                                ensemble_group=group_index,
                                member_seeds=tuple(group),
                                recipe=recipe,
                            )
                        )
            endpoint = ZERO_ENDPOINT
            recipe = _selection_recipe(
                selections,
                target=target,
                endpoint=endpoint,
                outer_fold=outer.outer_fold,
            )
            # Prediction sealing order is contractual: six response-free lanes
            # first, the descriptive privileged teacher only afterwards.
            for lane in endstate.ZERO_LABEL_RESPONSE_FREE_LANES + endstate.ZERO_LABEL_DESCRIPTIVE_LANES:
                for seed in zero_seeds:
                    cells.append(
                        _cell(
                            endpoint=endpoint,
                            target=target,
                            outer_fold=outer.outer_fold,
                            lane=lane,
                            row_kind=MEMBER_KIND,
                            seed=seed,
                            ensemble_group=None,
                            member_seeds=(seed,),
                            recipe=recipe,
                        )
                    )
                for group_index, group in enumerate(zero_groups, start=1):
                    cells.append(
                        _cell(
                            endpoint=endpoint,
                            target=target,
                            outer_fold=outer.outer_fold,
                            lane=lane,
                            row_kind=ENSEMBLE_KIND,
                            seed=None,
                            ensemble_group=group_index,
                            member_seeds=tuple(group),
                            recipe=recipe,
                        )
                    )
    # Privileged VEATIC is the immediate scientific priority.  Keep every
    # zero-label cell in the locked matrix, but seal it only after all
    # privileged continuous and true-BCE cells have completed.
    endpoint_order = {
        discovery.PRIVILEGED_CONTINUOUS: 0,
        discovery.PRIVILEGED_BINARY: 1,
        discovery.ZERO_LABEL_CONTINUOUS: 2,
    }
    cells.sort(key=lambda cell: endpoint_order[cell.endpoint])
    audit_confirmation_matrix(cells, plan=plan, smoke=smoke)
    return tuple(cells)


@dataclass(frozen=True)
class MatrixAudit:
    passed: bool
    smoke: bool
    observed_rows: int
    expected_rows: int
    endpoint_counts: tuple[tuple[str, int], ...]
    duplicate_keys: tuple[str, ...]
    failed_checks: tuple[str, ...]
    digest: str

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["endpoint_counts"] = [list(item) for item in self.endpoint_counts]
        payload["duplicate_keys"] = list(self.duplicate_keys)
        payload["failed_checks"] = list(self.failed_checks)
        return payload


def audit_confirmation_matrix(
    cells: Sequence[MatrixCell],
    *,
    plan: discovery.NestedDiscoveryPlan,
    smoke: bool = False,
) -> MatrixAudit:
    counts = Counter(cell.key for cell in cells)
    duplicates = tuple(sorted(key for key, count in counts.items() if count != 1))
    endpoint_counts = Counter(cell.endpoint for cell in cells)
    expected = 42 if smoke else endstate.SCORED_ROWS.grand_total
    # Smoke: 2 privileged endpoints * 7 lanes * (1 member + 1 ensemble)
    # plus 7 zero lanes * (1 member + 1 ensemble) = 42.
    checks = {
        "exact_row_count": len(cells) == expected,
        "unique_keys": not duplicates,
        "all_124_no_reserve": plan.video_count == 124,
        "exact_five_outer_folds": len(plan.outer_folds) == 5,
        "binary_rows_are_true_bce": all(
            cell.objective == "bce_with_frozen_ar_logit_offset"
            for cell in cells
            if cell.endpoint == "privileged_binary"
        ),
        "no_zero_label_binary_head": all(
            cell.endpoint != "zero_label_binary" for cell in cells
        ),
        "zero_event_from_continuous": all(
            cell.event_from_continuous_prediction
            for cell in cells
            if cell.endpoint == ZERO_ENDPOINT
        ),
        "zero_schema_response_free_except_teacher": all(
            (
                cell.response_free
                and not cell.descriptive_only
                and cell.lane in endstate.ZERO_LABEL_RESPONSE_FREE_LANES
            )
            or (
                not cell.response_free
                and cell.descriptive_only
                and cell.lane == "privileged_teacher_ceiling"
            )
            for cell in cells
            if cell.endpoint == ZERO_ENDPOINT
        ),
        "ensemble_members_fixed": all(
            cell.row_kind == MEMBER_KIND
            or (
                len(cell.member_seeds) == (1 if smoke else 3)
                and cell.seed is None
            )
            for cell in cells
        ),
    }
    if not smoke:
        checks.update(
            {
                "privileged_continuous_1680": endpoint_counts["privileged_continuous"]
                == endstate.SCORED_ROWS.continuous_total,
                "privileged_binary_1680": endpoint_counts["privileged_binary"]
                == endstate.SCORED_ROWS.binary_total,
                "zero_label_560": endpoint_counts[ZERO_ENDPOINT]
                == endstate.SCORED_ROWS.zero_label_total,
            }
        )
    failed = tuple(name for name, passed in checks.items() if not passed)
    payload = {
        "smoke": bool(smoke),
        "observed_rows": len(cells),
        "expected_rows": expected,
        "endpoint_counts": sorted(endpoint_counts.items()),
        "duplicate_keys": list(duplicates),
        "checks": checks,
    }
    audit = MatrixAudit(
        passed=not failed,
        smoke=bool(smoke),
        observed_rows=len(cells),
        expected_rows=expected,
        endpoint_counts=tuple(sorted(endpoint_counts.items())),
        duplicate_keys=duplicates,
        failed_checks=failed,
        digest=canonical_digest(payload),
    )
    if not audit.passed:
        raise EndStateRunError(
            f"confirmation matrix audit failed: {list(audit.failed_checks)}"
        )
    return audit


def assert_no_outer_leakage(
    *,
    plan: discovery.NestedDiscoveryPlan,
    outer_fold: int,
    fit_videos: Sequence[str],
    validation_videos: Sequence[str] = (),
) -> None:
    outer = plan.outer(outer_fold)
    fit = set(map(str, fit_videos))
    validation = set(map(str, validation_videos))
    test = set(outer.test_videos)
    if fit & validation:
        raise EndStateRunError("fit and validation video ownership overlaps")
    leaked = test & (fit | validation)
    if leaked:
        raise EndStateRunError(
            f"outer-test videos leaked into fitting/selection: {sorted(leaked)}"
        )
    if not (fit | validation) <= set(outer.train_videos):
        raise EndStateRunError("fit/validation ownership leaves outer training videos")


def assert_zero_label_schema(names: Sequence[str]) -> None:
    try:
        feature_contract.validate_feature_names(names)
    except feature_contract.Veatic21FeatureAuditError as exc:
        raise EndStateRunError(f"zero-label feature audit failed: {exc}") from exc


def prediction_path(output_root: Path, cell: MatrixCell) -> Path:
    suffix = (
        f"seed_{cell.seed}"
        if cell.row_kind == MEMBER_KIND
        else f"ensemble_{cell.ensemble_group}"
    )
    return (
        Path(output_root)
        / "predictions"
        / cell.endpoint
        / cell.target
        / f"fold_{cell.outer_fold}"
        / cell.lane
        / f"{suffix}.npz"
    )


def checkpoint_path(output_root: Path, cell: MatrixCell) -> Path:
    suffix = (
        f"seed_{cell.seed}"
        if cell.row_kind == MEMBER_KIND
        else f"ensemble_{cell.ensemble_group}"
    )
    return (
        Path(output_root)
        / "checkpoints"
        / cell.endpoint
        / cell.target
        / f"fold_{cell.outer_fold}"
        / cell.lane
        / f"{suffix}.npz"
    )


def seal_prediction_shard(
    *,
    path: Path,
    cell: MatrixCell,
    row_indices: np.ndarray,
    video_ids: np.ndarray,
    y_true: np.ndarray,
    prediction: np.ndarray,
    event_threshold: float,
    checkpoint: Path | None,
    run_identity_digest: str,
    extra_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    indices = np.asarray(row_indices, dtype=np.int64)
    videos = np.asarray(video_ids, dtype=np.str_)
    truth = np.asarray(y_true, dtype=np.float32)
    predicted = np.asarray(prediction, dtype=np.float32)
    if not (
        indices.ndim == videos.ndim == truth.ndim == predicted.ndim == 1
        and len(indices) == len(videos) == len(truth) == len(predicted)
        and len(indices) > 0
    ):
        raise EndStateRunError("prediction shard arrays are not one-dimensional/aligned")
    if len(np.unique(indices)) != len(indices) or not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise EndStateRunError("prediction shard rows are duplicate or non-finite")
    if not math.isfinite(float(event_threshold)):
        raise EndStateRunError("prediction shard event threshold is not finite")
    atomic_npz(
        path,
        row_index=indices,
        video_id=videos,
        y_true=truth,
        prediction=predicted,
    )
    manifest_payload = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "cell": cell.manifest(),
        "run_identity_digest": run_identity_digest,
        "event_threshold": float(event_threshold),
        "row_count": int(len(indices)),
        "row_index_sha256": array_digest(indices),
        "video_id_sha256": array_digest(videos),
        "y_true_sha256": array_digest(truth),
        "prediction_sha256": array_digest(predicted),
        "prediction_file": str(path.resolve()),
        "prediction_file_sha256": file_sha256(path),
        "checkpoint_file": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_file_sha256": file_sha256(checkpoint) if checkpoint else None,
        "extra_provenance": dict(extra_provenance),
    }
    manifest = {**manifest_payload, "manifest_digest": canonical_digest(manifest_payload)}
    atomic_json(path.with_suffix(".json"), manifest)
    return manifest


def verify_prediction_shard(
    path: Path,
    *,
    cell: MatrixCell,
    run_identity_digest: str,
) -> Mapping[str, Any]:
    manifest_path = Path(path).with_suffix(".json")
    if not Path(path).is_file() or not manifest_path.is_file():
        raise EndStateRunError(f"incomplete prediction shard pair: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = dict(manifest)
    observed_digest = payload.pop("manifest_digest", None)
    if observed_digest != canonical_digest(payload):
        raise EndStateRunError("prediction shard manifest digest mismatch")
    if manifest.get("cell") != cell.manifest():
        raise EndStateRunError("prediction shard belongs to a different matrix cell")
    if manifest.get("run_identity_digest") != run_identity_digest:
        raise EndStateRunError("prediction shard belongs to a different run identity")
    if manifest.get("prediction_file_sha256") != file_sha256(Path(path)):
        raise EndStateRunError("prediction shard checksum drift")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"row_index", "video_id", "y_true", "prediction"}:
            raise EndStateRunError("prediction shard array schema drift")
        for key in bundle.files:
            expected = manifest.get(f"{key}_sha256")
            if expected != array_digest(bundle[key]):
                raise EndStateRunError(f"prediction shard {key} digest mismatch")
    return manifest


def matrix_completion_audit(
    *,
    output_root: Path,
    cells: Sequence[MatrixCell],
    run_identity_digest: str,
    require_complete: bool,
) -> dict[str, Any]:
    complete: list[str] = []
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for cell in cells:
        path = prediction_path(output_root, cell)
        if not path.exists() and not path.with_suffix(".json").exists():
            missing.append(cell.key)
            continue
        try:
            verify_prediction_shard(
                path, cell=cell, run_identity_digest=run_identity_digest
            )
        except Exception as exc:  # fail-closed audit records exact broken key
            invalid.append({"key": cell.key, "error": str(exc)})
        else:
            complete.append(cell.key)
    payload = {
        "schema_version": "veatic21_matrix_completion_audit_v1",
        "expected_rows": len(cells),
        "complete_rows": len(complete),
        "missing_rows": len(missing),
        "invalid_rows": len(invalid),
        "complete_keys_digest": canonical_digest(sorted(complete)),
        "missing_keys": sorted(missing),
        "invalid": invalid,
        "passed": not missing and not invalid,
    }
    if require_complete and not payload["passed"]:
        raise EndStateRunError(
            f"matrix incomplete: missing={len(missing)} invalid={len(invalid)}"
        )
    return payload


@dataclass(frozen=True)
class GlobalRecipeSelection:
    target: str
    protocol: str
    selected_recipe: str
    outer_selection_count: int
    mean_leaderboard_rank: float
    complexity_score: int
    recipe_order: int
    source_selection_digest: str
    outer_test_scores_used: bool
    digest: str

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def select_global_recipes(
    artifact: discovery.DiscoverySelectionArtifact,
    plan: discovery.NestedDiscoveryPlan,
) -> tuple[GlobalRecipeSelection, ...]:
    """Collapse nested choices without consulting any outer-test score.

    Majority of the five outer selections wins.  Ties use mean inner-only
    leaderboard rank, then lower prespecified complexity, recipe order, name.
    """

    discovery.verify_selection_artifact(artifact, plan=plan)
    results: list[GlobalRecipeSelection] = []
    for target in plan.targets:
        for protocol in plan.protocols:
            selections = [
                artifact.selection(target, protocol, outer.outer_fold)
                for outer in plan.outer_folds
            ]
            counts = Counter(item.selected_recipe for item in selections)
            rank_values: dict[str, list[int]] = defaultdict(list)
            by_name = {recipe.name: recipe for recipe in plan.recipes}
            for selected in selections:
                for rank, entry in enumerate(selected.leaderboard, start=1):
                    rank_values[entry.recipe].append(rank)
            ranked = sorted(
                plan.recipes,
                key=lambda recipe: (
                    -counts[recipe.name],
                    float(np.mean(rank_values[recipe.name])),
                    recipe.complexity_score,
                    recipe.order,
                    recipe.name,
                ),
            )
            winner = ranked[0]
            payload = {
                "target": target,
                "protocol": protocol,
                "selected_recipe": winner.name,
                "outer_selection_count": counts[winner.name],
                "mean_leaderboard_rank": float(np.mean(rank_values[winner.name])),
                "complexity_score": by_name[winner.name].complexity_score,
                "recipe_order": by_name[winner.name].order,
                "source_selection_digest": artifact.artifact_digest,
                "outer_test_scores_used": False,
            }
            results.append(
                GlobalRecipeSelection(**payload, digest=canonical_digest(payload))
            )
    return tuple(results)


def derive_fixed_epoch(
    epochs: Sequence[int], *, max_epochs: int
) -> int:
    values = np.asarray(tuple(epochs), dtype=np.int64)
    if values.ndim != 1 or len(values) < 3 or np.any(values < 1) or np.any(values > max_epochs):
        raise EndStateRunError("fixed-epoch derivation requires at least three valid epochs")
    # Half-up median avoids Python's banker rounding and is deterministic.
    median = float(np.median(values))
    return min(int(max_epochs), max(1, int(math.floor(median + 0.5))))


def build_final_export_contract(
    *,
    identity: RunIdentity,
    plan: discovery.NestedDiscoveryPlan,
    selections: discovery.DiscoverySelectionArtifact,
    best_epochs: Mapping[tuple[str, str, int], Sequence[int]],
    max_epochs: int,
) -> dict[str, Any]:
    global_selections = select_global_recipes(selections, plan)
    epochs: list[dict[str, Any]] = []
    for selection in global_selections:
        key = (selection.target, selection.protocol, selection.recipe_order)
        source = best_epochs.get(key)
        if source is None:
            # Compatibility with callers keyed by (target, protocol, recipe name).
            source = best_epochs.get(  # type: ignore[arg-type]
                (selection.target, selection.protocol, selection.selected_recipe)
            )
        if source is None:
            raise EndStateRunError(f"missing fixed-epoch provenance for {key}")
        epochs.append(
            {
                "target": selection.target,
                "protocol": selection.protocol,
                "recipe": selection.selected_recipe,
                "fixed_epoch": derive_fixed_epoch(source, max_epochs=max_epochs),
                "source_epochs": list(map(int, source)),
                "source_epochs_digest": canonical_digest(list(map(int, source))),
            }
        )
    payload = {
        "schema_version": FINAL_EXPORT_SCHEMA_VERSION,
        "run_identity_digest": identity.digest,
        "plan_digest": plan.digest,
        "selection_artifact_digest": selections.artifact_digest,
        "video_count": 124,
        "reserve_count": 0,
        "all_video_refit": True,
        "fresh_pca": True,
        "fresh_normalizers": True,
        "fresh_target_specific_ar": True,
        "fresh_all_video_q90_thresholds": True,
        "selection_frozen_before_refit": True,
        "no_in_sample_metric_claim": True,
        "zero_label_response_inputs": False,
        "zero_label_starts_row0": True,
        "privileged_member_seeds": list(endstate.PRIVILEGED_CONFIRMATION_SEEDS),
        "privileged_fixed_triples": [
            list(group) for group in endstate.PRIVILEGED_CONFIRMATION_GROUPS
        ],
        "zero_label_member_seeds": list(endstate.ZERO_LABEL_CONFIRMATION_SEEDS),
        "zero_label_fixed_ensemble": list(endstate.ZERO_LABEL_CONFIRMATION_SEEDS),
        "global_selections": [item.manifest() for item in global_selections],
        "fixed_epochs": epochs,
        "expected_target_count": 4,
        "expected_protocol_count": 3,
        "source_cache_mutated": False,
    }
    return {**payload, "export_contract_digest": canonical_digest(payload)}


def audit_final_export_contract(payload: Mapping[str, Any]) -> None:
    checks = {
        "schema": payload.get("schema_version") == FINAL_EXPORT_SCHEMA_VERSION,
        "all_124": payload.get("video_count") == 124 and payload.get("reserve_count") == 0,
        "all_video_refit": payload.get("all_video_refit") is True,
        "fresh_fitted_state": all(
            payload.get(name) is True
            for name in (
                "fresh_pca",
                "fresh_normalizers",
                "fresh_target_specific_ar",
                "fresh_all_video_q90_thresholds",
            )
        ),
        "selection_frozen": payload.get("selection_frozen_before_refit") is True,
        "no_in_sample_claim": payload.get("no_in_sample_metric_claim") is True,
        "zero_response_free": payload.get("zero_label_response_inputs") is False,
        "row0": payload.get("zero_label_starts_row0") is True,
        "four_targets_three_protocols": len(payload.get("global_selections", [])) == 12,
        "fixed_epochs_complete": len(payload.get("fixed_epochs", [])) == 12,
        "privileged_triples": payload.get("privileged_fixed_triples")
        == [list(group) for group in endstate.PRIVILEGED_CONFIRMATION_GROUPS],
        "zero_ensemble": payload.get("zero_label_fixed_ensemble")
        == list(endstate.ZERO_LABEL_CONFIRMATION_SEEDS),
        "source_immutable": payload.get("source_cache_mutated") is False,
    }
    without_digest = dict(payload)
    observed = without_digest.pop("export_contract_digest", None)
    checks["digest"] = observed == canonical_digest(without_digest)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise EndStateRunError(f"final export contract failed: {failed}")


def discovery_expected_rows(plan: discovery.NestedDiscoveryPlan) -> int:
    return len(discovery.expected_score_keys(plan))


def write_plan_artifacts(
    *,
    output_root: Path,
    seal: DatasetSeal,
    plan: discovery.NestedDiscoveryPlan,
    identity: RunIdentity,
    args: argparse.Namespace,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "dataset_seal.json", seal.manifest())
    atomic_json(root / "nested_discovery_plan.json", plan.manifest())
    pca_requests = pca_requests_for_plan(plan)
    atomic_json(
        root / "pca_requests.json",
        {
            "schema_version": "veatic21_pca_request_plan_v1",
            "max_width_fit_once": True,
            "leading_slice_policy": True,
            "requests": [request.manifest() for request in pca_requests],
        },
    )
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_identity": identity.manifest(),
        "stage_requested": args.stage,
        "dry_run": bool(args.dry_run),
        "audit_only": bool(args.audit_only),
        "smoke": bool(args.smoke),
        "promotable": False,
        "promotion_requires_completed_canonical_gates": True,
        "dataset": seal.manifest(),
        "plan_digest": plan.digest,
        "endstate_contract": endstate.contract_manifest(),
        "settings": scientific_settings(args),
        "discovery_expected_rows": discovery_expected_rows(plan),
        "confirmation_expected_rows": endstate.SCORED_ROWS.grand_total,
        "stage_order": ["discovery", "confirmation", "final"],
    }
    atomic_json(root / "run_manifest.json", payload)
    write_or_verify_run_identity(root, identity)
    return payload


def _load_selection_artifact(path: Path) -> discovery.DiscoverySelectionArtifact:
    """Rehydrate a selection artifact without accepting unknown fields."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selections: list[discovery.OuterRecipeSelection] = []
    for item in payload["selections"]:
        leaderboard = tuple(
            discovery.RecipeAggregate(
                recipe=str(entry["recipe"]),
                recipe_order=int(entry["recipe_order"]),
                complexity_score=int(entry["complexity_score"]),
                score_rows=int(entry["score_rows"]),
                rank_values=tuple(
                    discovery.AggregateValue(name=str(value["name"]), value=float(value["value"]))
                    for value in entry["rank_values"]
                ),
                aggregate_digest=str(entry["aggregate_digest"]),
            )
            for entry in item["leaderboard"]
        )
        selections.append(
            discovery.OuterRecipeSelection(
                target=str(item["target"]),
                protocol=str(item["protocol"]),
                outer_fold=int(item["outer_fold"]),
                selected_recipe=str(item["selected_recipe"]),
                leaderboard=leaderboard,
                input_score_digest=str(item["input_score_digest"]),
                outer_test_scores_used=bool(item["outer_test_scores_used"]),
                digest=str(item["digest"]),
            )
        )
    return discovery.DiscoverySelectionArtifact(
        schema_version=str(payload["schema_version"]),
        plan_digest=str(payload["plan_digest"]),
        score_digest=str(payload["score_digest"]),
        score_rows=int(payload["score_rows"]),
        selections=tuple(selections),
        ownership_audit_digest=str(payload["ownership_audit_digest"]),
        outer_test_scores_used=bool(payload["outer_test_scores_used"]),
        artifact_digest=str(payload["artifact_digest"]),
    )


def _selection_path(output_root: Path) -> Path:
    return Path(output_root) / "discovery" / "selection_artifact.json"


def _score_rows_path(output_root: Path) -> Path:
    return Path(output_root) / "discovery" / "score_rows.json"


def _smoke_selection_path(output_root: Path) -> Path:
    return Path(output_root) / "discovery" / "smoke_selection_artifact.json"


def _smoke_score_rows_path(output_root: Path) -> Path:
    return Path(output_root) / "discovery" / "smoke_score_rows.json"


@dataclass(frozen=True)
class SmokeSelectionArtifact:
    """Measured one-target/one-fold selection that can never become canonical."""

    schema_version: str
    plan_digest: str
    target: str
    outer_fold: int
    discovery_seed: int
    score_digest: str
    score_rows: int
    selections: tuple[discovery.OuterRecipeSelection, ...]
    ownership_audit_digest: str
    outer_test_scores_used: bool
    explicitly_nonpromotable: bool
    artifact_digest: str

    def manifest(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "plan_digest": self.plan_digest,
            "target": self.target,
            "outer_fold": self.outer_fold,
            "discovery_seed": self.discovery_seed,
            "score_digest": self.score_digest,
            "score_rows": self.score_rows,
            "selections": [item.manifest() for item in self.selections],
            "ownership_audit_digest": self.ownership_audit_digest,
            "outer_test_scores_used": self.outer_test_scores_used,
            "explicitly_nonpromotable": self.explicitly_nonpromotable,
        }
        if include_digest:
            payload["artifact_digest"] = self.artifact_digest
        return payload

    def selection(
        self, target: str, protocol: str, outer_fold: int
    ) -> discovery.OuterRecipeSelection:
        key = (str(target), str(protocol), int(outer_fold))
        for item in self.selections:
            if (item.target, item.protocol, item.outer_fold) == key:
                return item
        raise EndStateRunError(f"no smoke recipe selection for {key}")


def smoke_expected_score_keys(
    plan: discovery.NestedDiscoveryPlan,
) -> frozenset[tuple[str, str, int, str, int, int]]:
    target = plan.targets[0]
    outer = plan.outer_folds[0]
    seed = plan.discovery_seeds[0]
    return frozenset(
        (
            target,
            protocol,
            outer.outer_fold,
            recipe.name,
            inner.fold,
            seed,
        )
        for protocol in plan.protocols
        for recipe in plan.recipes
        for inner in outer.inner_folds
    )


def audit_smoke_score_matrix(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: Iterable[discovery.DiscoveryScoreRow],
) -> tuple[dict[str, Any], tuple[discovery.DiscoveryScoreRow, ...]]:
    rows = tuple(sorted(score_rows, key=lambda row: row.key))
    expected = smoke_expected_score_keys(plan)
    counts = Counter(row.key for row in rows)
    observed = set(counts)
    ownership_failures: list[tuple[str, str, int, str, int, int]] = []
    metric_failures: list[tuple[str, str, int, str, int, int]] = []
    for row in rows:
        if row.key not in expected:
            continue
        inner = plan.inner(row.outer_fold, row.inner_fold)
        if (
            row.score_scope != discovery.INNER_VALIDATION_SCOPE
            or row.fit_videos != inner.train_videos
            or row.validation_videos != inner.validation_videos
            or row.ownership_digest != inner.digest
        ):
            ownership_failures.append(row.key)
        if not set(discovery.required_metrics(row.protocol)).issubset(row.metric_map()):
            metric_failures.append(row.key)
    missing = tuple(sorted(expected - observed))
    unexpected = tuple(sorted(observed - expected))
    duplicates = tuple(sorted(key for key, count in counts.items() if count != 1))
    score_digest = canonical_digest([row.manifest() for row in rows])
    audit = {
        "schema_version": "veatic21_nonpromotable_smoke_score_audit_v1",
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "missing_keys": [list(key) for key in missing],
        "unexpected_keys": [list(key) for key in unexpected],
        "duplicate_keys": [list(key) for key in duplicates],
        "ownership_failures": [list(key) for key in ownership_failures],
        "metric_failures": [list(key) for key in metric_failures],
        "score_digest": score_digest,
        "explicitly_nonpromotable": True,
        "passed": not (
            missing
            or unexpected
            or duplicates
            or ownership_failures
            or metric_failures
        ),
    }
    return audit, rows


def _smoke_recipe_aggregate(
    *,
    recipe: discovery.RecipeSpec,
    protocol: str,
    rows: Sequence[discovery.DiscoveryScoreRow],
    inner_folds: Sequence[int],
) -> discovery.RecipeAggregate:
    values: list[discovery.AggregateValue] = []
    for metric in discovery.required_metrics(protocol):
        vector = [float(row.metric_map()[metric]) for row in rows]
        fold_means = [
            float(
                np.mean(
                    [
                        row.metric_map()[metric]
                        for row in rows
                        if row.inner_fold == inner_fold
                    ]
                )
            )
            for inner_fold in inner_folds
        ]
        values.extend(
            (
                discovery.AggregateValue(f"mean_{metric}", round(float(np.mean(vector)), 12)),
                discovery.AggregateValue(
                    f"median_{metric}", round(float(statistics.median(vector)), 12)
                ),
                discovery.AggregateValue(
                    f"worst_inner_fold_mean_{metric}", round(min(fold_means), 12)
                ),
            )
        )
    payload = {
        "recipe": recipe.name,
        "recipe_order": recipe.order,
        "complexity_score": recipe.complexity_score,
        "score_rows": len(rows),
        "rank_values": [asdict(value) for value in values],
        "input_rows": [row.manifest() for row in rows],
        "smoke_only": True,
    }
    return discovery.RecipeAggregate(
        recipe=recipe.name,
        recipe_order=recipe.order,
        complexity_score=recipe.complexity_score,
        score_rows=len(rows),
        rank_values=tuple(values),
        aggregate_digest=canonical_digest(payload),
    )


def select_smoke_recipes(
    plan: discovery.NestedDiscoveryPlan,
    score_rows: Iterable[discovery.DiscoveryScoreRow],
) -> SmokeSelectionArtifact:
    audit, rows = audit_smoke_score_matrix(plan, score_rows)
    if not audit["passed"]:
        raise EndStateRunError("numerical smoke discovery returned an invalid score matrix")
    target = plan.targets[0]
    outer = plan.outer_folds[0]
    selections: list[discovery.OuterRecipeSelection] = []
    for protocol in plan.protocols:
        subset = tuple(row for row in rows if row.protocol == protocol)
        leaderboard = tuple(
            sorted(
                (
                    _smoke_recipe_aggregate(
                        recipe=recipe,
                        protocol=protocol,
                        rows=tuple(row for row in subset if row.recipe == recipe.name),
                        inner_folds=tuple(inner.fold for inner in outer.inner_folds),
                    )
                    for recipe in plan.recipes
                ),
                key=lambda item: tuple(-value for value in item.rank_vector())
                + (item.complexity_score, item.recipe_order, item.recipe),
            )
        )
        input_digest = canonical_digest([row.manifest() for row in subset])
        selection_payload = {
            "target": target,
            "protocol": protocol,
            "outer_fold": outer.outer_fold,
            "selected_recipe": leaderboard[0].recipe,
            "leaderboard": [item.manifest() for item in leaderboard],
            "input_score_digest": input_digest,
            "outer_test_scores_used": False,
        }
        selections.append(
            discovery.OuterRecipeSelection(
                target=target,
                protocol=protocol,
                outer_fold=outer.outer_fold,
                selected_recipe=leaderboard[0].recipe,
                leaderboard=leaderboard,
                input_score_digest=input_digest,
                outer_test_scores_used=False,
                digest=canonical_digest(selection_payload),
            )
        )
    ownership = discovery.audit_nested_ownership(plan)
    payload = {
        "schema_version": SMOKE_SELECTION_SCHEMA_VERSION,
        "plan_digest": plan.digest,
        "target": target,
        "outer_fold": outer.outer_fold,
        "discovery_seed": plan.discovery_seeds[0],
        "score_digest": audit["score_digest"],
        "score_rows": len(rows),
        "selections": [item.manifest() for item in selections],
        "ownership_audit_digest": ownership.digest,
        "outer_test_scores_used": False,
        "explicitly_nonpromotable": True,
    }
    return SmokeSelectionArtifact(
        schema_version=SMOKE_SELECTION_SCHEMA_VERSION,
        plan_digest=plan.digest,
        target=target,
        outer_fold=outer.outer_fold,
        discovery_seed=plan.discovery_seeds[0],
        score_digest=audit["score_digest"],
        score_rows=len(rows),
        selections=tuple(selections),
        ownership_audit_digest=ownership.digest,
        outer_test_scores_used=False,
        explicitly_nonpromotable=True,
        artifact_digest=canonical_digest(payload),
    )


def _load_smoke_selection_artifact(path: Path) -> SmokeSelectionArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selections: list[discovery.OuterRecipeSelection] = []
    for item in payload["selections"]:
        leaderboard = tuple(
            discovery.RecipeAggregate(
                recipe=str(entry["recipe"]),
                recipe_order=int(entry["recipe_order"]),
                complexity_score=int(entry["complexity_score"]),
                score_rows=int(entry["score_rows"]),
                rank_values=tuple(
                    discovery.AggregateValue(
                        name=str(value["name"]), value=float(value["value"])
                    )
                    for value in entry["rank_values"]
                ),
                aggregate_digest=str(entry["aggregate_digest"]),
            )
            for entry in item["leaderboard"]
        )
        selections.append(
            discovery.OuterRecipeSelection(
                target=str(item["target"]),
                protocol=str(item["protocol"]),
                outer_fold=int(item["outer_fold"]),
                selected_recipe=str(item["selected_recipe"]),
                leaderboard=leaderboard,
                input_score_digest=str(item["input_score_digest"]),
                outer_test_scores_used=bool(item["outer_test_scores_used"]),
                digest=str(item["digest"]),
            )
        )
    return SmokeSelectionArtifact(
        schema_version=str(payload["schema_version"]),
        plan_digest=str(payload["plan_digest"]),
        target=str(payload["target"]),
        outer_fold=int(payload["outer_fold"]),
        discovery_seed=int(payload["discovery_seed"]),
        score_digest=str(payload["score_digest"]),
        score_rows=int(payload["score_rows"]),
        selections=tuple(selections),
        ownership_audit_digest=str(payload["ownership_audit_digest"]),
        outer_test_scores_used=bool(payload["outer_test_scores_used"]),
        explicitly_nonpromotable=bool(payload["explicitly_nonpromotable"]),
        artifact_digest=str(payload["artifact_digest"]),
    )


def verify_smoke_selection_artifact(
    artifact: SmokeSelectionArtifact, plan: discovery.NestedDiscoveryPlan
) -> None:
    expected_keys = {
        (plan.targets[0], protocol, plan.outer_folds[0].outer_fold)
        for protocol in plan.protocols
    }
    observed_keys = {
        (item.target, item.protocol, item.outer_fold) for item in artifact.selections
    }
    payload = artifact.manifest(include_digest=False)
    checks = {
        "schema": artifact.schema_version == SMOKE_SELECTION_SCHEMA_VERSION,
        "plan": artifact.plan_digest == plan.digest,
        "target": artifact.target == plan.targets[0],
        "outer_fold": artifact.outer_fold == plan.outer_folds[0].outer_fold,
        "seed": artifact.discovery_seed == plan.discovery_seeds[0],
        "row_count": artifact.score_rows == len(smoke_expected_score_keys(plan)),
        "selection_keys": observed_keys == expected_keys,
        "recipes": all(
            item.selected_recipe in {recipe.name for recipe in plan.recipes}
            for item in artifact.selections
        ),
        "inner_only": artifact.outer_test_scores_used is False
        and all(not item.outer_test_scores_used for item in artifact.selections),
        "nonpromotable": artifact.explicitly_nonpromotable is True,
        "ownership": artifact.ownership_audit_digest
        == discovery.audit_nested_ownership(plan).digest,
        "digest": artifact.artifact_digest == canonical_digest(payload),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise EndStateRunError(f"smoke selection artifact failed verification: {failed}")


def load_and_verify_smoke_selection(
    output_root: Path, plan: discovery.NestedDiscoveryPlan
) -> SmokeSelectionArtifact:
    path = _smoke_selection_path(output_root)
    if not path.is_file():
        raise EndStateRunError("nonpromotable smoke discovery selection is missing")
    artifact = _load_smoke_selection_artifact(path)
    verify_smoke_selection_artifact(artifact, plan)
    return artifact


def load_and_verify_selection(
    output_root: Path, plan: discovery.NestedDiscoveryPlan
) -> discovery.DiscoverySelectionArtifact:
    path = _selection_path(output_root)
    if not path.is_file():
        raise EndStateRunError("discovery selection artifact is missing")
    artifact = _load_selection_artifact(path)
    discovery.verify_selection_artifact(artifact, plan=plan)
    if artifact.outer_test_scores_used:
        raise EndStateRunError("selection artifact used outer-test scores")
    return artifact


def dry_run_selection(
    plan: discovery.NestedDiscoveryPlan,
) -> discovery.DiscoverySelectionArtifact:
    """Create a non-evidentiary complete selection solely for plan expansion.

    A dry run needs concrete recipe names to enumerate confirmation cells, but
    it must not masquerade as measured discovery.  Deterministic synthetic
    inner scores are therefore kept in memory, never written as evidence, and
    the resulting artifact is referenced only from ``status=planned`` output.
    """

    rows: list[discovery.DiscoveryScoreRow] = []
    for target in plan.targets:
        for protocol in plan.protocols:
            for outer in plan.outer_folds:
                for recipe in plan.recipes:
                    # Earlier frozen recipe order wins this plan-only ranking.
                    quality = 1.0 - recipe.order * 0.01
                    for inner in outer.inner_folds:
                        for seed in plan.discovery_seeds:
                            metrics = (
                                {
                                    discovery.SPEARMAN: quality,
                                    discovery.TOP5_LIFT: quality / 10.0,
                                }
                                if protocol
                                in (
                                    discovery.PRIVILEGED_CONTINUOUS,
                                    discovery.ZERO_LABEL_CONTINUOUS,
                                )
                                else {discovery.TRAIN_Q90_PR_AUC: quality}
                            )
                            rows.append(
                                discovery.make_discovery_score_row(
                                    plan,
                                    target=target,
                                    protocol=protocol,
                                    outer_fold=outer.outer_fold,
                                    recipe=recipe.name,
                                    inner_fold=inner.fold,
                                    seed=seed,
                                    metrics=metrics,
                                )
                            )
    return discovery.select_nested_recipes(plan, rows)


def run_discovery_stage(
    *,
    args: argparse.Namespace,
    plan: discovery.NestedDiscoveryPlan,
    identity: RunIdentity,
    dataset: DenseDataset | None = None,
) -> Mapping[str, Any]:
    """Run or audit nested discovery.

    Full numerical execution is delegated to the local modeling module through
    ``execute_discovery`` when present.  Keeping that call behind an explicit
    API avoids silently falling back to historical AGAIN training code.
    """

    selection_path = (
        _smoke_selection_path(args.output_root)
        if args.smoke
        else _selection_path(args.output_root)
    )
    if selection_path.exists():
        artifact = (
            load_and_verify_smoke_selection(args.output_root, plan)
            if args.smoke
            else load_and_verify_selection(args.output_root, plan)
        )
        return {
            "status": "complete",
            "resumed": True,
            "selection_digest": artifact.artifact_digest,
            "score_rows": artifact.score_rows,
            "explicitly_nonpromotable": bool(args.smoke),
        }
    if args.audit_only:
        raise EndStateRunError("cannot audit discovery before its selection artifact exists")
    if args.dry_run:
        return {
            "status": "planned",
            "expected_score_rows": (
                len(smoke_expected_score_keys(plan))
                if args.smoke
                else discovery_expected_rows(plan)
            ),
            "outer_test_scores_used": False,
            "explicitly_nonpromotable": bool(args.smoke),
        }
    if dataset is None:
        raise EndStateRunError("numerical discovery requires the materialized dense dataset")
    try:
        from backend.scripts import veatic21_modeling as modeling
    except ImportError as exc:
        raise EndStateRunError("veatic21_modeling is required for numerical discovery") from exc
    executor = getattr(modeling, "execute_veatic21_nested_discovery", None)
    if executor is None:
        raise EndStateRunError(
            "veatic21_modeling does not expose execute_veatic21_nested_discovery; "
            "the runner will not fall back to historical fitted code"
        )
    score_rows = executor(
        args=args,
        plan=plan,
        dataset=dataset,
        run_identity_digest=identity.digest,
        output_root=Path(args.output_root) / "discovery",
        pca_parent_width=MAX_PCA_WIDTH,
        pca_slice_policy="leading_components_only",
        serial=True,
    )
    if args.smoke:
        audit, normalized = audit_smoke_score_matrix(plan, score_rows)
        if not audit["passed"]:
            raise EndStateRunError("numerical smoke discovery returned an invalid score matrix")
        artifact = select_smoke_recipes(plan, normalized)
        score_path = _smoke_score_rows_path(args.output_root)
    else:
        audit, normalized = discovery.audit_score_matrix(plan, score_rows)
        if not audit.passed:
            raise EndStateRunError("numerical discovery returned an invalid score matrix")
        artifact = discovery.select_nested_recipes(plan, normalized)
        score_path = _score_rows_path(args.output_root)
    atomic_json(score_path, [row.manifest() for row in normalized])
    atomic_json(selection_path, artifact.manifest())
    return {
        "status": "complete",
        "resumed": False,
        "selection_digest": artifact.artifact_digest,
        "score_rows": artifact.score_rows,
        "explicitly_nonpromotable": bool(args.smoke),
    }


def run_confirmation_stage(
    *,
    args: argparse.Namespace,
    plan: discovery.NestedDiscoveryPlan,
    identity: RunIdentity,
    dataset: DenseDataset | None = None,
) -> Mapping[str, Any]:
    if args.dry_run and not (
        _smoke_selection_path(args.output_root).is_file()
        if args.smoke
        else _selection_path(args.output_root).is_file()
    ):
        selections = dry_run_selection(plan)
    elif args.smoke:
        selections = load_and_verify_smoke_selection(args.output_root, plan)
    else:
        selections = load_and_verify_selection(args.output_root, plan)
    cells = build_confirmation_matrix(selections, plan, smoke=bool(args.smoke))
    audit = audit_confirmation_matrix(cells, plan=plan, smoke=bool(args.smoke))
    atomic_json(
        Path(args.output_root) / "confirmation" / "matrix_plan.json",
        {
            "schema_version": "veatic21_confirmation_matrix_plan_v1",
            "run_identity_digest": identity.digest,
            "selection_digest": selections.artifact_digest,
            "audit": audit.manifest(),
            "cells": [cell.manifest() for cell in cells],
        },
    )
    preflight = matrix_completion_audit(
        output_root=Path(args.output_root),
        cells=cells,
        run_identity_digest=identity.digest,
        require_complete=False,
    )
    if args.audit_only:
        if not preflight["passed"]:
            raise EndStateRunError("confirmation audit found an incomplete matrix")
        epoch_provenance = collect_best_epoch_provenance(
            output_root=Path(args.output_root),
            cells=cells,
            run_identity_digest=identity.digest,
            smoke=bool(args.smoke),
            write=False,
        )
        return {
            "status": "complete",
            "matrix": preflight,
            "best_epochs": epoch_provenance,
            "resumed": True,
        }
    if args.dry_run:
        return {
            "status": "planned",
            "expected_rows": len(cells),
            "matrix_digest": audit.digest,
        }
    if preflight["passed"]:
        epoch_provenance = collect_best_epoch_provenance(
            output_root=Path(args.output_root),
            cells=cells,
            run_identity_digest=identity.digest,
            smoke=bool(args.smoke),
            write=True,
        )
        return {
            "status": "complete",
            "matrix": preflight,
            "best_epochs": epoch_provenance,
            "resumed": True,
        }
    if dataset is None:
        raise EndStateRunError("numerical confirmation requires the materialized dense dataset")
    try:
        from backend.scripts import veatic21_modeling as modeling
    except ImportError as exc:
        raise EndStateRunError("veatic21_modeling is required for confirmation") from exc
    executor = getattr(modeling, "execute_veatic21_confirmation_cell", None)
    if executor is None:
        raise EndStateRunError(
            "veatic21_modeling does not expose execute_veatic21_confirmation_cell; "
            "the runner refuses an unsealed compatibility fallback"
        )
    # Strictly serial.  Each call must atomically seal its own checkpoint and
    # prediction before the next cell starts.  Response-free zero-label cells
    # precede the descriptive teacher by construction.
    for cell in cells:
        path = prediction_path(args.output_root, cell)
        if path.exists() or path.with_suffix(".json").exists():
            verify_prediction_shard(
                path, cell=cell, run_identity_digest=identity.digest
            )
            continue
        result = executor(
            args=args,
            plan=plan,
            selection=selections,
            cell=cell,
            dataset=dataset,
            run_identity_digest=identity.digest,
            output_root=Path(args.output_root),
            serial=True,
            pca_parent_width=MAX_PCA_WIDTH,
            pca_slice_policy="leading_components_only",
        )
        if not isinstance(result, Mapping):
            raise EndStateRunError("confirmation executor returned no sealed result mapping")
        seal_prediction_shard(
            path=path,
            cell=cell,
            row_indices=np.asarray(result["row_indices"]),
            video_ids=np.asarray(result["video_ids"]),
            y_true=np.asarray(result["y_true"]),
            prediction=np.asarray(result["prediction"]),
            event_threshold=float(result["event_threshold"]),
            checkpoint=Path(result["checkpoint"]) if result.get("checkpoint") else None,
            run_identity_digest=identity.digest,
            extra_provenance=dict(result.get("provenance", {})),
        )
    completed = matrix_completion_audit(
        output_root=Path(args.output_root),
        cells=cells,
        run_identity_digest=identity.digest,
        require_complete=True,
    )
    atomic_json(Path(args.output_root) / "confirmation" / "matrix_audit.json", completed)
    epoch_provenance = collect_best_epoch_provenance(
        output_root=Path(args.output_root),
        cells=cells,
        run_identity_digest=identity.digest,
        smoke=bool(args.smoke),
        write=True,
    )
    return {
        "status": "complete",
        "matrix": completed,
        "best_epochs": epoch_provenance,
        "resumed": False,
    }


def collect_best_epoch_provenance(
    *,
    output_root: Path,
    cells: Sequence[MatrixCell],
    run_identity_digest: str,
    smoke: bool,
    write: bool,
) -> Mapping[str, Any]:
    """Seal real-lane member epochs used by the later all-video refit."""

    real_lanes = {
        discovery.PRIVILEGED_CONTINUOUS: "real_residual",
        discovery.PRIVILEGED_BINARY: "real_residual",
        discovery.ZERO_LABEL_CONTINUOUS: "video_supervised_temporal",
    }
    grouped: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for cell in cells:
        if cell.row_kind != MEMBER_KIND or cell.lane != real_lanes[cell.endpoint]:
            continue
        manifest = verify_prediction_shard(
            prediction_path(output_root, cell),
            cell=cell,
            run_identity_digest=run_identity_digest,
        )
        provenance = manifest.get("extra_provenance")
        if not isinstance(provenance, Mapping):
            raise EndStateRunError("real member prediction lacks training provenance")
        epoch = provenance.get("best_epoch")
        recipe_order = provenance.get("recipe_order")
        if not isinstance(epoch, int) or epoch < 1 or not isinstance(recipe_order, int):
            raise EndStateRunError("real member prediction lacks valid best-epoch provenance")
        grouped[(cell.target, cell.endpoint, recipe_order, cell.recipe)].append(epoch)
    rows = [
        {
            "target": target,
            "protocol": protocol,
            "recipe_order": recipe_order,
            "recipe": recipe,
            "best_epochs": sorted(values),
            "best_epochs_digest": canonical_digest(sorted(values)),
        }
        for (target, protocol, recipe_order, recipe), values in sorted(grouped.items())
    ]
    if not rows:
        raise EndStateRunError("confirmation produced no real-lane best epochs")
    payload_without_digest = {
        "schema_version": "veatic21_confirmation_best_epochs_v1",
        "run_identity_digest": run_identity_digest,
        "smoke": bool(smoke),
        "explicitly_nonpromotable": bool(smoke),
        "rows": rows,
    }
    payload = {
        **payload_without_digest,
        "artifact_digest": canonical_digest(payload_without_digest),
    }
    path = (
        Path(output_root) / "confirmation" / "smoke_best_epochs.json"
        if smoke
        else Path(output_root) / "confirmation" / "best_epochs.json"
    )
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise EndStateRunError("confirmation best-epoch provenance drift")
    elif write:
        atomic_json(path, payload)
    else:
        raise EndStateRunError("confirmation best-epoch provenance is missing")
    return payload


def _load_best_epochs(output_root: Path) -> dict[tuple[str, str, Any], tuple[int, ...]]:
    path = Path(output_root) / "confirmation" / "best_epochs.json"
    if not path.is_file():
        raise EndStateRunError("confirmation best-epoch provenance is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str, Any], tuple[int, ...]] = {}
    for row in payload.get("rows", []):
        recipe_key: Any = row.get("recipe_order", row.get("recipe"))
        key = (str(row["target"]), str(row["protocol"]), recipe_key)
        if key in result:
            raise EndStateRunError("duplicate best-epoch provenance key")
        result[key] = tuple(int(value) for value in row["best_epochs"])
    return result


def run_final_stage(
    *,
    args: argparse.Namespace,
    plan: discovery.NestedDiscoveryPlan,
    identity: RunIdentity,
    dataset: DenseDataset | None = None,
) -> Mapping[str, Any]:
    if args.smoke and not args.dry_run:
        raise EndStateRunError(
            "--smoke cannot execute the all-124 final refit; test that executor through its bounded contract tests"
        )
    if args.dry_run:
        selections = (
            dry_run_selection(plan)
            if not _selection_path(args.output_root).is_file()
            else load_and_verify_selection(args.output_root, plan)
        )
        global_selections = select_global_recipes(selections, plan)
        return {
            "status": "planned",
            "video_count": 124,
            "reserve_count": 0,
            "selection_frozen": True,
            "no_in_sample_metric_claim": True,
            "global_selection_count": len(global_selections),
        }
    selections = load_and_verify_selection(args.output_root, plan)
    epochs = _load_best_epochs(args.output_root)
    export_contract = build_final_export_contract(
        identity=identity,
        plan=plan,
        selections=selections,
        best_epochs=epochs,
        max_epochs=int(args.max_epochs),
    )
    audit_final_export_contract(export_contract)
    path = Path(args.output_root) / "final" / "export_contract.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != export_contract:
            raise EndStateRunError("final export resume contract mismatch")
    else:
        atomic_json(path, export_contract)
    completion_path = Path(args.output_root) / "final" / "export_complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("export_contract_digest") != export_contract["export_contract_digest"]:
            raise EndStateRunError("final export completion belongs to another contract")
        return {"status": "complete", "resumed": True, "completion": completion}
    if args.audit_only:
        raise EndStateRunError("final export has not completed")
    if dataset is None:
        raise EndStateRunError("final refit requires the materialized dense dataset")
    try:
        from backend.scripts import veatic21_modeling as modeling
    except ImportError as exc:
        raise EndStateRunError("veatic21_modeling is required for final refit") from exc
    executor = getattr(modeling, "execute_veatic21_all124_refit", None)
    if executor is None:
        raise EndStateRunError(
            "veatic21_modeling does not expose execute_veatic21_all124_refit; "
            "the runner refuses to relabel confirmation checkpoints as all-video exports"
        )
    result = executor(
        args=args,
        plan=plan,
        export_contract=export_contract,
        dataset=dataset,
        output_root=Path(args.output_root) / "final",
        serial=True,
        pca_parent_width=MAX_PCA_WIDTH,
        pca_slice_policy="leading_components_only",
        score_training_rows=False,
    )
    if not isinstance(result, Mapping):
        raise EndStateRunError("final refit executor returned no completion mapping")
    artifacts = [Path(value) for value in result.get("artifacts", [])]
    if not artifacts or any(not path.is_file() for path in artifacts):
        raise EndStateRunError("final refit did not return complete artifact files")
    completion_payload = {
        "schema_version": "veatic21_all124_export_completion_v1",
        "export_contract_digest": export_contract["export_contract_digest"],
        "artifact_count": len(artifacts),
        "artifacts": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in artifacts
        ],
        "all_124_refit": True,
        "in_sample_metrics_reported": False,
        "source_cache_mutated": False,
        "completed_at": utc_now(),
    }
    completion = {
        **completion_payload,
        "completion_digest": canonical_digest(completion_payload),
    }
    atomic_json(completion_path, completion)
    return {"status": "complete", "resumed": False, "completion": completion}


def _canonical_cache(args: argparse.Namespace) -> compact.Veatic21CompactCache:
    return compact.Veatic21CompactCache(
        args.cache_root,
        upstream_root=args.upstream_root,
        identity_manifest_path=args.identity_manifest,
        verify_checksums=not bool(args.skip_checksums),
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    validate_cli_args(args)
    endstate.validate_endstate_contract()
    cache = _canonical_cache(args)
    report = cache.validate()
    seal = dataset_seal_from_report(report)
    plan = build_plan(seal)
    identity = build_run_identity(args=args, seal=seal, plan=plan)
    manifest = write_plan_artifacts(
        output_root=args.output_root,
        seal=seal,
        plan=plan,
        identity=identity,
        args=args,
    )
    if args.smoke and args.stage == "final" and not args.dry_run:
        raise EndStateRunError("a smoke run cannot execute the all-124 final refit")
    stages = (
        (("discovery", "confirmation") if args.smoke else ("discovery", "confirmation", "final"))
        if args.stage == "all"
        else (args.stage,)
    )
    dataset = None
    if not args.dry_run and not args.audit_only:
        shared_root = Path(
            args.shared_derived_root or (Path(args.output_root) / "derived")
        )
        dataset = materialize_dense_dataset(
            cache=cache,
            seal=seal,
            shared_root=shared_root,
        )
    results: dict[str, Any] = {}
    for stage in stages:
        if stage == "discovery":
            results[stage] = run_discovery_stage(
                args=args, plan=plan, identity=identity, dataset=dataset
            )
        elif stage == "confirmation":
            results[stage] = run_confirmation_stage(
                args=args, plan=plan, identity=identity, dataset=dataset
            )
        elif stage == "final":
            results[stage] = run_final_stage(
                args=args, plan=plan, identity=identity, dataset=dataset
            )
        else:  # pragma: no cover - argparse prevents this path
            raise EndStateRunError(f"unknown stage {stage}")
    canonical_gates_passed = bool(
        results.get("confirmation", {}).get("canonical_gates_passed", False)
    )
    promotable = bool(
        canonical_gates_passed
        and not args.dry_run
        and not args.audit_only
        and not args.smoke
        and not args.skip_checksums
    )
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_identity_digest": identity.digest,
        "stage_requested": args.stage,
        "smoke": bool(args.smoke),
        "promotable": promotable,
        "canonical_gates_passed": canonical_gates_passed,
        "dry_run": bool(args.dry_run),
        "results": results,
        "manifest_digest": canonical_digest(manifest),
        "finished_at": utc_now(),
    }
    atomic_json(Path(args.output_root) / "run_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        print(f"VEATIC 2.1 end-state run failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DISCOVERY_PROTOCOLS",
    "DatasetSeal",
    "EndStateRunError",
    "FINAL_EXPORT_SCHEMA_VERSION",
    "GlobalRecipeSelection",
    "INNER_FOLD_COUNT",
    "MAX_PCA_WIDTH",
    "MatrixAudit",
    "MatrixCell",
    "PcaRequest",
    "RUN_SCHEMA_VERSION",
    "RunIdentity",
    "STAGES",
    "assert_no_outer_leakage",
    "assert_zero_label_schema",
    "audit_confirmation_matrix",
    "audit_final_export_contract",
    "build_confirmation_matrix",
    "build_final_export_contract",
    "build_parser",
    "build_plan",
    "build_run_identity",
    "canonical_digest",
    "dataset_seal_from_report",
    "derive_fixed_epoch",
    "dry_run_selection",
    "discovery_expected_rows",
    "matrix_completion_audit",
    "pca_requests_for_plan",
    "pca_slice_manifest",
    "prediction_path",
    "scientific_settings",
    "seal_prediction_shard",
    "select_global_recipes",
    "synthetic_dataset_seal",
    "validate_cli_args",
    "validate_dataset_seal",
    "verify_prediction_shard",
    "write_or_verify_run_identity",
]
