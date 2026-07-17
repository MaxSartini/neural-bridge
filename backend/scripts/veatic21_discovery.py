"""Leakage-safe nested recipe discovery for the VEATIC 2.1 end state.

This module is deliberately limited to ownership and selection logic.  It does
not fit PCA, models, thresholds, or baselines and it never receives outer-test
metrics.  Every outer fold gets its own recipe decision from a complete matrix
of inner-validation scores drawn exclusively from that outer fold's training
videos.

The default contract uses all 124 VEATIC videos, five deterministic
row-balanced outer grouped-video folds, three deterministic row-balanced inner
grouped-video folds, the six recipes frozen in
``veatic21_endstate_contract.py``, and discovery seeds that are disjoint from
all confirmation seeds.  Generic recipe mappings remain supported for an
isolated consumer that does not have the end-state contract available.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "veatic21_nested_discovery_v1"
EXPECTED_VIDEO_COUNT = 124
OUTER_FOLD_COUNT = 5
INNER_FOLD_COUNT = 3
INNER_VALIDATION_SCOPE = "inner_validation"

PRIVILEGED_CONTINUOUS = "privileged_continuous"
PRIVILEGED_BINARY = "privileged_binary"
ZERO_LABEL_CONTINUOUS = "zero_label_continuous"
ZERO_LABEL_BINARY = "zero_label_binary"
SUPPORTED_PROTOCOLS = (
    PRIVILEGED_CONTINUOUS,
    PRIVILEGED_BINARY,
    ZERO_LABEL_CONTINUOUS,
    ZERO_LABEL_BINARY,
)

SPEARMAN = "spearman"
TOP5_LIFT = "top_5pct_lift"
TRAIN_Q90_PR_AUC = "train_q90_pr_auc"

_CONTINUOUS_PROTOCOLS = frozenset((PRIVILEGED_CONTINUOUS, ZERO_LABEL_CONTINUOUS))
_BINARY_PROTOCOLS = frozenset((PRIVILEGED_BINARY, ZERO_LABEL_BINARY))
_OUTER_SCORE_TOKENS = (
    "outer_test",
    "outer-test",
    "outer.test",
    "outer_score",
    "outer-score",
    "outer_metric",
    "outer-metric",
    "held_out",
    "held-out",
    "heldout",
    "holdout",
)


try:  # The fallback keeps this selector usable as an isolated contract module.
    from backend.scripts import veatic21_endstate_contract as _endstate_contract
except ImportError:  # pragma: no cover - exercised only outside this repository.
    _endstate_contract = None


class DiscoveryContractError(ValueError):
    """Raised when discovery ownership, matrix, or provenance fails closed."""


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest of a JSON-safe value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _video_sort_key(value: str) -> tuple[int, int | str, str]:
    text = str(value)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def _stable_video_hash(namespace: str, video_id: str) -> str:
    return hashlib.blake2b(
        f"{namespace}|{video_id}".encode("utf-8"), digest_size=16
    ).hexdigest()


def _ordered_unique_strings(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if not normalized or any(not value for value in normalized):
        raise DiscoveryContractError(f"{name} must contain non-empty values")
    if len(normalized) != len(set(normalized)):
        raise DiscoveryContractError(f"{name} must be unique")
    return normalized


def _ordered_unique_seeds(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if not values:
        raise DiscoveryContractError(f"{name} must not be empty")
    seeds: list[int] = []
    for raw in values:
        if isinstance(raw, bool):
            raise DiscoveryContractError(f"{name} must contain integer seeds")
        seed = int(raw)
        if seed != raw:
            raise DiscoveryContractError(f"{name} must contain integer seeds")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise DiscoveryContractError(f"{name} must be unique")
    return tuple(seeds)


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = dict(value)
    elif is_dataclass(value):
        raw = asdict(value)
    elif hasattr(value, "__dict__"):
        raw = dict(vars(value))
    else:
        raise DiscoveryContractError("Recipes must be dataclasses or mappings")
    try:
        encoded = json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DiscoveryContractError("Recipe mappings must be finite and JSON-safe") from exc
    if not isinstance(normalized, dict):
        raise DiscoveryContractError("Recipe payload must normalize to an object")
    return normalized


@dataclass(frozen=True)
class RecipeSpec:
    """Immutable normalized recipe with a prespecified complexity tie-break."""

    order: int
    name: str
    feature_family: str
    pca_width: int
    head: str
    causal_rows: int
    complexity_score: int
    payload_json: str
    digest: str

    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def manifest(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "name": self.name,
            "feature_family": self.feature_family,
            "pca_width": self.pca_width,
            "head": self.head,
            "causal_rows": self.causal_rows,
            "complexity_score": self.complexity_score,
            "payload": self.payload(),
            "digest": self.digest,
        }


def _recipe_spec(value: Any, order: int) -> RecipeSpec:
    if isinstance(value, RecipeSpec):
        if value.order == int(order):
            return value
        value = value.payload()
    payload = _json_mapping(value)
    try:
        name = str(payload["name"])
    except KeyError as exc:
        raise DiscoveryContractError("Every recipe needs a name") from exc
    if not name:
        raise DiscoveryContractError("Recipe names must be non-empty")
    feature_family = str(payload.get("feature_family", "unspecified"))
    head = str(payload.get("head", "unspecified"))
    try:
        pca_width = int(payload.get("pca_width", 0))
        causal_rows = int(payload.get("causal_rows", 1))
    except (TypeError, ValueError) as exc:
        raise DiscoveryContractError("Recipe PCA width and causal rows must be integers") from exc
    if pca_width < 0 or causal_rows < 1:
        raise DiscoveryContractError("Recipe PCA width and causal rows are invalid")
    explicit_complexity = payload.get("complexity_score")
    if explicit_complexity is None:
        complexity = max(1, pca_width) * causal_rows
    else:
        try:
            complexity = int(explicit_complexity)
        except (TypeError, ValueError) as exc:
            raise DiscoveryContractError("Recipe complexity must be a positive integer") from exc
        if complexity < 1:
            raise DiscoveryContractError("Recipe complexity must be a positive integer")
    payload_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    digest = canonical_digest(
        {
            "order": int(order),
            "payload": payload,
            "complexity_score": complexity,
        }
    )
    return RecipeSpec(
        order=int(order),
        name=name,
        feature_family=feature_family,
        pca_width=pca_width,
        head=head,
        causal_rows=causal_rows,
        complexity_score=complexity,
        payload_json=payload_json,
        digest=digest,
    )


def canonical_recipe_specs() -> tuple[RecipeSpec, ...]:
    """Return the exact bounded recipe grid from the end-state contract."""

    if _endstate_contract is None:
        raise DiscoveryContractError(
            "No end-state contract is available; pass recipe mappings explicitly"
        )
    return tuple(
        _recipe_spec(recipe, order)
        for order, recipe in enumerate(_endstate_contract.DISCOVERY_RECIPES)
    )


def normalize_recipes(
    recipes: Sequence[Any] | None = None,
    *,
    enforce_canonical: bool | None = None,
) -> tuple[RecipeSpec, ...]:
    """Normalize recipes, enforcing the end-state grid whenever it is available."""

    canonical = canonical_recipe_specs() if _endstate_contract is not None else None
    if recipes is None:
        if canonical is None:
            raise DiscoveryContractError("Explicit recipe mappings are required")
        return canonical
    specs = tuple(_recipe_spec(recipe, order) for order, recipe in enumerate(recipes))
    if not specs or len({recipe.name for recipe in specs}) != len(specs):
        raise DiscoveryContractError("Discovery recipes must be non-empty and uniquely named")
    should_enforce = canonical is not None if enforce_canonical is None else bool(enforce_canonical)
    if should_enforce and specs != canonical:
        raise DiscoveryContractError(
            "Discovery recipes must equal the exact bounded end-state recipe grid"
        )
    return specs


@dataclass(frozen=True)
class GroupedOwnership:
    fold: int
    train_videos: tuple[str, ...]
    validation_videos: tuple[str, ...]
    train_rows: int
    validation_rows: int
    digest: str

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OuterDiscoveryFold:
    outer_fold: int
    train_videos: tuple[str, ...]
    test_videos: tuple[str, ...]
    train_rows: int
    test_rows: int
    inner_folds: tuple[GroupedOwnership, ...]
    digest: str

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_fold": self.outer_fold,
            "train_videos": list(self.train_videos),
            "test_videos": list(self.test_videos),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "inner_folds": [fold.manifest() for fold in self.inner_folds],
            "digest": self.digest,
        }


@dataclass(frozen=True)
class OwnershipAudit:
    passed: bool
    checks: tuple[tuple[str, bool], ...]
    failed_checks: tuple[str, ...]
    video_count: int
    outer_fold_count: int
    inner_fold_count: int
    digest: str

    def check_map(self) -> dict[str, bool]:
        return dict(self.checks)


@dataclass(frozen=True)
class NestedDiscoveryPlan:
    schema_version: str
    namespace: str
    video_count: int
    row_counts: tuple[tuple[str, int], ...]
    targets: tuple[str, ...]
    protocols: tuple[str, ...]
    recipes: tuple[RecipeSpec, ...]
    discovery_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    outer_folds: tuple[OuterDiscoveryFold, ...]
    inner_fold_count: int
    digest: str

    def manifest(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "namespace": self.namespace,
            "video_count": self.video_count,
            "row_counts": [[video, rows] for video, rows in self.row_counts],
            "targets": list(self.targets),
            "protocols": list(self.protocols),
            "recipes": [recipe.manifest() for recipe in self.recipes],
            "discovery_seeds": list(self.discovery_seeds),
            "confirmation_seeds": list(self.confirmation_seeds),
            "outer_folds": [fold.manifest() for fold in self.outer_folds],
            "inner_fold_count": self.inner_fold_count,
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def outer(self, outer_fold: int) -> OuterDiscoveryFold:
        for fold in self.outer_folds:
            if fold.outer_fold == int(outer_fold):
                return fold
        raise DiscoveryContractError(f"Unknown outer fold {outer_fold}")

    def inner(self, outer_fold: int, inner_fold: int) -> GroupedOwnership:
        outer = self.outer(outer_fold)
        for fold in outer.inner_folds:
            if fold.fold == int(inner_fold):
                return fold
        raise DiscoveryContractError(
            f"Unknown inner fold {inner_fold} for outer fold {outer_fold}"
        )


def _row_balanced_partition(
    row_counts: Mapping[str, int],
    videos: Sequence[str],
    *,
    fold_count: int,
    namespace: str,
) -> tuple[tuple[str, ...], ...]:
    selected = tuple(sorted({str(video) for video in videos}, key=_video_sort_key))
    if fold_count < 2 or fold_count > len(selected):
        raise DiscoveryContractError("Grouped fold count is incompatible with video count")
    if len(selected) != len(videos):
        raise DiscoveryContractError("Grouped video ownership contains duplicates")
    if any(video not in row_counts or int(row_counts[video]) <= 0 for video in selected):
        raise DiscoveryContractError("Every grouped video needs a positive row count")
    buckets: list[list[str]] = [[] for _ in range(fold_count)]
    totals = [0] * fold_count
    ordered = sorted(
        selected,
        key=lambda video: (
            -int(row_counts[video]),
            _stable_video_hash(namespace, video),
            _video_sort_key(video),
        ),
    )
    for video in ordered:
        destination = min(
            range(fold_count),
            key=lambda index: (totals[index], len(buckets[index]), index),
        )
        buckets[destination].append(video)
        totals[destination] += int(row_counts[video])
    return tuple(
        tuple(sorted(bucket, key=_video_sort_key)) for bucket in buckets
    )


def _ownership(
    *,
    fold: int,
    all_videos: Sequence[str],
    validation_videos: Sequence[str],
    row_counts: Mapping[str, int],
    namespace: str,
) -> GroupedOwnership:
    validation = tuple(sorted({str(video) for video in validation_videos}, key=_video_sort_key))
    validation_set = set(validation)
    train = tuple(video for video in all_videos if video not in validation_set)
    payload = {
        "namespace": namespace,
        "fold": int(fold),
        "train_videos": list(train),
        "validation_videos": list(validation),
        "train_rows": sum(int(row_counts[video]) for video in train),
        "validation_rows": sum(int(row_counts[video]) for video in validation),
    }
    return GroupedOwnership(
        fold=int(fold),
        train_videos=train,
        validation_videos=validation,
        train_rows=int(payload["train_rows"]),
        validation_rows=int(payload["validation_rows"]),
        digest=canonical_digest(payload),
    )


def _defaults() -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    if _endstate_contract is None:
        targets: tuple[str, ...] = ()
        discovery = (20260716, 20260717, 20260718)
        confirmation = tuple(range(20260801, 20260813))
    else:
        targets = tuple(target.name for target in _endstate_contract.PRIMARY_TARGETS)
        discovery = tuple(_endstate_contract.DISCOVERY_SEEDS)
        confirmation = tuple(
            dict.fromkeys(
                tuple(_endstate_contract.PRIVILEGED_CONFIRMATION_SEEDS)
                + tuple(_endstate_contract.ZERO_LABEL_CONFIRMATION_SEEDS)
            )
        )
    return targets, discovery, confirmation


def build_nested_discovery_plan(
    row_counts: Mapping[str, int],
    *,
    targets: Sequence[str] | None = None,
    protocols: Sequence[str] = SUPPORTED_PROTOCOLS,
    recipes: Sequence[Any] | None = None,
    discovery_seeds: Sequence[int] | None = None,
    confirmation_seeds: Sequence[int] | None = None,
    expected_video_count: int = EXPECTED_VIDEO_COUNT,
    outer_fold_count: int = OUTER_FOLD_COUNT,
    inner_fold_count: int = INNER_FOLD_COUNT,
    namespace: str = "veatic21_nested_discovery_20260716_v1",
    enforce_canonical_recipes: bool | None = None,
) -> NestedDiscoveryPlan:
    """Build and seal deterministic outer/inner grouped-video ownership."""

    default_targets, default_discovery, default_confirmation = _defaults()
    target_names = _ordered_unique_strings(
        tuple(default_targets if targets is None else targets), name="targets"
    )
    protocol_names = _ordered_unique_strings(tuple(protocols), name="protocols")
    unknown_protocols = set(protocol_names) - set(SUPPORTED_PROTOCOLS)
    if unknown_protocols:
        raise DiscoveryContractError(
            f"Unsupported discovery protocols: {sorted(unknown_protocols)}"
        )
    recipe_specs = normalize_recipes(
        recipes, enforce_canonical=enforce_canonical_recipes
    )
    discovery = _ordered_unique_seeds(
        tuple(default_discovery if discovery_seeds is None else discovery_seeds),
        name="discovery_seeds",
    )
    confirmation = _ordered_unique_seeds(
        tuple(default_confirmation if confirmation_seeds is None else confirmation_seeds),
        name="confirmation_seeds",
    )
    overlap = set(discovery) & set(confirmation)
    if overlap:
        raise DiscoveryContractError(
            f"Discovery and confirmation seeds must be disjoint: {sorted(overlap)}"
        )

    normalized_counts: dict[str, int] = {}
    for raw_video, raw_rows in row_counts.items():
        video = str(raw_video)
        if not video or video in normalized_counts:
            raise DiscoveryContractError("Video identifiers must be unique after normalization")
        if isinstance(raw_rows, bool):
            raise DiscoveryContractError("Row counts must be positive integers")
        rows = int(raw_rows)
        if rows != raw_rows or rows <= 0:
            raise DiscoveryContractError("Row counts must be positive integers")
        normalized_counts[video] = rows
    if len(normalized_counts) != int(expected_video_count):
        raise DiscoveryContractError(
            f"Expected all {expected_video_count} videos, found {len(normalized_counts)}"
        )
    if int(outer_fold_count) != OUTER_FOLD_COUNT:
        raise DiscoveryContractError("VEATIC 2.1 discovery requires exactly five outer folds")
    if int(inner_fold_count) < 2:
        raise DiscoveryContractError("Nested discovery requires at least two inner folds")

    all_videos = tuple(sorted(normalized_counts, key=_video_sort_key))
    outer_buckets = _row_balanced_partition(
        normalized_counts,
        all_videos,
        fold_count=int(outer_fold_count),
        namespace=f"{namespace}|outer",
    )
    outer_folds: list[OuterDiscoveryFold] = []
    for outer_index, test_videos in enumerate(outer_buckets, start=1):
        test_set = set(test_videos)
        train_videos = tuple(video for video in all_videos if video not in test_set)
        inner_namespace = f"{namespace}|outer={outer_index}|inner"
        inner_buckets = _row_balanced_partition(
            normalized_counts,
            train_videos,
            fold_count=int(inner_fold_count),
            namespace=inner_namespace,
        )
        inner = tuple(
            _ownership(
                fold=inner_index,
                all_videos=train_videos,
                validation_videos=validation,
                row_counts=normalized_counts,
                namespace=inner_namespace,
            )
            for inner_index, validation in enumerate(inner_buckets, start=1)
        )
        payload = {
            "namespace": namespace,
            "outer_fold": outer_index,
            "train_videos": list(train_videos),
            "test_videos": list(test_videos),
            "train_rows": sum(normalized_counts[video] for video in train_videos),
            "test_rows": sum(normalized_counts[video] for video in test_videos),
            "inner_folds": [fold.manifest() for fold in inner],
        }
        outer_folds.append(
            OuterDiscoveryFold(
                outer_fold=outer_index,
                train_videos=train_videos,
                test_videos=test_videos,
                train_rows=int(payload["train_rows"]),
                test_rows=int(payload["test_rows"]),
                inner_folds=inner,
                digest=canonical_digest(payload),
            )
        )

    plan_without_digest = {
        "schema_version": SCHEMA_VERSION,
        "namespace": str(namespace),
        "video_count": len(all_videos),
        "row_counts": [[video, normalized_counts[video]] for video in all_videos],
        "targets": list(target_names),
        "protocols": list(protocol_names),
        "recipes": [recipe.manifest() for recipe in recipe_specs],
        "discovery_seeds": list(discovery),
        "confirmation_seeds": list(confirmation),
        "outer_folds": [fold.manifest() for fold in outer_folds],
        "inner_fold_count": int(inner_fold_count),
    }
    plan = NestedDiscoveryPlan(
        schema_version=SCHEMA_VERSION,
        namespace=str(namespace),
        video_count=len(all_videos),
        row_counts=tuple((video, normalized_counts[video]) for video in all_videos),
        targets=target_names,
        protocols=protocol_names,
        recipes=recipe_specs,
        discovery_seeds=discovery,
        confirmation_seeds=confirmation,
        outer_folds=tuple(outer_folds),
        inner_fold_count=int(inner_fold_count),
        digest=canonical_digest(plan_without_digest),
    )
    audit = audit_nested_ownership(plan)
    if not audit.passed:
        raise DiscoveryContractError(
            f"Nested ownership audit failed: {list(audit.failed_checks)}"
        )
    validate_plan_digest(plan)
    return plan


def validate_plan_digest(plan: NestedDiscoveryPlan) -> None:
    observed = canonical_digest(plan.manifest(include_digest=False))
    if observed != plan.digest:
        raise DiscoveryContractError("Nested discovery plan digest mismatch")


def audit_nested_ownership(plan: NestedDiscoveryPlan) -> OwnershipAudit:
    """Audit complete, disjoint outer ownership and nested train-only ownership."""

    all_videos = {video for video, _ in plan.row_counts}
    counts = dict(plan.row_counts)
    outer_test = [video for fold in plan.outer_folds for video in fold.test_videos]
    outer_disjoint = all(
        not set(fold.train_videos) & set(fold.test_videos)
        and set(fold.train_videos) | set(fold.test_videos) == all_videos
        for fold in plan.outer_folds
    )
    all_once = len(outer_test) == len(set(outer_test)) == len(all_videos) and set(
        outer_test
    ) == all_videos
    inner_partition = True
    outer_excluded = True
    row_counts_match = True
    inner_counts_match = True
    for outer in plan.outer_folds:
        outer_train = set(outer.train_videos)
        outer_test_set = set(outer.test_videos)
        validations = [
            video for inner in outer.inner_folds for video in inner.validation_videos
        ]
        inner_partition &= (
            len(validations) == len(set(validations)) == len(outer_train)
            and set(validations) == outer_train
            and all(
                not set(inner.train_videos) & set(inner.validation_videos)
                and set(inner.train_videos) | set(inner.validation_videos) == outer_train
                for inner in outer.inner_folds
            )
        )
        outer_excluded &= all(
            not outer_test_set
            & (set(inner.train_videos) | set(inner.validation_videos))
            for inner in outer.inner_folds
        )
        row_counts_match &= (
            outer.train_rows == sum(counts[video] for video in outer.train_videos)
            and outer.test_rows == sum(counts[video] for video in outer.test_videos)
        )
        inner_counts_match &= all(
            inner.train_rows == sum(counts[video] for video in inner.train_videos)
            and inner.validation_rows
            == sum(counts[video] for video in inner.validation_videos)
            for inner in outer.inner_folds
        )
    checks = (
        ("all_videos_enter_outer_cv", plan.video_count == EXPECTED_VIDEO_COUNT),
        ("exactly_five_outer_folds", len(plan.outer_folds) == OUTER_FOLD_COUNT),
        ("outer_train_test_disjoint", outer_disjoint),
        ("each_video_outer_test_once", all_once),
        ("inner_validation_partitions_outer_train", inner_partition),
        ("outer_test_excluded_from_inner", outer_excluded),
        ("outer_row_counts_match", row_counts_match),
        ("inner_row_counts_match", inner_counts_match),
        (
            "discovery_confirmation_seeds_disjoint",
            not set(plan.discovery_seeds) & set(plan.confirmation_seeds),
        ),
    )
    failed = tuple(name for name, passed in checks if not passed)
    payload = {
        "checks": list(checks),
        "plan_digest": plan.digest,
        "video_count": plan.video_count,
        "outer_fold_count": len(plan.outer_folds),
        "inner_fold_count": plan.inner_fold_count,
    }
    return OwnershipAudit(
        passed=not failed,
        checks=checks,
        failed_checks=failed,
        video_count=plan.video_count,
        outer_fold_count=len(plan.outer_folds),
        inner_fold_count=plan.inner_fold_count,
        digest=canonical_digest(payload),
    )


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float

    def manifest(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class DiscoveryScoreRow:
    target: str
    protocol: str
    outer_fold: int
    recipe: str
    inner_fold: int
    seed: int
    score_scope: str
    fit_videos: tuple[str, ...]
    validation_videos: tuple[str, ...]
    ownership_digest: str
    metrics: tuple[MetricValue, ...]

    @property
    def key(self) -> tuple[str, str, int, str, int, int]:
        return (
            self.target,
            self.protocol,
            self.outer_fold,
            self.recipe,
            self.inner_fold,
            self.seed,
        )

    def metric_map(self) -> dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}

    def manifest(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "protocol": self.protocol,
            "outer_fold": self.outer_fold,
            "recipe": self.recipe,
            "inner_fold": self.inner_fold,
            "seed": self.seed,
            "score_scope": self.score_scope,
            "fit_videos": list(self.fit_videos),
            "validation_videos": list(self.validation_videos),
            "ownership_digest": self.ownership_digest,
            "metrics": [metric.manifest() for metric in self.metrics],
        }


def _metric_values(metrics: Mapping[str, float]) -> tuple[MetricValue, ...]:
    if not metrics:
        raise DiscoveryContractError("Discovery score metrics must not be empty")
    values: list[MetricValue] = []
    for raw_name, raw_value in metrics.items():
        name = str(raw_name)
        lowered = name.lower()
        if not name or any(token in lowered for token in _OUTER_SCORE_TOKENS):
            raise DiscoveryContractError(
                f"Outer-test or held-out metric fields are forbidden: {name!r}"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise DiscoveryContractError(f"Metric {name!r} must be finite")
        values.append(MetricValue(name=name, value=value))
    if len(values) != len({metric.name for metric in values}):
        raise DiscoveryContractError("Metric names must be unique")
    return tuple(sorted(values, key=lambda metric: metric.name))


def make_discovery_score_row(
    plan: NestedDiscoveryPlan,
    *,
    target: str,
    protocol: str,
    outer_fold: int,
    recipe: str,
    inner_fold: int,
    seed: int,
    metrics: Mapping[str, float],
    score_scope: str = INNER_VALIDATION_SCOPE,
    fit_videos: Sequence[str] | None = None,
    validation_videos: Sequence[str] | None = None,
    ownership_digest: str | None = None,
) -> DiscoveryScoreRow:
    """Create a score row sealed to one exact inner train/validation split."""

    inner = plan.inner(int(outer_fold), int(inner_fold))
    fit = tuple(inner.train_videos if fit_videos is None else map(str, fit_videos))
    validation = tuple(
        inner.validation_videos
        if validation_videos is None
        else map(str, validation_videos)
    )
    return DiscoveryScoreRow(
        target=str(target),
        protocol=str(protocol),
        outer_fold=int(outer_fold),
        recipe=str(recipe),
        inner_fold=int(inner_fold),
        seed=int(seed),
        score_scope=str(score_scope),
        fit_videos=fit,
        validation_videos=validation,
        ownership_digest=(inner.digest if ownership_digest is None else str(ownership_digest)),
        metrics=_metric_values(metrics),
    )


_SCORE_MAPPING_FIELDS = frozenset(
    (
        "target",
        "protocol",
        "outer_fold",
        "recipe",
        "inner_fold",
        "seed",
        "score_scope",
        "fit_videos",
        "validation_videos",
        "ownership_digest",
        "metrics",
    )
)


def _coerce_score_row(
    plan: NestedDiscoveryPlan, row: DiscoveryScoreRow | Mapping[str, Any]
) -> DiscoveryScoreRow:
    if isinstance(row, DiscoveryScoreRow):
        return row
    if not isinstance(row, Mapping):
        raise DiscoveryContractError("Discovery scores must be score rows or mappings")
    unknown = set(row) - _SCORE_MAPPING_FIELDS
    if unknown:
        raise DiscoveryContractError(
            f"Unknown score fields are forbidden (outer scores have no input path): {sorted(unknown)}"
        )
    required = _SCORE_MAPPING_FIELDS - {"score_scope"}
    missing = required - set(row)
    if missing:
        raise DiscoveryContractError(f"Discovery score row is missing {sorted(missing)}")
    return make_discovery_score_row(
        plan,
        target=str(row["target"]),
        protocol=str(row["protocol"]),
        outer_fold=int(row["outer_fold"]),
        recipe=str(row["recipe"]),
        inner_fold=int(row["inner_fold"]),
        seed=int(row["seed"]),
        score_scope=str(row.get("score_scope", INNER_VALIDATION_SCOPE)),
        fit_videos=tuple(map(str, row["fit_videos"])),
        validation_videos=tuple(map(str, row["validation_videos"])),
        ownership_digest=str(row["ownership_digest"]),
        metrics=dict(row["metrics"]),
    )


def required_metrics(protocol: str) -> tuple[str, ...]:
    if protocol in _CONTINUOUS_PROTOCOLS:
        return (SPEARMAN, TOP5_LIFT)
    if protocol in _BINARY_PROTOCOLS:
        return (TRAIN_Q90_PR_AUC,)
    raise DiscoveryContractError(f"Unsupported protocol {protocol!r}")


def expected_score_keys(
    plan: NestedDiscoveryPlan,
) -> frozenset[tuple[str, str, int, str, int, int]]:
    return frozenset(
        (
            target,
            protocol,
            outer.outer_fold,
            recipe.name,
            inner.fold,
            seed,
        )
        for target in plan.targets
        for protocol in plan.protocols
        for outer in plan.outer_folds
        for recipe in plan.recipes
        for inner in outer.inner_folds
        for seed in plan.discovery_seeds
    )


@dataclass(frozen=True)
class ScoreMatrixAudit:
    passed: bool
    expected_rows: int
    observed_rows: int
    missing_keys: tuple[tuple[str, str, int, str, int, int], ...]
    unexpected_keys: tuple[tuple[str, str, int, str, int, int], ...]
    duplicate_keys: tuple[tuple[str, str, int, str, int, int], ...]
    ownership_failures: tuple[tuple[str, str, int, str, int, int], ...]
    scope_failures: tuple[tuple[str, str, int, str, int, int], ...]
    metric_failures: tuple[tuple[str, str, int, str, int, int], ...]
    score_digest: str


def audit_score_matrix(
    plan: NestedDiscoveryPlan,
    score_rows: Iterable[DiscoveryScoreRow | Mapping[str, Any]],
) -> tuple[ScoreMatrixAudit, tuple[DiscoveryScoreRow, ...]]:
    """Audit exact matrix completeness plus row-level ownership and metric scope."""

    validate_plan_digest(plan)
    rows = tuple(_coerce_score_row(plan, row) for row in score_rows)
    ordered = tuple(sorted(rows, key=lambda row: row.key))
    expected = expected_score_keys(plan)
    counts: dict[tuple[str, str, int, str, int, int], int] = {}
    for row in ordered:
        counts[row.key] = counts.get(row.key, 0) + 1
    observed = set(counts)
    missing = tuple(sorted(expected - observed))
    unexpected = tuple(sorted(observed - expected))
    duplicates = tuple(sorted(key for key, count in counts.items() if count != 1))
    ownership_failures: list[tuple[str, str, int, str, int, int]] = []
    scope_failures: list[tuple[str, str, int, str, int, int]] = []
    metric_failures: list[tuple[str, str, int, str, int, int]] = []
    for row in ordered:
        if row.key not in expected:
            continue
        inner = plan.inner(row.outer_fold, row.inner_fold)
        outer = plan.outer(row.outer_fold)
        if (
            row.ownership_digest != inner.digest
            or row.fit_videos != inner.train_videos
            or row.validation_videos != inner.validation_videos
            or set(row.fit_videos) & set(row.validation_videos)
            or set(outer.test_videos)
            & (set(row.fit_videos) | set(row.validation_videos))
        ):
            ownership_failures.append(row.key)
        if row.score_scope != INNER_VALIDATION_SCOPE:
            scope_failures.append(row.key)
        metrics = row.metric_map()
        if not set(required_metrics(row.protocol)).issubset(metrics):
            metric_failures.append(row.key)
    score_digest = canonical_digest([row.manifest() for row in ordered])
    passed = not (
        missing
        or unexpected
        or duplicates
        or ownership_failures
        or scope_failures
        or metric_failures
    )
    audit = ScoreMatrixAudit(
        passed=passed,
        expected_rows=len(expected),
        observed_rows=len(rows),
        missing_keys=missing,
        unexpected_keys=unexpected,
        duplicate_keys=duplicates,
        ownership_failures=tuple(sorted(set(ownership_failures))),
        scope_failures=tuple(sorted(set(scope_failures))),
        metric_failures=tuple(sorted(set(metric_failures))),
        score_digest=score_digest,
    )
    return audit, ordered


@dataclass(frozen=True)
class AggregateValue:
    name: str
    value: float


@dataclass(frozen=True)
class RecipeAggregate:
    recipe: str
    recipe_order: int
    complexity_score: int
    score_rows: int
    rank_values: tuple[AggregateValue, ...]
    aggregate_digest: str

    def rank_vector(self) -> tuple[float, ...]:
        return tuple(value.value for value in self.rank_values)

    def manifest(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "recipe_order": self.recipe_order,
            "complexity_score": self.complexity_score,
            "score_rows": self.score_rows,
            "rank_values": [asdict(value) for value in self.rank_values],
            "aggregate_digest": self.aggregate_digest,
        }


@dataclass(frozen=True)
class OuterRecipeSelection:
    target: str
    protocol: str
    outer_fold: int
    selected_recipe: str
    leaderboard: tuple[RecipeAggregate, ...]
    input_score_digest: str
    outer_test_scores_used: bool
    digest: str

    def manifest(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "protocol": self.protocol,
            "outer_fold": self.outer_fold,
            "selected_recipe": self.selected_recipe,
            "leaderboard": [entry.manifest() for entry in self.leaderboard],
            "input_score_digest": self.input_score_digest,
            "outer_test_scores_used": self.outer_test_scores_used,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class DiscoverySelectionArtifact:
    schema_version: str
    plan_digest: str
    score_digest: str
    score_rows: int
    selections: tuple[OuterRecipeSelection, ...]
    ownership_audit_digest: str
    outer_test_scores_used: bool
    artifact_digest: str

    def manifest(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "plan_digest": self.plan_digest,
            "score_digest": self.score_digest,
            "score_rows": self.score_rows,
            "selections": [selection.manifest() for selection in self.selections],
            "ownership_audit_digest": self.ownership_audit_digest,
            "outer_test_scores_used": self.outer_test_scores_used,
        }
        if include_digest:
            payload["artifact_digest"] = self.artifact_digest
        return payload

    def selection(self, target: str, protocol: str, outer_fold: int) -> OuterRecipeSelection:
        key = (str(target), str(protocol), int(outer_fold))
        for selection in self.selections:
            if (selection.target, selection.protocol, selection.outer_fold) == key:
                return selection
        raise DiscoveryContractError(f"No recipe selection for {key}")


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _mean(values: Sequence[float]) -> float:
    return _rounded(math.fsum(values) / len(values))


def _recipe_aggregate(
    *,
    recipe: RecipeSpec,
    protocol: str,
    rows: Sequence[DiscoveryScoreRow],
    inner_folds: Sequence[int],
) -> RecipeAggregate:
    metric_names = required_metrics(protocol)
    metric_vectors = {
        metric: [row.metric_map()[metric] for row in rows] for metric in metric_names
    }
    values: list[AggregateValue] = []
    for metric in metric_names:
        vector = metric_vectors[metric]
        fold_means = [
            _mean(
                [
                    row.metric_map()[metric]
                    for row in rows
                    if row.inner_fold == inner_fold
                ]
            )
            for inner_fold in inner_folds
        ]
        values.extend(
            (
                AggregateValue(f"mean_{metric}", _mean(vector)),
                AggregateValue(
                    f"median_{metric}", _rounded(statistics.median(vector))
                ),
                AggregateValue(f"worst_inner_fold_mean_{metric}", min(fold_means)),
            )
        )
    payload = {
        "recipe": recipe.name,
        "recipe_order": recipe.order,
        "complexity_score": recipe.complexity_score,
        "score_rows": len(rows),
        "rank_values": [asdict(value) for value in values],
        "input_rows": [row.manifest() for row in sorted(rows, key=lambda row: row.key)],
    }
    return RecipeAggregate(
        recipe=recipe.name,
        recipe_order=recipe.order,
        complexity_score=recipe.complexity_score,
        score_rows=len(rows),
        rank_values=tuple(values),
        aggregate_digest=canonical_digest(payload),
    )


def _leaderboard_key(entry: RecipeAggregate) -> tuple[Any, ...]:
    # Higher aggregate metrics win.  An exact aggregate tie goes to the lower
    # prespecified complexity, then the frozen recipe order, then the name.
    return tuple(-value for value in entry.rank_vector()) + (
        entry.complexity_score,
        entry.recipe_order,
        entry.recipe,
    )


def select_nested_recipes(
    plan: NestedDiscoveryPlan,
    score_rows: Iterable[DiscoveryScoreRow | Mapping[str, Any]],
) -> DiscoverySelectionArtifact:
    """Freeze per-target/protocol/outer-fold choices from inner scores only."""

    ownership_audit = audit_nested_ownership(plan)
    if not ownership_audit.passed:
        raise DiscoveryContractError(
            f"Nested ownership audit failed: {list(ownership_audit.failed_checks)}"
        )
    matrix_audit, rows = audit_score_matrix(plan, score_rows)
    if not matrix_audit.passed:
        failures = {
            "missing": len(matrix_audit.missing_keys),
            "unexpected": len(matrix_audit.unexpected_keys),
            "duplicates": len(matrix_audit.duplicate_keys),
            "ownership": len(matrix_audit.ownership_failures),
            "scope": len(matrix_audit.scope_failures),
            "metrics": len(matrix_audit.metric_failures),
        }
        raise DiscoveryContractError(f"Incomplete or invalid discovery score matrix: {failures}")

    selections: list[OuterRecipeSelection] = []
    for target in plan.targets:
        for protocol in plan.protocols:
            for outer in plan.outer_folds:
                subset = tuple(
                    row
                    for row in rows
                    if row.target == target
                    and row.protocol == protocol
                    and row.outer_fold == outer.outer_fold
                )
                input_digest = canonical_digest([row.manifest() for row in subset])
                leaderboard = tuple(
                    sorted(
                        (
                            _recipe_aggregate(
                                recipe=recipe,
                                protocol=protocol,
                                rows=tuple(row for row in subset if row.recipe == recipe.name),
                                inner_folds=tuple(
                                    inner.fold for inner in outer.inner_folds
                                ),
                            )
                            for recipe in plan.recipes
                        ),
                        key=_leaderboard_key,
                    )
                )
                selected = leaderboard[0].recipe
                payload = {
                    "target": target,
                    "protocol": protocol,
                    "outer_fold": outer.outer_fold,
                    "selected_recipe": selected,
                    "leaderboard": [entry.manifest() for entry in leaderboard],
                    "input_score_digest": input_digest,
                    "outer_test_scores_used": False,
                }
                selections.append(
                    OuterRecipeSelection(
                        target=target,
                        protocol=protocol,
                        outer_fold=outer.outer_fold,
                        selected_recipe=selected,
                        leaderboard=leaderboard,
                        input_score_digest=input_digest,
                        outer_test_scores_used=False,
                        digest=canonical_digest(payload),
                    )
                )

    artifact_without_digest = {
        "schema_version": SCHEMA_VERSION,
        "plan_digest": plan.digest,
        "score_digest": matrix_audit.score_digest,
        "score_rows": len(rows),
        "selections": [selection.manifest() for selection in selections],
        "ownership_audit_digest": ownership_audit.digest,
        "outer_test_scores_used": False,
    }
    artifact = DiscoverySelectionArtifact(
        schema_version=SCHEMA_VERSION,
        plan_digest=plan.digest,
        score_digest=matrix_audit.score_digest,
        score_rows=len(rows),
        selections=tuple(selections),
        ownership_audit_digest=ownership_audit.digest,
        outer_test_scores_used=False,
        artifact_digest=canonical_digest(artifact_without_digest),
    )
    verify_selection_artifact(artifact, plan=plan, score_rows=rows)
    return artifact


def verify_selection_artifact(
    artifact: DiscoverySelectionArtifact,
    *,
    plan: NestedDiscoveryPlan | None = None,
    score_rows: Iterable[DiscoveryScoreRow | Mapping[str, Any]] | None = None,
) -> None:
    """Verify immutable artifact, plan, and optional exact score provenance."""

    if artifact.outer_test_scores_used or any(
        selection.outer_test_scores_used for selection in artifact.selections
    ):
        raise DiscoveryContractError("Outer-test scores cannot influence recipe selection")
    if canonical_digest(artifact.manifest(include_digest=False)) != artifact.artifact_digest:
        raise DiscoveryContractError("Discovery selection artifact digest mismatch")
    keys = [
        (selection.target, selection.protocol, selection.outer_fold)
        for selection in artifact.selections
    ]
    if len(keys) != len(set(keys)):
        raise DiscoveryContractError("Discovery selection artifact contains duplicate decisions")
    if plan is not None:
        validate_plan_digest(plan)
        if artifact.plan_digest != plan.digest:
            raise DiscoveryContractError("Selection artifact does not belong to this plan")
        expected = {
            (target, protocol, outer.outer_fold)
            for target in plan.targets
            for protocol in plan.protocols
            for outer in plan.outer_folds
        }
        if set(keys) != expected:
            raise DiscoveryContractError("Selection artifact decision matrix is incomplete")
    if score_rows is not None:
        if plan is None:
            raise DiscoveryContractError("A plan is required to verify score rows")
        audit, rows = audit_score_matrix(plan, score_rows)
        if not audit.passed or audit.score_digest != artifact.score_digest:
            raise DiscoveryContractError("Selection artifact score provenance mismatch")
        if artifact.score_rows != len(rows):
            raise DiscoveryContractError("Selection artifact score-row count mismatch")


__all__ = [
    "DiscoveryContractError",
    "DiscoveryScoreRow",
    "DiscoverySelectionArtifact",
    "GroupedOwnership",
    "INNER_FOLD_COUNT",
    "INNER_VALIDATION_SCOPE",
    "MetricValue",
    "NestedDiscoveryPlan",
    "OUTER_FOLD_COUNT",
    "OuterDiscoveryFold",
    "OuterRecipeSelection",
    "PRIVILEGED_BINARY",
    "PRIVILEGED_CONTINUOUS",
    "RecipeSpec",
    "RecipeAggregate",
    "ScoreMatrixAudit",
    "SUPPORTED_PROTOCOLS",
    "TRAIN_Q90_PR_AUC",
    "TOP5_LIFT",
    "SPEARMAN",
    "ZERO_LABEL_BINARY",
    "ZERO_LABEL_CONTINUOUS",
    "audit_nested_ownership",
    "audit_score_matrix",
    "build_nested_discovery_plan",
    "canonical_digest",
    "canonical_recipe_specs",
    "expected_score_keys",
    "make_discovery_score_row",
    "normalize_recipes",
    "required_metrics",
    "select_nested_recipes",
    "verify_selection_artifact",
]
