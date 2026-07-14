"""Audit the repo for stale guidance, local paths, and accidentally tracked bulk."""

from __future__ import annotations

import json
import re
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    "ROADMAP.md",
    "REQUIREMENTS.md",
    ".env.example",
    "docs/README.md",
    "docs/current_project_state.md",
    "docs/current_claim_status.json",
    "docs/neural_bridge_phase5_5_evidence_ladder.md",
    "docs/how_neural_bridge_was_discovered.md",
    "docs/executable_validation_index.md",
    "docs/executable_validation_manifest.csv",
    "docs/executable_validation_manifest.json",
    "docs/test_suite_result_20260714.json",
    "docs/again_dense_h100_cache.md",
    "docs/veatic_v2_evidence_summary.md",
    "docs/veatic_v2_evidence_freeze.md",
    "docs/veatic_raw_representation_audit.md",
    "docs/external_assets_manifest.md",
    "reports/README.md",
    "benchmarks/veatic/veatic_v2_evidence_manifest.json",
    "outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md",
    "outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_summary.json",
    "outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_verification.json",
]

FORBIDDEN_PATTERNS = {
    "old project name": re.compile(r"\b(?:MiroFish|mirofish|NeuroFish|neurofish)\b"),
    "retired OpenLAV references": re.compile(r"\b(?:OPENLAV|openlav)\b"),
    "retired Brain-JEPA references": re.compile(r"\b(?:Brain-JEPA|Brain JEPA)\b"),
    "machine-specific user path": re.compile(r"/Users/maxsartini\b"),
    "machine-specific SSD path": re.compile(r"/Volumes/onn\. Drive\b"),
    "old checkout path": re.compile(r"MiroFish-Offline"),
}

CONTROLLED_EVIDENCE_PATTERNS = {
    "retained disabled non-cortical metadata": re.compile(r"\bsubcortical\b", re.IGNORECASE),
}

CONTROLLED_EVIDENCE_FILES = re.compile(
    r"^benchmarks/veatic/veatic_neuro_benchmark_124video_.*(?:\.json|\.summary\.md)$"
)

FORBIDDEN_TRACKED_SUFFIXES = {
    ".ckpt",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
    ".mp3",
    ".mp4",
    ".wav",
}

MAX_TRACKED_FILE_BYTES = 15 * 1024 * 1024
REVIEW_DOSSIER_PREFIX = "evidence/current_phase_5_5_review/"
EVIDENCE_TRACKED_PREFIXES = ("evidence/phase_", REVIEW_DOSSIER_PREFIX)
MAX_EVIDENCE_BUNDLE_FILE_BYTES = 70 * 1024 * 1024
EVIDENCE_BUNDLE_ALLOWED_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".parquet", ".py", ".txt"}
EVIDENCE_BUNDLE_ALLOWED_DOUBLE_SUFFIXES = {(".csv", ".gz")}


def git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        check=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def tracked_files() -> list[Path]:
    return [ROOT / line for line in git_lines("ls-files")]


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
        return True
    except UnicodeDecodeError:
        return False


def audit_required_files(errors: list[str]) -> None:
    for item in REQUIRED_FILES:
        if not (ROOT / item).is_file():
            errors.append(f"missing required orientation file: {item}")


def audit_tracked_bulk(files: list[Path], errors: list[str]) -> None:
    for path in files:
        rel = relative(path)
        if not path.exists():
            continue
        if rel.startswith(EVIDENCE_TRACKED_PREFIXES):
            size = path.stat().st_size
            if (
                path.name != ".codexignore"
                and path.suffix not in EVIDENCE_BUNDLE_ALLOWED_SUFFIXES
                and tuple(path.suffixes[-2:]) not in EVIDENCE_BUNDLE_ALLOWED_DOUBLE_SUFFIXES
            ):
                errors.append(f"tracked evidence-bundle file has unexpected suffix: {rel}")
            if size > MAX_EVIDENCE_BUNDLE_FILE_BYTES:
                errors.append(
                    f"tracked evidence-bundle file exceeds {MAX_EVIDENCE_BUNDLE_FILE_BYTES} bytes: {rel} ({size})"
                )
            continue
        if path.suffix in FORBIDDEN_TRACKED_SUFFIXES:
            errors.append(f"tracked heavyweight artifact: {rel}")
        size = path.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            errors.append(f"tracked file exceeds {MAX_TRACKED_FILE_BYTES} bytes: {rel} ({size})")


def audit_stale_terms(files: list[Path], errors: list[str], warnings: list[str]) -> None:
    for path in files:
        rel = relative(path)
        if not path.exists():
            continue
        if rel == "backend/scripts/audit_repo_readiness.py":
            continue
        if not is_text(path):
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                if label.startswith("machine-specific") and rel.startswith("evidence/"):
                    warnings.append(f"historical evidence provenance: {rel}:{line}")
                else:
                    errors.append(f"{label}: {rel}:{line}")
        for label, pattern in CONTROLLED_EVIDENCE_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                if CONTROLLED_EVIDENCE_FILES.match(rel) or rel.startswith("evidence/"):
                    warnings.append(f"{label}: {rel}:{line}")
                else:
                    errors.append(f"retired active-reference term: {rel}:{line}")


def audit_orientation_content(errors: list[str]) -> None:
    checks = {
        "README.md": ["Best Results First", "future_arousal_max_delta_rows_4_10_train_q90", "short_temporal_conv_residual", "Raw predicted cortical/fMRI features alone fail badly", "grouped continuous future-movement ranking/lift", "how_neural_bridge_was_discovered.md", "No 504 run has been promoted", "Executable Validation"],
        "AGENTS.md": ["future_arousal_max_delta_rows_4_10_train_q90", "short_temporal_conv_residual", "updated grouped compatibility", "Raw predicted cortical/fMRI features alone fail badly", "grouped continuous future-movement ranking/lift", "Test And Script Validation"],
        "REQUIREMENTS.md": ["video-dominant", "TRIBE_TEXT_ENCODER_LOCAL_DIR", "Llama-3.2-3B"],
        "ROADMAP.md": ["Best Current Result", "Phase 5.5 evidence ladder", "504", "evidence/"],
        "docs/current_project_state.md": ["future_arousal_max_delta_rows_4_10_train_q90", "short_temporal_conv_residual", "raw_cortical_only", "Exact continuous-value forecasting and blocked continuous generalization remain open", "grouped continuous future-movement ranking/lift", "Executable Validation"],
        "docs/neural_bridge_phase5_5_evidence_ladder.md": ["Best AGAIN Results", "Raw Predicted Cortical/FMRI Features Alone Fail Badly", "Continuous Future-Movement Ranking Result", "Grouped Compatibility Block", "Current Evidence Boundaries", "Executable Validation"],
        "docs/how_neural_bridge_was_discovered.md": ["three bounded victories", "continuous_ranking_lift_pass", "Washout-Gap Targets Changed The Scientific Question", "+0.0055230967"],
        "reports/README.md": ["Current Claim-Bearing Reports", "Superseded And Historical Reports", "grouped continuous-ranking/lift pass"],
        "reports/again_dense_2hz_phase5_temporal_residual_grouped_compat_20260630_033520.md": ["SUPERSEDED VERDICT", "UPDATED_VERDICT"],
        "reports/again_dense_2hz_phase5_continuous_residual_blocked_summary_20260630_000219.md": ["HISTORICAL, PROTOCOL-SPECIFIC FAILURE", "grouped continuous future-movement ranking/lift pass"],
        "evidence/current_phase_5_5_review/CLAIM_LEDGER.md": ["proven_grouped_continuous_ranking_lift", "open_exact_and_blocked_continuous"],
        "docs/executable_validation_index.md": ["Best Validation First", "2026-07-14", "Current Claim-Bearing Runners", "Deterministic Tests"],
        "docs/veatic_v2_evidence_freeze.md": ["evidence:verify", "does not re-encode videos", "Post-freeze Tensor Contract"],
        "docs/veatic_raw_representation_audit.md": ["pca_sequence_128_causal_past_2s_mean", "roi_parcel_features", "topk_vertices_512"],
        "outputs/veatic_124_raw_representation_tensor_export_v1/tensor_export_report.md": ["84 tensor contracts", "420", "No videos were re-encoded"],
    }
    for rel, needles in checks.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"{rel} does not mention required context: {needle}")


def audit_package_scripts(errors: list[str]) -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    scripts = package.get("scripts", {})
    for name in ["audit:repo", "evidence:freeze", "evidence:verify", "verify", "test"]:
        if name not in scripts:
            errors.append(f"package.json missing script: {name}")
    if "audit:repo" not in scripts.get("verify", ""):
        errors.append("package.json verify script must run audit:repo")


def audit_remote(errors: list[str]) -> None:
    remotes = "\n".join(git_lines("remote", "-v"))
    if "MaxSartini/neural-bridge.git" not in remotes:
        errors.append("origin remote is not MaxSartini/neural-bridge.git")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print controlled retained-evidence matches as individual lines.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    files = tracked_files()
    audit_required_files(errors)
    audit_tracked_bulk(files, errors)
    audit_stale_terms(files, errors, warnings)
    audit_orientation_content(errors)
    audit_package_scripts(errors)
    audit_remote(errors)

    if args.verbose:
        for warning in warnings:
            print(f"controlled_evidence: {warning}")

    if errors:
        for error in errors:
            print(f"error: {error}")
        print(f"repo_readiness\tfail\t{len(errors)}")
        return 1

    print(f"repo_readiness\tpass\tcontrolled_evidence_items={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
