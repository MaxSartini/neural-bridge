"""Scout MLX V-JEPA2 candidates for the cortical TRIBE video contract.

This is a discovery/probe utility. It does not modify benchmark caches, model
directories, or video-window caches.

The cortical TRIBE contract currently uses facebook/vjepa2-vitg-fpc64-256:
64 frames, 256px, ViT-G, hidden size 1408, 40 blocks. Published MLX ViT-L
ports are not drop-in replacements for that contract.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HF_QUERIES = (
    "vjepa2 mlx",
    "vjepa2 vitg mlx",
    "vjepa2-vitg-fpc64-256 mlx",
    "V-JEPA2 vitg fpc64 MLX safetensors",
)

GITHUB_QUERIES = (
    "vjepa2 mlx",
    "vjepa2 mlx vitg",
    "vjepa2-vitg-fpc64-256 mlx",
    "V-JEPA2 vitg MLX",
)

EXPECTED_CORTICAL = {
    "base_model": "facebook/vjepa2-vitg-fpc64-256",
    "frames": 64,
    "resolution": 256,
    "hidden_size": 1408,
    "depth": 40,
}

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("NEURAL_BRIDGE_EXTERNAL_ROOT", str(ROOT / "external_assets"))).expanduser()
DEFAULT_CORTICAL_HF_MODEL_DIR = str(
    EXTERNAL_ROOT / "models" / "cortical-upstream" / "facebook-vjepa2-vitg-fpc64-256"
)


@dataclass
class Candidate:
    source: str
    identifier: str
    url: str
    classification: str
    reason: str
    tags: list[str]


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "neural_bridge-vjepa2-mlx-probe"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def classify_text(identifier: str, tags: list[str], description: str = "") -> tuple[str, str]:
    haystack = " ".join([identifier, description, *tags]).lower()
    if "vjepa2-vitg-fpc64-256" in haystack and "mlx" in haystack:
        return "exact_candidate", "mentions the exact cortical ViT-G FPC64-256 model and MLX"
    if "vitg" in haystack or "vit-g" in haystack or "giant" in haystack:
        if "action-conditioned" in haystack or "ac-" in haystack or "robotics" in haystack:
            return "near_miss_vitg_ac", "ViT-G MLX artifact, but action-conditioned/robotics variant"
        return "near_candidate_vitg", "ViT-G/giant MLX artifact; feature-contract parity still required"
    if "vitl" in haystack or "vit-l" in haystack or "large" in haystack:
        return "not_cortical_drop_in", "ViT-L/large is hidden 1024, not cortical TRIBE ViT-G hidden 1408"
    return "unknown", "requires manual inspection"


def search_hugging_face() -> list[Candidate]:
    candidates: dict[str, Candidate] = {}
    for query in HF_QUERIES:
        url = "https://huggingface.co/api/models?search=" + urllib.parse.quote(query) + "&limit=50"
        for item in fetch_json(url):
            model_id = item.get("modelId", "")
            tags = [str(tag) for tag in item.get("tags", [])]
            if "vjepa" not in model_id.lower() and not any("vjepa" in tag.lower() for tag in tags):
                continue
            classification, reason = classify_text(model_id, tags)
            candidates[model_id] = Candidate(
                source="huggingface",
                identifier=model_id,
                url=f"https://huggingface.co/{model_id}",
                classification=classification,
                reason=reason,
                tags=tags[:16],
            )
    return sorted(candidates.values(), key=lambda item: (item.classification, item.identifier))


def search_github() -> list[Candidate]:
    candidates: dict[str, Candidate] = {}
    for query in GITHUB_QUERIES:
        url = (
            "https://api.github.com/search/repositories?q="
            + urllib.parse.quote(query)
            + "&sort=updated&order=desc&per_page=30"
        )
        data = fetch_json(url)
        for item in data.get("items", []):
            full_name = item.get("full_name", "")
            description = item.get("description") or ""
            if "vjepa" not in (full_name + " " + description).lower():
                continue
            classification, reason = classify_text(full_name, [], description)
            candidates[full_name] = Candidate(
                source="github",
                identifier=full_name,
                url=item.get("html_url", ""),
                classification=classification,
                reason=reason,
                tags=[description] if description else [],
            )
    return sorted(candidates.values(), key=lambda item: (item.classification, item.identifier))


def run_git_clone(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def find_vit_giant_rope(port_path: Path) -> dict[str, Any]:
    """Inspect a cloned MLX port for a V-JEPA 2.0 vit_giant_rope constructor."""
    matches = list(port_path.rglob("vision_transformer.py"))
    report: dict[str, Any] = {
        "port_path": str(port_path),
        "vision_transformer_files": [str(path.relative_to(port_path)) for path in matches],
        "has_vit_giant_rope": False,
        "has_v2_vit_giant_rope": False,
        "vit_giant_rope_matches": [],
        "expected_contract": EXPECTED_CORTICAL,
    }
    for path in matches:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "def vit_giant_rope" not in text:
            continue
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "vit_giant_rope":
                source = ast.get_source_segment(text, node) or ""
                relative = str(path.relative_to(port_path))
                is_v2 = "/v2/" in f"/{relative}"
                report["has_vit_giant_rope"] = True
                report["has_v2_vit_giant_rope"] = bool(report["has_v2_vit_giant_rope"] or is_v2)
                report["vit_giant_rope_matches"].append(
                    {
                        "file": relative,
                        "is_vjepa2_0_path": is_v2,
                        "mentions_hidden_1408": "embed_dim=1408" in source.replace(" ", ""),
                        "mentions_depth_40": "depth=40" in source.replace(" ", ""),
                        "function_source_excerpt": "\n".join(source.splitlines()[:24]),
                    }
                )
    return report


def inspect_xocialize_port(port_path: Path) -> dict[str, Any]:
    config_path = port_path / "vjepa2_mlx" / "config.py"
    convert_path = port_path / "scripts" / "convert_to_mlx.py"
    model_path = port_path / "vjepa2_mlx" / "models" / "modeling_vjepa2.py"
    report: dict[str, Any] = {
        "port_path": str(port_path),
        "has_config": config_path.exists(),
        "has_convert_script": convert_path.exists(),
        "has_modeling": model_path.exists(),
        "supports_hf_split_qkv_keys": False,
        "has_vitg_conversion_example": False,
        "expected_contract": EXPECTED_CORTICAL,
    }
    if model_path.exists():
        model_text = model_path.read_text(encoding="utf-8", errors="replace")
        report["supports_hf_split_qkv_keys"] = all(
            token in model_text
            for token in (
                "self.query = nn.Linear",
                "self.key = nn.Linear",
                "self.value = nn.Linear",
            )
        )
    if convert_path.exists():
        convert_text = convert_path.read_text(encoding="utf-8", errors="replace")
        compact = convert_text.replace(" ", "")
        report["has_vitg_conversion_example"] = (
            "hidden_size=1408" in compact
            and "num_hidden_layers=40" in compact
            and "num_attention_heads=22" in compact
            and "V-JEPA2-AC-vitg" in convert_text
        )
    weights_path = port_path / "vjepa2_mlx" / "utils" / "weights.py"
    if weights_path.exists():
        weights_text = weights_path.read_text(encoding="utf-8", errors="replace")
        compact = weights_text.replace(" ", "")
        report["has_vitg_weights_helper"] = (
            "hidden_size=1408" in compact
            and "num_hidden_layers=40" in compact
            and "num_attention_heads=22" in compact
        )
    return report


def inspect_local_hf_checkpoint(model_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "expected_contract": EXPECTED_CORTICAL,
    }
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        report["error"] = "config.json or model.safetensors is missing"
        return report

    config = json.loads(config_path.read_text(encoding="utf-8"))
    report["config_subset"] = {
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "frames_per_clip": config.get("frames_per_clip"),
        "image_size": config.get("image_size"),
        "tubelet_size": config.get("tubelet_size"),
        "mlp_ratio": config.get("mlp_ratio"),
    }
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - dependency missing path
        report["error"] = f"safetensors unavailable: {type(exc).__name__}: {exc}"
        return report

    required_shapes = {
        "encoder.embeddings.patch_embeddings.proj.weight": (1408, 3, 2, 16, 16),
        "encoder.layer.0.attention.query.weight": (1408, 1408),
        "encoder.layer.0.attention.key.weight": (1408, 1408),
        "encoder.layer.0.attention.value.weight": (1408, 1408),
        "encoder.layer.39.mlp.fc2.weight": (1408, 6144),
        "encoder.layernorm.weight": (1408,),
    }
    with safe_open(str(weights_path), framework="np") as bundle:
        keys = set(bundle.keys())
        observed = {}
        missing = []
        mismatched = []
        for key, expected in required_shapes.items():
            if key not in keys:
                missing.append(key)
                continue
            shape = tuple(bundle.get_tensor(key).shape)
            observed[key] = shape
            if shape != expected:
                mismatched.append({"key": key, "expected": expected, "observed": shape})
        report.update(
            {
                "num_weight_keys": len(keys),
                "required_shapes": observed,
                "missing_required_keys": missing,
                "mismatched_required_shapes": mismatched,
                "hf_split_qkv_key_layout": all(
                    key in keys
                    for key in (
                        "encoder.layer.0.attention.query.weight",
                        "encoder.layer.0.attention.key.weight",
                        "encoder.layer.0.attention.value.weight",
                    )
                ),
            }
        )
    report["matches_expected_cortical_config"] = (
        config.get("hidden_size") == EXPECTED_CORTICAL["hidden_size"]
        and config.get("num_hidden_layers") == EXPECTED_CORTICAL["depth"]
        and config.get("frames_per_clip") == EXPECTED_CORTICAL["frames"]
        and config.get("image_size") == EXPECTED_CORTICAL["resolution"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--clone-dir",
        default="/tmp/neural_bridge-vjepa2-mlx-scout",
        help="Scratch directory for optional GitHub inspection.",
    )
    parser.add_argument(
        "--inspect-dgrauet",
        action="store_true",
        help="Clone and inspect dgrauet/vjepa2-mlx for a ViT-G MLX constructor.",
    )
    parser.add_argument(
        "--inspect-xocialize",
        action="store_true",
        help="Clone and inspect xocialize/vjepa2-mlx for HF-style ViT-G support.",
    )
    parser.add_argument(
        "--local-hf-model-dir",
        default=DEFAULT_CORTICAL_HF_MODEL_DIR,
        help="Local HF cortical V-JEPA2 model directory to inspect without loading all weights.",
    )
    parser.add_argument(
        "--inspect-local-hf",
        action="store_true",
        help="Inspect local cortical HF config/key shapes for MLX conversion feasibility.",
    )
    args = parser.parse_args()

    hf_candidates = search_hugging_face()
    gh_candidates = search_github()
    inspections = []

    if args.inspect_dgrauet:
        target = Path(args.clone_dir).expanduser().resolve() / "dgrauet-vjepa2-mlx"
        run_git_clone("https://github.com/dgrauet/vjepa2-mlx.git", target)
        inspections.append(find_vit_giant_rope(target))
    if args.inspect_xocialize:
        target = Path(args.clone_dir).expanduser().resolve() / "xocialize-vjepa2-mlx"
        run_git_clone("https://github.com/xocialize/vjepa2-mlx.git", target)
        inspections.append(inspect_xocialize_port(target))
    if args.inspect_local_hf:
        inspections.append(inspect_local_hf_checkpoint(Path(args.local_hf_model_dir).expanduser()))

    exact = [
        candidate
        for candidate in [*hf_candidates, *gh_candidates]
        if candidate.classification == "exact_candidate"
    ]
    report = {
        "expected_cortical_contract": EXPECTED_CORTICAL,
        "summary": {
            "exact_mlx_cortical_candidates": len(exact),
            "huggingface_candidates": len(hf_candidates),
            "github_candidates": len(gh_candidates),
        },
        "huggingface": [asdict(candidate) for candidate in hf_candidates],
        "github": [asdict(candidate) for candidate in gh_candidates],
        "inspections": inspections,
        "decision": (
            "No published exact MLX drop-in was found for facebook/vjepa2-vitg-fpc64-256. "
            "Community code appears to include enough ViT-G architecture/key-layout support "
            "to build one from the exact local HF weights; next step is converting/loading "
            "the exact HF ViT-G weights and parity-testing TRIBE layer/token aggregation."
        ),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(report["summary"], indent=2))
        print(report["decision"])
        for candidate in hf_candidates + gh_candidates:
            print(f"{candidate.source}\t{candidate.classification}\t{candidate.identifier}\t{candidate.url}")
        if inspections:
            print(json.dumps({"inspections": inspections}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
