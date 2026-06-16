"""Resume selected pretrained behavior-component downloads to external storage."""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ROOT = Path(
    os.environ.get("NEURAL_BRIDGE_EXTERNAL_ASSET_ROOT")
    or os.environ.get("MIROFISH_EXTERNAL_ASSET_ROOT", "/Volumes/onn. Drive/Neural Bridge")
)

COMPONENTS = {
    "moment_small": {
        "repo_id": "AutonLab/MOMENT-1-small",
        "local_dir": "models/MOMENT-1-small",
        "allow_patterns": ["config.json", "model.safetensors", "README.md"],
    },
    "moment_large": {
        "repo_id": "AutonLab/MOMENT-1-large",
        "local_dir": "models/MOMENT-1-large",
        "allow_patterns": ["config.json", "model.safetensors", "README.md"],
    },
    "brainlm_111m": {
        "repo_id": "vandijklab/brainlm",
        "local_dir": "models/BrainLM-111M",
        "allow_patterns": ["README.md", "vitmae_111M/config.json", "vitmae_111M/pytorch_model.bin"],
    },
    "brain_dit": {
        "repo_id": "BrainDiT/BrainDiT",
        "local_dir": "models/Brain-DiT",
        "allow_patterns": ["BrainDit.pt", ".gitattributes"],
    },
    "brain_jepa": {
        "repo_id": "eugenehp/brainjepa",
        "local_dir": "models/Brain-JEPA",
    },
    "neurostorm": {
        "repo_id": "zxcvb20001/NeuroSTORM",
        "local_dir": "models/NeuroSTORM-pretrain",
        "allow_patterns": ["pretraining/pt_neurostorm_mae_5ds.ckpt"],
    },
    "tribe_subcortical": {
        "repo_id": "loganf26/tribev2-subcortical",
        "local_dir": "models/TRIBE-v2-subcortical",
        "allow_patterns": ["best.safetensors", "build_args.json", "config.yaml", "eval.json", "README.md"],
    },
    "minitaur_mlx": {
        "repo_id": "HillPhelmuth/Llama-3.1-Minitaur-8B-mlx-4Bit",
        "local_dir": "models/Llama-3.1-Minitaur-8B-mlx-4Bit",
    },
    "psych_101": {
        "repo_id": "marcelbinz/Psych-101",
        "repo_type": "dataset",
        "local_dir": "datasets/Psych-101",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("components", nargs="+", choices=["all", *COMPONENTS.keys()])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    selected = list(COMPONENTS) if "all" in args.components else args.components

    for name in selected:
        config = dict(COMPONENTS[name])
        config["local_dir"] = str(root / config["local_dir"])
        print(f"Downloading/resuming {name} -> {config['local_dir']}", flush=True)
        snapshot_download(**config)


if __name__ == "__main__":
    main()
