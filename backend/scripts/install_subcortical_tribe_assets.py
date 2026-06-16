"""Install and verify exact assets required by the TRIBE subcortical head."""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


CHECKPOINT_SHA256 = "14fc14a1d1d5939d8a0503b828f284919c71ef603848489043efc656163d45c8"
UPSTREAM_MODELS = {
    "Qwen/Qwen3-0.6B": ("Qwen-Qwen3-0.6B", ["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"]),
    "facebook/w2v-bert-2.0": ("facebook-w2v-bert-2.0", ["*.json", "*.safetensors", "README.md"]),
    "facebook/vjepa2-vitl-fpc64-256": ("facebook-vjepa2-vitl-fpc64-256", ["*.json", "*.safetensors", "README.md"]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        default="/Volumes/onn. Drive/Neural Bridge/models",
        help="Large-model storage root",
    )
    parser.add_argument("--skip-upstream", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    local_model = project / "models" / "tribe" / "loganf26-tribev2-subcortical"
    checkpoint = local_model / "best.safetensors"
    if not checkpoint.exists() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("Subcortical best.safetensors is missing or has the wrong SHA256")

    external = Path(args.external_root).expanduser().resolve()
    converted = external / "TRIBE-v2-subcortical" / "best.ckpt"
    if not converted.exists():
        subprocess.run(
            [
                sys.executable,
                str(project / "backend" / "scripts" / "convert_subcortical_safetensors_to_tribe_checkpoint.py"),
                str(local_model),
                "--output",
                str(converted),
            ],
            check=True,
        )
    link = local_model / "best.ckpt"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(converted)

    if not args.skip_upstream:
        upstream = external / "subcortical-upstream"
        for repo_id, (directory, allow_patterns) in UPSTREAM_MODELS.items():
            target = upstream / directory
            print(f"Installing/resuming {repo_id} -> {target}", flush=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                allow_patterns=allow_patterns,
            )

    env_lines = [
        f"TRIBE_SUBCORTICAL_LOCAL_DIR={local_model}",
        f"TRIBE_SUBCORTICAL_TEXT_ENCODER_LOCAL_DIR={external / 'subcortical-upstream' / 'Qwen-Qwen3-0.6B'}",
        f"TRIBE_SUBCORTICAL_AUDIO_ENCODER_LOCAL_DIR={external / 'subcortical-upstream' / 'facebook-w2v-bert-2.0'}",
        f"TRIBE_SUBCORTICAL_VIDEO_ENCODER_LOCAL_DIR={external / 'subcortical-upstream' / 'facebook-vjepa2-vitl-fpc64-256'}",
    ]
    output = project / "models" / "tribe" / "subcortical.env"
    output.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print({"checkpoint_sha256": CHECKPOINT_SHA256, "environment_file": str(output)})


if __name__ == "__main__":
    main()
