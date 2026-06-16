"""Convert the released subcortical safetensors file for TribeModel loading."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else model_dir / "best.ckpt"
    build_args = json.loads((model_dir / "build_args.json").read_text(encoding="utf-8"))
    state = load_file(str(model_dir / "best.safetensors"), device="cpu")
    # TribeModel.from_pretrained intentionally runs inference in average-subject
    # mode. Average the ten measured Lahner participants and exclude the
    # eleventh subject-dropout head used during training.
    state["predictor.weights"] = state["predictor.weights"][:10].mean(dim=0, keepdim=True)
    state["predictor.bias"] = state["predictor.bias"][:10].mean(dim=0, keepdim=True)
    checkpoint = {
        "model_build_args": build_args,
        "state_dict": {f"model.{key}": value for key, value in state.items()},
        "source": {
            "format": "loganf26/tribev2-subcortical safetensors",
            "conversion": "ten measured participant heads averaged for TribeModel average-subject inference",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print({"output": str(output), "tensor_count": len(state), "bytes": output.stat().st_size})


if __name__ == "__main__":
    main()
