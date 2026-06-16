"""Run one bounded-memory V-JEPA2 ViT-G window on Apple MPS."""

import os
import time
from pathlib import Path

import numpy as np
import torch
from moviepy import VideoFileClip
from transformers import AutoModel, AutoVideoProcessor


MODEL = Path(
    "/Volumes/onn. Drive/Neural Bridge/models/cortical-upstream/facebook-vjepa2-vitg-fpc64-256"
)
VIDEO = Path("/Volumes/onn. Drive/Neural Bridge/datasets/openlav_videos/VID_1001.webm")


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    fraction = float(os.environ.get("TRIBE_MPS_MEMORY_FRACTION", "0.45"))
    torch.mps.set_per_process_memory_fraction(fraction)
    started = time.time()
    model = AutoModel.from_pretrained(
        MODEL,
        local_files_only=True,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    ).eval().to("mps")
    processor = AutoVideoProcessor.from_pretrained(MODEL, local_files_only=True)
    video = VideoFileClip(str(VIDEO))
    frame_count = int(os.environ.get("TRIBE_VIDEO_NUM_FRAMES", "32"))
    times = np.linspace(0, min(4.0, video.duration), frame_count, endpoint=False)
    frames = [video.get_frame(float(t)).astype(np.uint8) for t in times]
    video.close()
    inputs = processor(videos=frames, return_tensors="pt").to("mps")
    with torch.inference_mode():
        output = model(**inputs, skip_predictor=True)
        selected = torch.stack(
            [output.hidden_states[index].mean(dim=1) for index in (20, 30, 40)]
        ).float().cpu().numpy()
    print({
        "vjepa_mps_window_ok": True,
        "seconds": round(time.time() - started, 2),
        "selected_shape": list(selected.shape),
        "finite": bool(np.isfinite(selected).all()),
        "memory_fraction": fraction,
        "frame_count": frame_count,
        "driver_allocated_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2),
    })


if __name__ == "__main__":
    main()
