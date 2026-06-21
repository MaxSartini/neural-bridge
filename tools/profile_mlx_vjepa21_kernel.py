#!/usr/bin/env python3
"""Profile one fixed-shape V-JEPA 2.1 MLX kernel forward."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.app.config import Config  # noqa: E402
from backend.app.services.mlx_vjepa2_cortical import (  # noqa: E402
    _decode_video_grid_ffmpeg,
    _sample_decoded_grid,
)
from backend.app.services.mlx_vjepa21_cortical import (  # noqa: E402
    MlxVjepa21FeatureModel,
    MlxVjepa21Video,
    _preprocess_video_batch,
)
from probe_veatic_vjepa21_real_video_smoke import (  # noqa: E402
    load_manifest_rows,
    resolve_external_path,
    select_short_video,
    write_csv,
    write_json,
)


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def timed(label: str, rows: list[dict[str, Any]], func: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    result = func()
    if isinstance(result, mx.array):
        mx.eval(result)
    elif isinstance(result, (list, tuple)):
        mx.eval(*[item for item in result if isinstance(item, mx.array)])
    elapsed = time.perf_counter() - started
    rows.append({"stage": label, "seconds": elapsed})
    return result


def build_real_window(
    *,
    video_path: Path,
    duration_seconds: float,
    image_size: int,
    num_frames: int,
    clip_duration: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    decode_fps = num_frames / clip_duration
    started = time.perf_counter()
    grid = _decode_video_grid_ffmpeg(video_path, fps=decode_fps, image_size=image_size)
    decode_seconds = time.perf_counter() - started
    timepoint = min(max(clip_duration, duration_seconds / 2.0), duration_seconds)
    subtimes = [index / num_frames * clip_duration for index in reversed(range(num_frames))]
    frame_times = [max(0.0, timepoint - delta) for delta in subtimes]
    started = time.perf_counter()
    window = _sample_decoded_grid(grid, fps=decode_fps, times=frame_times)
    sample_seconds = time.perf_counter() - started
    return window, {
        "decode_fps": decode_fps,
        "decoded_frame_count": int(grid.shape[0]),
        "decoded_grid_shape": list(grid.shape),
        "timepoint_seconds": timepoint,
        "ffmpeg_decode_seconds": decode_seconds,
        "window_sample_seconds": sample_seconds,
    }


def profile_encoder_groups(
    model: MlxVjepa21FeatureModel,
    video: mx.array,
    selected: list[int],
    *,
    group_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected_set = set(selected)
    captured: dict[int, mx.array] = {}
    encoder = model.encoder
    batch, _channels, frames, height, width = video.shape
    temporal = frames // encoder.config.tubelet_size
    h_patches = height // encoder.config.patch_size
    w_patches = width // encoder.config.patch_size

    hidden = timed("patch_embed", rows, lambda: encoder.patch_embed(video))
    if encoder.config.modality_embedding:
        hidden = timed(
            "modality_embedding_add",
            rows,
            lambda: hidden + mx.broadcast_to(encoder.video_mod_embed, (batch, 1, encoder.config.hidden_size)),
        )
    if 0 in selected_set:
        captured[0] = timed("capture_layer_0_mean", rows, lambda: mx.mean(hidden, axis=1))

    group_started = time.perf_counter()
    group_start = 1
    capture_seconds = 0.0
    for index, block in enumerate(encoder.blocks, start=1):
        block_started = time.perf_counter()
        hidden = block(hidden, temporal=temporal, h_patches=h_patches, w_patches=w_patches)
        mx.eval(hidden)
        rows.append({"stage": f"block_{index:02d}", "seconds": time.perf_counter() - block_started})
        if index in selected_set:
            capture_started = time.perf_counter()
            state = encoder.norms_block[-1](hidden) if index == len(encoder.blocks) else hidden
            captured[index] = mx.mean(state, axis=1)
            mx.eval(captured[index])
            capture_seconds += time.perf_counter() - capture_started
        if index % group_size == 0 or index == len(encoder.blocks):
            rows.append(
                {
                    "stage": f"block_group_{group_start:02d}_{index:02d}",
                    "seconds": time.perf_counter() - group_started,
                }
            )
            group_started = time.perf_counter()
            group_start = index + 1

    rows.append({"stage": "selected_capture_total", "seconds": capture_seconds})
    missing = [index for index in selected if index not in captured]
    if missing:
        raise ValueError(f"selected hidden-state indices not captured: {missing}")
    output = timed("stack_selected", rows, lambda: mx.stack([captured[index] for index in selected], axis=1)[:, :, None, :])
    return np.asarray(output, dtype=np.float32), rows


def measure_forward(
    label: str,
    func: Callable[[mx.array], mx.array],
    video: mx.array,
    *,
    repeats: int,
) -> dict[str, Any]:
    times: list[float] = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = func(video)
        mx.eval(output)
        times.append(time.perf_counter() - started)
    arr = np.asarray(output, dtype=np.float32)
    return {
        "mode": label,
        "repeats": repeats,
        "median_seconds": median(times),
        "min_seconds": float(min(times)) if times else None,
        "max_seconds": float(max(times)) if times else None,
        "times_json": json.dumps(times),
        "output_shape": json.dumps(list(arr.shape)),
        "output_nonfinite_count": int((~np.isfinite(arr)).sum()),
        "output_mean": float(arr.mean()),
        "output_std": float(arr.std()),
        "output_min": float(arr.min()),
        "output_max": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("benchmarks/veatic/veatic_manifest_124_complete_20260616.report.json"))
    parser.add_argument("--video-id", default="52")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--clip-duration", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--selected-count", type=int, default=20)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_root = ROOT / "outputs" / f"vjepa21_mlx_kernel_profile_{args.timestamp}"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    external_root = Path(Config.NEURAL_BRIDGE_EXTERNAL_ROOT).expanduser()
    selected_video = select_short_video(args.report, args.video_id)
    video_id = str(selected_video["video_id"])
    manifest_rows = [row for row in load_manifest_rows(args.manifest) if str(row.get("video_id")) == video_id]
    if not manifest_rows:
        raise ValueError(f"No manifest rows found for video_id={video_id}")
    video_path = resolve_external_path(manifest_rows[0]["media_path"], external_root)
    weights_dir = external_root / "models" / "vjepa21_mlx" / "vitg"

    setup_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    model = MlxVjepa21FeatureModel(str(weights_dir), image_size=args.image_size)
    setup_rows.append({"stage": "load_vjepa21_model", "seconds": time.perf_counter() - started})
    if args.selected_count > 0:
        n_states = model.config.num_hidden_layers + 1
        selected = [int(round(index)) for index in np.linspace(0, n_states - 1, args.selected_count)]
    else:
        extractor = MlxVjepa21Video(mlx_weights_dir=str(weights_dir), image_size=args.image_size, frequency=1.0)
        extractor._model = model
        selected = extractor._selected_hidden_state_indices()

    raw_window, window_info = build_real_window(
        video_path=video_path,
        duration_seconds=float(selected_video["duration_seconds"]),
        image_size=args.image_size,
        num_frames=model.num_frames,
        clip_duration=args.clip_duration,
    )
    setup_rows.extend(
        [
            {"stage": "ffmpeg_decode_grid", "seconds": window_info["ffmpeg_decode_seconds"]},
            {"stage": "sample_one_window", "seconds": window_info["window_sample_seconds"]},
        ]
    )
    started = time.perf_counter()
    preprocessed = _preprocess_video_batch(raw_window, args.image_size)
    setup_rows.append({"stage": "preprocess_one_window", "seconds": time.perf_counter() - started})
    video = mx.array(preprocessed)
    mx.eval(video)

    full_forward = lambda x: model.encoder.selected_token_mean_states(x, selected)
    eager_summary = measure_forward("eager_selected_token_mean_states", full_forward, video, repeats=args.repeats)
    profiled_output, profile_rows = profile_encoder_groups(model, video, selected, group_size=args.group_size)

    compile_summary: dict[str, Any]
    try:
        compiled_forward = mx.compile(full_forward)
        started = time.perf_counter()
        compiled_out = compiled_forward(video)
        mx.eval(compiled_out)
        compile_seconds = time.perf_counter() - started
        compiled_summary = measure_forward(
            "compiled_selected_token_mean_states",
            compiled_forward,
            video,
            repeats=args.repeats,
        )
        compiled_arr = np.asarray(compiled_out, dtype=np.float32)
        eager_arr = np.asarray(full_forward(video), dtype=np.float32)
        compiled_summary["compile_first_call_seconds"] = compile_seconds
        compiled_summary["max_abs_diff_vs_eager"] = float(np.max(np.abs(compiled_arr - eager_arr)))
        compiled_summary["compile_success"] = True
    except Exception as exc:  # noqa: BLE001 - capability probe
        compiled_summary = {
            "mode": "compiled_selected_token_mean_states",
            "compile_success": False,
            "compile_error": repr(exc),
        }

    stage_rows = setup_rows + profile_rows
    summary = {
        "created_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "video_id": video_id,
        "video_path": str(video_path),
        "image_size": args.image_size,
        "num_frames": model.num_frames,
        "clip_duration_seconds": args.clip_duration,
        "input_shape": list(preprocessed.shape),
        "selected_hidden_state_indices": selected,
        "selected_hidden_state_count": len(selected),
        "profiled_output_shape": list(profiled_output.shape),
        "profiled_output_nonfinite_count": int((~np.isfinite(profiled_output)).sum()),
        "window_info": window_info,
        "eager_forward": eager_summary,
        "compiled_forward": compiled_summary,
        "guardrails": {
            "full_veatic_encoding_run": False,
            "again_encoding_run": False,
            "benchmark_run": False,
            "models_trained": False,
            "precision_reduced": False,
            "accuracy_reduction_attempted": False,
        },
    }
    if compiled_summary.get("compile_success") and eager_summary.get("median_seconds"):
        compiled_median = float(compiled_summary.get("median_seconds") or 0.0)
        if compiled_median > 0:
            summary["compiled_speedup_vs_eager_median"] = float(eager_summary["median_seconds"]) / compiled_median

    write_csv(output_root / "vjepa21_mlx_kernel_profile_stages.csv", stage_rows)
    write_csv(output_root / "vjepa21_mlx_kernel_forward_comparison.csv", [eager_summary, compiled_summary])
    write_json(output_root / "vjepa21_mlx_kernel_profile_summary.json", summary)
    report = [
        "# V-JEPA 2.1 MLX Kernel Profile",
        "",
        "## Verdict",
        f"- Eager median forward seconds: `{eager_summary.get('median_seconds')}`",
        f"- Compile success: `{compiled_summary.get('compile_success', False)}`",
        f"- Compiled median forward seconds: `{compiled_summary.get('median_seconds')}`",
        f"- Compiled speedup vs eager: `{summary.get('compiled_speedup_vs_eager_median')}`",
        "- Precision reduced: `false`",
        "- Accuracy reduction attempted: `false`",
        "",
        "## Input",
        f"- Video: `{video_id}`",
        f"- Image size: `{args.image_size}`",
        f"- Input shape: `{list(preprocessed.shape)}`",
        f"- Selected hidden states: `{selected}`",
        "",
        "## Outputs",
        f"- Stage CSV: `{output_root / 'vjepa21_mlx_kernel_profile_stages.csv'}`",
        f"- Forward comparison CSV: `{output_root / 'vjepa21_mlx_kernel_forward_comparison.csv'}`",
        f"- Summary JSON: `{output_root / 'vjepa21_mlx_kernel_profile_summary.json'}`",
    ]
    (output_root / "vjepa21_mlx_kernel_profile_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
