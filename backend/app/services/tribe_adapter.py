"""Lazy optional adapter for TRIBE v2 neuro-response prediction."""

import json
import os
import platform
import subprocess
import sys
import tempfile
import gc
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..config import Config
from ..utils.logger import get_logger
from .neuro_roi_calibrator import NeuroRoiCalibrator
from .subcortical_roi_adapter import SubcorticalRoiAdapter

logger = get_logger('neural_bridge.tribe_adapter')


class TribeAdapter:
    _last_event_quality: Dict[str, Any] = {}
    _last_segment_quality: Dict[str, Any] = {}

    def is_available(self) -> bool:
        for path in (Config.TRIBE_APPLE_SILICON_SOURCE_DIR, Config.TRIBE_OFFICIAL_SOURCE_DIR):
            if os.path.isdir(path):
                return True
        return False

    def predict(
        self,
        stimulus_text: str = "",
        stimulus_type: str = "text",
        media_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._configure_mps_memory_limit()
        backend = backend or Config.NEURO_PRIOR_MODE
        if backend == "apple_silicon_tribe":
            return self._predict_with_apple_silicon_branch(stimulus_text, stimulus_type, media_path, output_dir)
        if backend == "official_tribe":
            return self._predict_with_official_tribe(stimulus_text, stimulus_type, media_path, output_dir)
        if backend == "tribe_mlx":
            return self._predict_with_mlx(stimulus_text, stimulus_type, media_path, output_dir)
        return {"success": False, "backend": backend, "error": f"Unsupported TRIBE backend: {backend}"}

    @staticmethod
    def _configure_mps_memory_limit() -> None:
        try:
            import torch

            if torch.backends.mps.is_available():
                fraction = max(0.1, min(0.8, float(Config.TRIBE_MPS_MEMORY_FRACTION)))
                torch.mps.set_per_process_memory_fraction(fraction)
                os.environ.setdefault("TRIBE_VIDEO_DTYPE", Config.TRIBE_VIDEO_DTYPE)
        except Exception as exc:
            logger.warning("Unable to configure bounded MPS memory: %s", exc)

    def _predict_with_apple_silicon_branch(
        self,
        stimulus_text: str,
        stimulus_type: str,
        media_path: Optional[str],
        output_dir: Optional[str],
    ) -> Dict[str, Any]:
        return self._predict_with_source_dir(
            Config.TRIBE_APPLE_SILICON_SOURCE_DIR,
            "apple_silicon_tribe",
            stimulus_text,
            stimulus_type,
            media_path,
            output_dir,
        )

    def _predict_with_official_tribe(
        self,
        stimulus_text: str,
        stimulus_type: str,
        media_path: Optional[str],
        output_dir: Optional[str],
    ) -> Dict[str, Any]:
        return self._predict_with_source_dir(
            Config.TRIBE_OFFICIAL_SOURCE_DIR,
            "official_tribe",
            stimulus_text,
            stimulus_type,
            media_path,
            output_dir,
        )

    def _predict_with_mlx(
        self,
        stimulus_text: str,
        stimulus_type: str,
        media_path: Optional[str],
        output_dir: Optional[str],
    ) -> Dict[str, Any]:
        if not Config.TRIBE_MLX_ENABLED:
            return {"success": False, "backend": "tribe_mlx", "error": "TRIBE_MLX_ENABLED is false."}
        feature_path = media_path if media_path and media_path.endswith((".npy", ".npz")) else ""
        try:
            self._enable_local_huggingface_model_paths()
            from .mlx_tribe_encoder import MlxTribeEncoder

            encoder = MlxTribeEncoder(self._resolve_path(Config.TRIBE_MLX_DIR))
            if feature_path:
                with np.load(feature_path) as bundle:
                    features = {
                        key: np.asarray(bundle[key])
                        for key in ("text", "audio", "video")
                        if key in bundle.files
                    }
                    subcortical_features = {
                        key: np.asarray(bundle[f"subcortical_{key}"])
                        for key in ("text", "audio", "video")
                        if f"subcortical_{key}" in bundle.files
                    }
                if not features:
                    return {
                        "success": False,
                        "backend": "tribe_mlx",
                        "error": "Feature archive contains none of: text, audio, video.",
                    }
                raw_preds = encoder.predict(features)
                preds = np.transpose(raw_preds, (0, 2, 1)).reshape(-1, raw_preds.shape[1])
                del encoder
                gc.collect()
                subcortical_preds = self._predict_subcortical_from_feature_archive(
                    feature_path, subcortical_features
                )
                segments: Any = {}
            else:
                preds, segments, events = self._extract_features_and_predict_mlx(
                    encoder, stimulus_text, stimulus_type, media_path
                )
                # Run the distinct exact-provenance subcortical branch only
                # after cortical MLX inference has completed.
                del encoder
                gc.collect()
                subcortical_preds = self._predict_subcortical_events_isolated(events)
            summary = self._summarise_bold_output_to_neuro_prior(
                preds,
                segments,
                output_dir,
                "tribe_mlx",
                subcortical_preds=subcortical_preds,
            )
            summary.update({"success": True, "backend": "tribe_mlx"})
            return summary
        except Exception as exc:
            logger.exception(f"TRIBE-MLX prediction failed: {exc}")
            return {"success": False, "backend": "tribe_mlx", "error": str(exc)}

    def _extract_features_and_predict_mlx(
        self,
        encoder: Any,
        stimulus_text: str,
        stimulus_type: str,
        media_path: Optional[str],
    ) -> tuple[np.ndarray, list, Any]:
        source_dir = self._resolve_path(Config.TRIBE_APPLE_SILICON_SOURCE_DIR)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        from tribev2 import TribeModel  # type: ignore

        model = TribeModel.from_pretrained(
            self._model_source(),
            cache_folder=self._resolve_path(Config.TRIBE_CACHE_DIR),
            device="cpu",
            config_update=self._config_update(),
        )
        tmp_path = None
        try:
            if stimulus_type == "text":
                if not stimulus_text.strip():
                    raise ValueError("No stimulus_text supplied.")
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                    tmp.write(stimulus_text)
                    tmp_path = tmp.name
                events = model.get_events_dataframe(text_path=tmp_path)
            elif media_path:
                events = model.get_events_dataframe(**{
                    "video_path" if stimulus_type == "video" else "audio_path": media_path
                })
            else:
                raise ValueError(f"No media_path supplied for {stimulus_type} stimulus.")
            events = self._repair_text_context(events)
            self._last_event_quality = events.attrs.get("neural_bridge_quality", {})
            loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
            predictions = []
            all_segments = []
            total_expanded_segments = 0
            kept_expanded_segments = 0
            for batch in loader:
                expanded_segments = []
                for segment in batch.segments:
                    for offset in np.arange(0, segment.duration - 1e-2, model.data.TR):
                        expanded_segments.append(segment.copy(offset=offset, duration=model.data.TR))
                keep = np.array([len(segment.ns_events) > 0 for segment in expanded_segments])
                total_expanded_segments += int(len(expanded_segments))
                kept_expanded_segments += int(np.sum(keep))
                features = {
                    key: batch.data[key].detach().cpu().numpy()
                    for key in ("text", "audio", "video")
                    if key in batch.data
                }
                batch_preds = encoder.predict(features)
                batch_preds = np.transpose(batch_preds, (0, 2, 1)).reshape(-1, batch_preds.shape[1])
                predictions.append(batch_preds[keep])
                all_segments.extend(segment for index, segment in enumerate(expanded_segments) if keep[index])
            if not predictions:
                raise ValueError("No TRIBE-compatible features were extracted.")
            self._last_segment_quality = {
                "expanded_segments": total_expanded_segments,
                "kept_segments": kept_expanded_segments,
                "dropped_segments": total_expanded_segments - kept_expanded_segments,
                "retention_ratio": (
                    kept_expanded_segments / total_expanded_segments
                    if total_expanded_segments
                    else 0.0
                ),
            }
            return np.concatenate(predictions), all_segments, events
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _predict_with_source_dir(
        self,
        source_dir: str,
        backend: str,
        stimulus_text: str,
        stimulus_type: str,
        media_path: Optional[str],
        output_dir: Optional[str],
    ) -> Dict[str, Any]:
        source_dir = self._resolve_path(source_dir)
        if not os.path.isdir(source_dir):
            return {"success": False, "backend": backend, "error": f"TRIBE source dir not found: {source_dir}"}

        try:
            self._enable_local_huggingface_model_paths()
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
            from tribev2 import TribeModel  # type: ignore
        except Exception as exc:
            logger.warning(f"Failed to import TRIBE from {source_dir}: {exc}")
            return {"success": False, "backend": backend, "error": f"TRIBE import failed: {exc}"}

        tmp_path = None
        try:
            model_source = self._model_source()
            model = TribeModel.from_pretrained(
                model_source,
                cache_folder=self._resolve_path(Config.TRIBE_CACHE_DIR),
                device=self._resolve_device(),
                config_update=self._config_update(),
            )
            event_kwargs: Dict[str, Any] = {}
            if stimulus_type == "text":
                if not stimulus_text.strip():
                    return {"success": False, "backend": backend, "error": "No stimulus_text supplied."}
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
                    tmp.write(stimulus_text)
                    tmp_path = tmp.name
                event_kwargs["text_path"] = tmp_path
            elif media_path:
                key = "video_path" if stimulus_type == "video" else "audio_path"
                event_kwargs[key] = media_path
            else:
                return {"success": False, "backend": backend, "error": f"No media_path supplied for {stimulus_type} stimulus."}

            events = model.get_events_dataframe(**event_kwargs)
            events = self._repair_text_context(events)
            self._last_event_quality = events.attrs.get("neural_bridge_quality", {})
            self._last_segment_quality = {}
            preds, segments = model.predict(events=events)
            del model
            gc.collect()
            subcortical_preds = self._predict_subcortical_events(events)
            summary = self._summarise_bold_output_to_neuro_prior(
                preds,
                segments,
                output_dir,
                backend,
                subcortical_preds=subcortical_preds,
            )
            summary["success"] = True
            summary["backend"] = backend
            return summary
        except Exception as exc:
            logger.warning(f"TRIBE prediction failed for {backend}: {exc}")
            return {"success": False, "backend": backend, "error": str(exc)}
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _resolve_device(self) -> str:
        configured = (getattr(Config, "TRIBE_DEVICE", "auto") or "auto").lower()
        if configured in {"cpu", "mps", "cuda"}:
            return configured
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _enable_local_huggingface_model_paths() -> None:
        """Allow neuralset extractors to use verified local offline model dirs."""
        from neuralset.extractors.base import HuggingFaceMixin

        if getattr(HuggingFaceMixin, "_neural_bridge_local_path_patch", False):
            return
        original_repo_exists = HuggingFaceMixin.repo_exists

        def repo_exists_or_local(self: Any) -> bool:
            model_name = os.path.expanduser(str(self.model_name))
            return os.path.isdir(model_name) or original_repo_exists(self)

        HuggingFaceMixin.repo_exists = repo_exists_or_local
        HuggingFaceMixin._neural_bridge_local_path_patch = True

    def _model_source(self) -> str:
        local_model_dir = self._resolve_path(os.path.join("models", "tribe", "facebook-tribev2"))
        if os.path.exists(os.path.join(local_model_dir, "config.yaml")) and os.path.exists(
            os.path.join(local_model_dir, "best.ckpt")
        ):
            return local_model_dir
        return Config.TRIBE_MODEL_ID

    def _config_update(self) -> Dict[str, Any]:
        update: Dict[str, Any] = {}
        local_text_encoder = self._resolve_path(Config.TRIBE_TEXT_ENCODER_LOCAL_DIR)
        mlx_text_encoder = self._resolve_path(Config.TRIBE_TEXT_ENCODER_MLX_DIR)
        audio_encoder = self._encoder_source(Config.TRIBE_AUDIO_ENCODER_LOCAL_DIR, Config.TRIBE_AUDIO_ENCODER_ID)
        video_encoder = self._encoder_source(Config.TRIBE_VIDEO_ENCODER_LOCAL_DIR, Config.TRIBE_VIDEO_ENCODER_ID)
        video_encoder_backend = self._resolve_video_encoder_backend()

        update["data.num_workers"] = 0
        update["data.batch_size"] = 1
        update["data.text_feature.batch_size"] = Config.TRIBE_TEXT_BATCH_SIZE
        if self._looks_like_transformers_model_dir(local_text_encoder):
            update["data.text_feature.model_name"] = local_text_encoder
        elif self._looks_like_mlx_model_dir(mlx_text_encoder):
            from .mlx_text_extractor import MlxText  # noqa: F401

            update["data.text_feature.name"] = "MlxText"
            update["data.text_feature.model_name"] = mlx_text_encoder
            update["data.text_feature.device"] = "cpu"
        else:
            update["data.text_feature.model_name"] = Config.TRIBE_TEXT_ENCODER_ID
        update["data.audio_feature.model_name"] = audio_encoder
        update["data.video_feature.image.model_name"] = video_encoder
        update["data.image_feature.image.model_name"] = video_encoder
        if video_encoder_backend == "mlx":
            from .mlx_vjepa2_cortical import MlxVjepa2Video  # noqa: F401

            update["data.video_feature.name"] = "MlxVjepa2Video"
            update["data.video_feature.mlx_weights_dir"] = self._resolve_path(
                Config.TRIBE_VIDEO_ENCODER_MLX_DIR
            )
            update["data.video_feature.processor_model_name"] = video_encoder
            update["data.video_feature.cache_model_name"] = (
                "mlx-vjepa2-vitg-fpc64-256-selected-hidden-states-v1"
            )
        update["data.video_feature.num_frames"] = Config.TRIBE_VIDEO_NUM_FRAMES
        video_device = (Config.TRIBE_VIDEO_DEVICE or "auto").lower()
        if video_device == "auto":
            try:
                import torch

                video_device = "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                video_device = "cpu"
        if (
            video_device == "mps"
            and "vjepa2-vitg" in str(video_encoder).lower()
            and not Config.TRIBE_ALLOW_UNSAFE_VITG_MPS
        ):
            logger.warning(
                "Forcing cortical V-JEPA2 ViT-G to CPU: sustained MPS inference "
                "kernel-panicked this 32 GB Apple Silicon host. Set "
                "TRIBE_ALLOW_UNSAFE_VITG_MPS=true only as an explicit unsafe ablation."
            )
            video_device = "cpu"
        update["data.video_feature.image.device"] = video_device
        update["data.image_feature.image.device"] = video_device
        return update

    def _resolve_video_encoder_backend(self) -> str:
        configured = (getattr(Config, "TRIBE_VIDEO_ENCODER_BACKEND", "auto") or "auto").lower()
        if configured in {"torch", "transformers", "hf", "mps", "cpu"}:
            return "torch"
        if configured == "mlx":
            return "mlx"
        mlx_dir = self._resolve_path(getattr(Config, "TRIBE_VIDEO_ENCODER_MLX_DIR", ""))
        if os.path.exists(os.path.join(mlx_dir, "model.safetensors")) and os.path.exists(
            os.path.join(mlx_dir, "config.json")
        ):
            return "mlx"
        return "torch"

    def _encoder_source(self, local_dir: str, repo_id: str) -> str:
        resolved = self._resolve_path(local_dir)
        if self._looks_like_encoder_model_dir(resolved):
            return resolved
        return repo_id

    def _looks_like_encoder_model_dir(self, path: str) -> bool:
        """Validate non-text HF encoders, which correctly have no tokenizer."""
        if not os.path.isdir(path):
            return False
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_weights = any(
            name.endswith((".safetensors", ".bin", ".pt"))
            for name in os.listdir(path)
        )
        return has_config and has_weights

    def _looks_like_transformers_model_dir(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        has_config = os.path.exists(os.path.join(path, "config.json"))
        has_weights = any(
            name.endswith((".safetensors", ".bin", ".pt"))
            for name in os.listdir(path)
        )
        has_tokenizer = any(
            os.path.exists(os.path.join(path, name))
            for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
        )
        return has_config and has_weights and has_tokenizer

    def _looks_like_mlx_model_dir(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        names = set(os.listdir(path))
        if any(name.endswith(".part") or name.startswith("downloading_") for name in names):
            return False
        has_config = "config.json" in names
        has_tokenizer = "tokenizer.json" in names or "tokenizer.model" in names
        has_weights = any(name.endswith(".safetensors") for name in names)
        return has_config and has_tokenizer and has_weights

    def _repair_text_context(self, events: Any) -> Any:
        if "type" not in events:
            return events
        events = events.copy()
        quality: Dict[str, Any] = {
            "event_count": int(len(events)),
            "type_counts": {
                str(key): int(value)
                for key, value in events["type"].value_counts(dropna=False).to_dict().items()
            },
            "missing_text": True,
            "missing_audio": "Audio" not in set(events["type"].astype(str)),
            "missing_video": "Video" not in set(events["type"].astype(str)),
            "word_duration_repairs": 0,
            "word_duration_repair_method": "none",
            "empty_context_repairs": 0,
        }
        if "context" not in events:
            events.attrs["neural_bridge_quality"] = quality
            return events
        word_mask = events["type"] == "Word"
        if not word_mask.any():
            events.attrs["neural_bridge_quality"] = quality
            return events
        quality["missing_text"] = False
        events, word_mask, degenerate_quality = self._drop_degenerate_text_events(
            events, word_mask
        )
        quality.update(degenerate_quality)
        if not word_mask.any():
            quality["missing_text"] = True
            events.attrs["neural_bridge_quality"] = quality
            return events
        empty_context = events["context"].isna() | (events["context"].astype(str).str.len() == 0)
        repair_mask = word_mask & empty_context
        if repair_mask.any() and "sentence" in events:
            events.loc[repair_mask, "context"] = events.loc[repair_mask, "sentence"].fillna(
                events.loc[repair_mask, "text"]
            )
            quality["empty_context_repairs"] = int(repair_mask.sum())
        if "duration" in events:
            word_durations = events.loc[word_mask, "duration"].astype(float)
            bad_duration = word_mask & (
                events["duration"].isna() | (events["duration"].astype(float) <= 0.0)
            )
            if bad_duration.any():
                repaired = events.loc[word_mask, ["start", "duration"]].copy()
                starts = repaired["start"].astype(float).to_numpy()
                positive = repaired["duration"].astype(float).to_numpy()
                positive = positive[positive > 0.0]
                fallback = float(np.median(positive)) if positive.size else 0.18
                for local_index, row_index in enumerate(repaired.index):
                    current = float(events.at[row_index, "duration"] or 0.0)
                    if current > 0.0:
                        continue
                    if local_index + 1 < len(starts):
                        inferred = max(0.02, min(1.0, float(starts[local_index + 1] - starts[local_index])))
                    else:
                        inferred = fallback
                    events.at[row_index, "duration"] = inferred
                quality["word_duration_repairs"] = int(bad_duration.sum())
                quality["word_duration_repair_method"] = "next_word_start_or_median_positive_duration"
            quality["null_word_durations_after_repair"] = int(
                (events.loc[word_mask, "duration"].astype(float) <= 0.0).sum()
            )
            quality["word_duration_min"] = float(events.loc[word_mask, "duration"].astype(float).min())
            quality["word_duration_median"] = float(events.loc[word_mask, "duration"].astype(float).median())
            quality["word_duration_max"] = float(events.loc[word_mask, "duration"].astype(float).max())
        events.attrs["neural_bridge_quality"] = quality
        return events

    def _drop_degenerate_text_events(self, events: Any, word_mask: Any) -> tuple[Any, Any, Dict[str, Any]]:
        """Remove hallucinated silence transcripts while preserving audio/video.

        MLX Whisper can hallucinate dense repeated tokens on low-speech clips
        (for example hundreds of identical "True" words with zero duration).
        Feeding that into TRIBE makes text features dominate the runtime and
        degrades scientific validity.  When the transcript is clearly
        degenerate, drop only text-derived events and keep the audio/video
        modalities active.
        """

        quality: Dict[str, Any] = {
            "degenerate_text_dropped": False,
            "degenerate_text_reason": "none",
        }
        if not word_mask.any() or "text" not in events:
            return events, word_mask, quality

        words = (
            events.loc[word_mask, "text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        non_empty_words = words[words != ""]
        if non_empty_words.empty:
            quality.update(
                {
                    "degenerate_text_dropped": True,
                    "degenerate_text_reason": "empty_transcript_words",
                }
            )
        else:
            total_words = int(len(words))
            unique_words = int(non_empty_words.nunique())
            top_word_fraction = float(non_empty_words.value_counts(normalize=True).iloc[0])
            zero_duration_fraction = 0.0
            if "duration" in events:
                durations = events.loc[word_mask, "duration"].astype(float)
                zero_duration_fraction = float((durations <= 0.0).mean())
            span_seconds = 0.0
            if "start" in events:
                starts = events.loc[word_mask, "start"].astype(float)
                span_seconds = max(0.0, float(starts.max() - starts.min()))
            word_density = total_words / max(span_seconds, 1.0)
            quality.update(
                {
                    "word_unique_count": unique_words,
                    "top_word_fraction": top_word_fraction,
                    "zero_duration_word_fraction": zero_duration_fraction,
                    "word_density_per_second": word_density,
                }
            )
            repeated_single_token = (
                total_words >= 40
                and unique_words <= 2
                and top_word_fraction >= 0.95
            )
            malformed_dense_transcript = (
                total_words >= 100
                and zero_duration_fraction >= 0.50
                and word_density >= 8.0
            )
            if repeated_single_token and zero_duration_fraction >= 0.25:
                quality.update(
                    {
                        "degenerate_text_dropped": True,
                        "degenerate_text_reason": "repeated_single_token_zero_duration_transcript",
                    }
                )
            elif malformed_dense_transcript:
                quality.update(
                    {
                        "degenerate_text_dropped": True,
                        "degenerate_text_reason": "dense_zero_duration_transcript",
                    }
                )

        if not quality["degenerate_text_dropped"]:
            return events, word_mask, quality

        text_derived = events["type"].isin(["Word", "Sentence", "Text"])
        quality["degenerate_text_events_removed"] = int(text_derived.sum())
        events = events.loc[~text_derived].copy()
        word_mask = events["type"] == "Word"
        return events, word_mask, quality

    def _resolve_path(self, path: str) -> str:
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        return os.path.abspath(os.path.join(project_root, path))

    def _summarise_bold_output_to_neuro_prior(
        self,
        preds: Any,
        segments: Any,
        output_dir: Optional[str],
        backend: str,
        subcortical_preds: Any = None,
    ) -> Dict[str, Any]:
        arr = np.asarray(preds, dtype=float)
        arr = np.nan_to_num(arr)
        calibration = NeuroRoiCalibrator().calibrate_predictions(arr)
        global_metrics = calibration.get("roi_summary", {}).get("global_metrics", {})
        mean_activation = float(np.clip(global_metrics.get("mean_abs", np.mean(np.abs(arr))), 0.0, 1.0))
        temporal_variance = float(np.clip(global_metrics.get("temporal_variance", np.var(arr)), 0.0, 1.0))
        peak_response = float(np.clip(global_metrics.get("peak_abs", np.max(np.abs(arr)) if arr.size else 0.0), 0.0, 1.0))
        volatility = float(np.clip(global_metrics.get("std", np.std(arr)), 0.0, 1.0))
        subcortical_summary: Dict[str, Any] = {}
        subcortical_arr = None
        if subcortical_preds is not None:
            uncertainty = None
            if isinstance(subcortical_preds, dict):
                uncertainty = subcortical_preds.get("subject_disagreement")
                subcortical_preds = subcortical_preds.get("predictions")
            subcortical_arr = np.nan_to_num(np.asarray(subcortical_preds, dtype=np.float32))
            subcortical_summary = SubcorticalRoiAdapter().project(subcortical_arr)
            subcortical_summary.pop("region_trajectories", None)
            if uncertainty is not None:
                uncertainty_arr = np.nan_to_num(np.asarray(uncertainty, dtype=np.float32))
                subcortical_summary["subject_disagreement"] = {
                    "mean": float(np.mean(uncertainty_arr)),
                    "p95": float(np.percentile(uncertainty_arr, 95)),
                    "peak": float(np.max(uncertainty_arr)),
                    "interpretation": "Dispersion across the ten measured Lahner participant heads.",
                }

        raw_output_path = ""
        if output_dir and Config.NEURO_PRIOR_SAVE_RAW_OUTPUT:
            os.makedirs(output_dir, exist_ok=True)
            raw_npz = os.path.join(output_dir, "tribe_raw_output.npz")
            raw_arrays = {"predictions": arr}
            if subcortical_arr is not None:
                raw_arrays["subcortical_predictions"] = subcortical_arr
            event_quality = dict(getattr(self, "_last_event_quality", {}) or {})
            segment_quality = dict(getattr(self, "_last_segment_quality", {}) or {})
            raw_arrays["modality_missing_flags"] = np.asarray(
                [
                    float(bool(event_quality.get("missing_text", True))),
                    float(bool(event_quality.get("missing_audio", True))),
                    float(bool(event_quality.get("missing_video", True))),
                ],
                dtype=np.float32,
            )
            raw_arrays["segment_retention_features"] = np.asarray(
                [
                    float(segment_quality.get("retention_ratio", 0.0)),
                    float(segment_quality.get("kept_segments", 0.0)),
                    float(segment_quality.get("dropped_segments", 0.0)),
                    float(event_quality.get("word_duration_repairs", 0.0)),
                    float(event_quality.get("null_word_durations_after_repair", 0.0)),
                ],
                dtype=np.float32,
            )
            np.savez(raw_npz, **raw_arrays)
            raw_output_path = raw_npz
            with open(os.path.join(output_dir, "tribe_segments.json"), "w", encoding="utf-8") as f:
                json.dump({"segments": str(segments)}, f, ensure_ascii=False, indent=2)
            with open(os.path.join(output_dir, "tribe_summary.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "mean_activation_proxy": mean_activation,
                    "temporal_variance_proxy": temporal_variance,
                    "peak_response_proxy": peak_response,
                    "response_volatility_proxy": volatility,
                    "roi_summary": calibration.get("roi_summary", {}),
                    "behavioural_axes": calibration.get("behavioural_axes", {}),
                    "calibration_trace": calibration.get("calibration_trace", {}),
                    "subcortical_summary": subcortical_summary,
                    "event_quality": event_quality,
                    "segment_quality": segment_quality,
                    "backend": backend,
                }, f, ensure_ascii=False, indent=2)

        profile = {
            "raw_backend": backend,
            "raw_output_path": raw_output_path,
            "subcortical_summary": subcortical_summary,
        }
        profile.update(calibration)
        return profile

    def _predict_subcortical_from_feature_archive(
        self, feature_path: str, features: Dict[str, np.ndarray]
    ) -> Any:
        """Run subcortical inference only for explicitly compatible features."""
        if not Config.TRIBE_ENABLE_SUBCORTICAL:
            return None
        metadata_path = os.path.splitext(feature_path)[0] + ".json"
        if not os.path.exists(metadata_path):
            logger.warning(
                "Skipping subcortical inference: feature archive lacks provenance metadata %s",
                metadata_path,
            )
            return None
        metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
        expected = {
            "text": "Qwen/Qwen3-0.6B",
            "audio": "facebook/w2v-bert-2.0",
            "video": "facebook/vjepa2-vitl-fpc64-256",
        }
        actual = metadata.get("subcortical_feature_models") or {}
        available = [key for key in ("text", "audio", "video") if key in features]
        mismatches = [
            key for key in available if actual.get(key) != expected[key]
        ]
        if not available or mismatches:
            logger.warning(
                "Skipping subcortical inference: incompatible or missing feature provenance; "
                "available=%s mismatches=%s",
                available,
                mismatches,
            )
            return None
        from .mlx_subcortical_tribe_encoder import MlxSubcorticalTribeEncoder

        predictions, disagreement = MlxSubcorticalTribeEncoder(
            self._resolve_path(Config.TRIBE_SUBCORTICAL_LOCAL_DIR)
        ).predict_with_uncertainty(features)
        return {
            "predictions": np.transpose(predictions, (0, 2, 1)).reshape(-1, predictions.shape[1]),
            "subject_disagreement": np.transpose(disagreement, (0, 2, 1)).reshape(
                -1, disagreement.shape[1]
            ),
        }

    def _predict_subcortical_events(self, events: Any) -> Any:
        """Run the exact subcortical model after releasing the cortical model."""
        if not Config.TRIBE_ENABLE_SUBCORTICAL:
            return None
        model_dir = self._resolve_path(Config.TRIBE_SUBCORTICAL_LOCAL_DIR)
        checkpoint = os.path.join(model_dir, "best.ckpt")
        if not os.path.exists(checkpoint):
            logger.warning(
                "Skipping subcortical inference: converted Tribe checkpoint is absent at %s",
                checkpoint,
            )
            return None
        try:
            from tribev2 import TribeModel  # type: ignore

            self._enable_local_huggingface_model_paths()
            model = TribeModel.from_pretrained(
                model_dir,
                checkpoint_name="best.ckpt",
                cache_folder=self._resolve_path(Config.TRIBE_CACHE_DIR),
                device=self._resolve_device(),
                config_update=self._subcortical_config_update(),
            )
            predictions, _ = model.predict(events=events)
            return predictions
        except Exception as exc:
            logger.warning(f"Subcortical TRIBE prediction failed: {exc}")
            if Config.NEURO_PRIOR_STRICT:
                raise
            return None
        finally:
            if "model" in locals():
                del model
            gc.collect()

    def _predict_subcortical_events_isolated(self, events: Any) -> Any:
        """Run subcortical inference in a fresh process so MPS memory is reusable."""
        if not Config.TRIBE_ENABLE_SUBCORTICAL:
            return None
        script = Path(__file__).resolve().parents[2] / "scripts" / "run_subcortical_events.py"
        with tempfile.TemporaryDirectory(prefix="neural_bridge-subcortical-") as tmp:
            root = Path(tmp)
            events_path = root / "events.pkl"
            output_path = root / "predictions.npz"
            events.to_pickle(events_path)
            env = os.environ.copy()
            env["TRIBE_VIDEO_WINDOW_BATCH_SIZE"] = str(
                max(1, int(Config.TRIBE_SUBCORTICAL_VIDEO_WINDOW_BATCH_SIZE))
            )
            env["TRIBE_SUBCORTICAL_TEXT_BATCH_SIZE"] = "1"
            env["TRIBE_SUBCORTICAL_TEXT_DEVICE"] = "cpu"
            env["NEURO_PRIOR_STRICT"] = "true"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--events",
                    str(events_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                env=env,
            )
            if completed.returncode != 0 or not output_path.exists():
                logger.warning(
                    "Isolated subcortical TRIBE prediction failed with return code %s",
                    completed.returncode,
                )
                if Config.NEURO_PRIOR_STRICT:
                    raise RuntimeError("Isolated subcortical TRIBE prediction failed")
                return None
            with np.load(output_path, allow_pickle=False) as bundle:
                return bundle["predictions"]

    def _subcortical_config_update(self) -> Dict[str, Any]:
        update: Dict[str, Any] = {
            "data.num_workers": 0,
            "data.batch_size": 1,
            "data.text_feature.batch_size": Config.TRIBE_SUBCORTICAL_TEXT_BATCH_SIZE,
            "data.text_feature.model_name": self._resolve_path(
                Config.TRIBE_SUBCORTICAL_TEXT_ENCODER_LOCAL_DIR
            ),
            "data.text_feature.device": Config.TRIBE_SUBCORTICAL_TEXT_DEVICE,
            "data.audio_feature.model_name": self._resolve_path(
                Config.TRIBE_SUBCORTICAL_AUDIO_ENCODER_LOCAL_DIR
            ),
            "data.audio_feature.device": self._resolve_device(),
            "data.video_feature.image.model_name": self._resolve_path(
                Config.TRIBE_SUBCORTICAL_VIDEO_ENCODER_LOCAL_DIR
            ),
            "data.video_feature.image.device": self._resolve_device(),
            "data.video_feature.image.batch_size": 1,
            "data.video_feature.num_frames": Config.TRIBE_VIDEO_NUM_FRAMES,
        }
        return update
