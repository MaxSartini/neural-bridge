"""NeuralSet text extractor backed by MLX LLaMA hidden states.

This is an Apple Silicon bridge for mlx-community LLaMA folders.
It preserves NeuralSet's HuggingFaceText aggregation semantics while replacing
the backend model execution with MLX.
"""

from __future__ import annotations

from typing import Iterator

import mlx.core as mx
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralset.extractors import HuggingFaceText
from neuralset.extractors.text import TextDataset


class MlxText(HuggingFaceText):
    """Drop-in text extractor for MLX-format LLaMA models."""

    def repo_exists(self) -> bool:
        return True

    @property
    def model(self):
        if not hasattr(self, "_model"):
            from mlx_lm import load
            from transformers import AutoTokenizer

            model, _ = load(self.model_name, lazy=False)
            self._model = model
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                truncation_side="left",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._pad_id = self._tokenizer.eos_token_id
        return self._model

    def _get_data(self, events: list) -> Iterator[np.ndarray]:
        dataset = TextDataset(events)
        dloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for target_words, context in dloader:
                text = context if self.contextualized else target_words
                if isinstance(text, tuple):
                    text = list(text)
                if not all(text):
                    raise ValueError(f"Empty text or context for target_words {target_words!r}")

                inputs = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                )
                input_ids_np = inputs["input_ids"]
                input_ids = mx.array(input_ids_np.astype(np.int32))
                hidden_states = self._mlx_hidden_states(input_ids)
                n_layers, _, n_tokens, _ = hidden_states.shape

                for i, target_word in enumerate(target_words):
                    hidden_state = hidden_states[:, i]
                    n_pads = int(np.sum(input_ids_np[i] == self._pad_id))
                    if n_pads:
                        hidden_state = hidden_state[:, :-n_pads]

                    if self.contextualized:
                        prefix = context[i][: -len(target_word)].rstrip()
                        n_prefix = (
                            len(self.tokenizer.encode(prefix, add_special_tokens=False))
                            if prefix
                            else 0
                        )
                        n_target = n_tokens - n_pads - n_prefix
                        word_state = hidden_state[:, -max(1, n_target) :]
                    else:
                        word_state = hidden_state

                    word_state_np = np.asarray(word_state, dtype=np.float32)
                    word_state_np = self._aggregate_tokens_np(word_state_np)
                    if not self.cache_all_layers and self.cache_n_layers is None:
                        word_state_np = self._aggregate_layers(word_state_np)
                    if np.isnan(word_state_np).any():
                        raise ValueError(
                            f"NaN in output for target_word {target_word} with context {context}"
                        )
                    yield word_state_np
                del hidden_states, input_ids
                mx.clear_cache()
        mx.clear_cache()

    def _mlx_hidden_states(self, input_ids: mx.array) -> np.ndarray:
        model = self.model.model
        h = model.embed_tokens(input_ids)
        states = [h]

        cache = [None] * len(model.layers)
        fa_mask = self._attention_mask(model, h, cache, model.fa_idx)
        swa_mask = None
        if model.swa_idx is not None:
            swa_mask = self._attention_mask(
                model,
                h,
                cache,
                model.swa_idx,
                window_size=model.sliding_window,
            )

        for idx, layer in enumerate(model.layers):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=cache[idx])
            if idx < len(model.layers) - 1:
                states.append(h)

        states.append(model.norm(h))
        if self.cache_n_layers is not None and self.cache_n_layers < len(states):
            indices = np.linspace(0, len(states) - 1, self.cache_n_layers).round().astype(int)
            states = [states[index] for index in indices]
        stacked = mx.stack(states)
        mx.eval(stacked)
        return np.asarray(stacked, dtype=np.float32)

    def _attention_mask(self, model, h, cache, idx: int, window_size=None):
        from mlx_lm.models.base import create_attention_mask

        return create_attention_mask(h, cache[idx], window_size=window_size)

    def _aggregate_tokens_np(self, latents: np.ndarray) -> np.ndarray:
        latents = self._layer_subselection(latents)
        if self.token_aggregation is None:
            return latents
        if self.token_aggregation == "first":
            return latents[:, 0]
        if self.token_aggregation == "last":
            return latents[:, -1]
        if self.token_aggregation == "mean":
            return latents.mean(axis=1)
        if self.token_aggregation == "sum":
            return latents.sum(axis=1)
        if self.token_aggregation == "max":
            return latents.max(axis=1)
        raise ValueError(f"Unknown token aggregation: {self.token_aggregation}")
