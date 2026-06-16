"""
EmbeddingService — local embedding via LM Studio (OpenAI-compatible API).

Uses nomic-embed-text-v1.5 on LM Studio port 1235 (768-dim vectors).
"""

import logging
import time
from typing import List, Optional

from openai import OpenAI

from ..config import Config

logger = logging.getLogger('neural_bridge.embedding')


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingService:
    """Generate embeddings using LM Studio's /v1/embeddings endpoint."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
    ):
        self.model = model or Config.EMBEDDING_MODEL
        self.base_url = (base_url or Config.EMBEDDING_BASE_URL).rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key='lm-studio',
            timeout=timeout,
        )

        self._cache: dict[str, List[float]] = {}
        self._cache_max_size = 2000

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        text = text.strip()
        if text in self._cache:
            return self._cache[text]

        vectors = self._request_embeddings([text])
        vector = vectors[0]
        self._cache_put(text, vector)
        return vector

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            text = text.strip() if text else ""
            if text in self._cache:
                results[i] = self._cache[text]
            elif text:
                uncached_indices.append(i)
                uncached_texts.append(text)
            else:
                results[i] = [0.0] * Config.EMBEDDING_DIMENSIONS

        if uncached_texts:
            all_vectors: List[List[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start:start + batch_size]
                vectors = self._request_embeddings(batch)
                all_vectors.extend(vectors)

            for idx, vec, text in zip(uncached_indices, all_vectors, uncached_texts):
                results[idx] = vec
                self._cache_put(text, vec)

        return results  # type: ignore

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(texts):
                    raise EmbeddingError(
                        f"Expected {len(texts)} embeddings, got {len(vectors)}"
                    )
                return vectors
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise EmbeddingError(
            f"LM Studio embedding failed after {self.max_retries} retries: {last_error}"
        )

    def _cache_put(self, text: str, vector: List[float]) -> None:
        if len(self._cache) >= self._cache_max_size:
            keys_to_remove = list(self._cache.keys())[:self._cache_max_size // 10]
            for key in keys_to_remove:
                del self._cache[key]
        self._cache[text] = vector

    def health_check(self) -> bool:
        try:
            vec = self.embed("health check")
            return len(vec) > 0
        except Exception:
            return False
