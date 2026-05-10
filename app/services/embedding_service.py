from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import List, Sequence

import httpx


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "text-embedding-3-small"
    timeout_s: float = 30.0


class EmbeddingService:
    """
    Embedding wrapper used by RAG:

    - If OPENAI_API_KEY is set, calls OpenAI Embeddings API.
    - Otherwise, falls back to a deterministic local embedding.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        if config is None:
            config = EmbeddingConfig(
                provider=os.getenv("EMBED_PROVIDER", "openai").lower(),
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                openai_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
                timeout_s=float(os.getenv("EMBED_TIMEOUT_S", os.getenv("LLM_TIMEOUT_S", "30"))),
            )
        self.config = config

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self.config.provider == "openai" and self.config.openai_api_key:
            return self._embed_openai(texts)
        return [self._embed_local(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def _embed_openai(self, texts: Sequence[str]) -> List[List[float]]:
        url = f"{self.config.openai_base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.openai_model, "input": list(texts)}
        with httpx.Client(timeout=self.config.timeout_s) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        try:
            # Expected: data["data"] = [{"embedding": [...], "index": 0}, ...]
            embeddings = [item["embedding"] for item in data["data"]]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unexpected embeddings response shape: {data}") from exc

        return embeddings

    def _embed_local(self, text: str, dimension: int = 384) -> List[float]:
        vec = [0.0] * dimension
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vec

        for token in tokens:
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(token_hash[:8], 16) % dimension
            vec[bucket] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]

