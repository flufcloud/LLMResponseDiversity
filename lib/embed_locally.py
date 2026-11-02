from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.compute_identifiers import compute_stable_hash
from lib.store_embeddings import (
    EmbeddingCache,
    get_keyed_embedding_texts,
    resolve_response_embeddings,
)


@dataclass(frozen=True)
class LocalEmbeddingSettings:
    result_name: str
    model: str
    revision: str
    dimensions: int
    normalize_embeddings: bool
    max_length: int
    batch_size: int
    checkpoint_batch_size: int
    device: str
    dtype: str = "float32"
    prompt_name: str | None = None
    prompt: str | None = None


def load_local_embedding_settings(path: Path) -> LocalEmbeddingSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return LocalEmbeddingSettings(
        raw["result_name"],
        raw["model"],
        raw["revision"],
        raw["dimensions"],
        raw["normalize_embeddings"],
        raw["max_length"],
        raw["batch_size"],
        raw.get("checkpoint_batch_size", raw["batch_size"] * 16),
        raw["device"],
        raw.get("dtype", "float32"),
        raw.get("prompt_name"),
        raw.get("prompt"),
    )


def get_local_embedding_fingerprint(settings: LocalEmbeddingSettings) -> str:
    normalization = "normalized" if settings.normalize_embeddings else "unnormalized"
    fingerprint = (
        f"hf:{settings.model}@{settings.revision}:dense:{settings.dtype}:"
        f"max{settings.max_length}:{normalization}"
    )
    if settings.prompt:
        fingerprint += f":prompt={compute_stable_hash(settings.prompt)}"
    return fingerprint


class LocalSentenceEncoder:
    def __init__(
        self,
        settings: LocalEmbeddingSettings,
        allow_model_download: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install sentence-transformers with `uv sync`"
            ) from error

        self.settings = settings
        self.model = SentenceTransformer(
            settings.model,
            revision=settings.revision,
            device=settings.device,
            trust_remote_code=False,
            local_files_only=not allow_model_download,
            model_kwargs={"dtype": settings.dtype},
        )
        self.model.eval()
        self.model.max_seq_length = settings.max_length

    def get_token_lengths(self, texts: list[str]) -> list[int]:
        if self.settings.prompt:
            texts = [self.settings.prompt + text for text in texts]
        encoded = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        return [int(length) for length in encoded["length"]]

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        arguments = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": self.settings.normalize_embeddings,
            "show_progress_bar": True,
        }
        if self.settings.prompt:
            arguments["prompt"] = self.settings.prompt
        return np.asarray(self.model.encode(texts, **arguments), dtype=np.float32)


def fetch_local_embeddings(
    responses: pd.DataFrame,
    cache: EmbeddingCache,
    encoder: Any,
    settings: LocalEmbeddingSettings,
) -> tuple[dict[str, np.ndarray], str]:
    fingerprint = get_local_embedding_fingerprint(settings)
    records = get_keyed_embedding_texts(responses, fingerprint, settings.dimensions)
    cached = cache.get_many([key for key, _, _ in records])
    missing = [record for record in records if record[0] not in cached]
    print(
        f"{settings.result_name} embeddings: "
        f"{len(cached)} hits, {len(missing)} misses"
    )

    for start in range(0, len(missing), settings.checkpoint_batch_size):
        batch = missing[start : start + settings.checkpoint_batch_size]
        texts = [text for _, _, text in batch]
        lengths = encoder.get_token_lengths(texts)
        overlength = [length for length in lengths if length > settings.max_length]
        if overlength:
            raise ValueError(
                f"Input exceeds {settings.max_length} tokens; maximum observed: "
                f"{max(overlength)}"
            )
        vectors = encoder.encode(texts, batch_size=settings.batch_size)
        if vectors.shape != (len(batch), settings.dimensions):
            raise ValueError(
                f"Embedding model returned unexpected shape: {vectors.shape}"
            )
        invalid_norm = np.any(np.linalg.norm(vectors, axis=1) == 0)
        if not np.isfinite(vectors).all() or invalid_norm:
            raise ValueError("Embedding model returned invalid embeddings")
        cache_rows = [
            (key, text_hash, fingerprint, settings.dimensions, vector)
            for (key, text_hash, _), vector in zip(batch, vectors, strict=True)
        ]
        cache.put_many(cache_rows)
        cached.update({key: vector for key, _, _, _, vector in cache_rows})
        completed = min(start + settings.checkpoint_batch_size, len(missing))
        print(f"Embedded {completed}/{len(missing)} missing texts")

    embeddings = resolve_response_embeddings(
        responses,
        cache,
        cached,
        fingerprint,
        settings.dimensions,
    )
    return embeddings, fingerprint
