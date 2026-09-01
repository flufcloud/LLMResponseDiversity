from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.store_embeddings import (
    EmbeddingCache,
    get_keyed_embedding_texts,
    resolve_response_embeddings,
)


BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_DIMENSIONS = 1024
BGE_MAX_LENGTH = 8192


@dataclass(frozen=True)
class BGEEmbeddingSettings:
    model: str
    revision: str
    dimensions: int
    normalize_embeddings: bool
    max_length: int
    batch_size: int
    device: str


def load_bge_settings(path: Path) -> BGEEmbeddingSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return BGEEmbeddingSettings(
        raw["model"],
        raw["revision"],
        raw["dimensions"],
        raw["normalize_embeddings"],
        raw["max_length"],
        raw["batch_size"],
        raw["device"],
    )


def get_bge_fingerprint(
    model: str = BGE_MODEL,
    revision: str = BGE_REVISION,
    max_length: int = BGE_MAX_LENGTH,
    normalize_embeddings: bool = True,
) -> str:
    normalization = "normalized" if normalize_embeddings else "unnormalized"
    return f"hf:{model}@{revision}:dense:float32:max{max_length}:{normalization}"


class BGEM3Encoder:
    def __init__(
        self,
        model: str = BGE_MODEL,
        revision: str = BGE_REVISION,
        device: str = "mps",
        max_length: int = BGE_MAX_LENGTH,
        normalize_embeddings: bool = True,
        allow_model_download: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install sentence-transformers with `uv sync`") from error

        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(
            model,
            revision=revision,
            device=device,
            trust_remote_code=False,
            local_files_only=not allow_model_download,
        )
        self.model.float()
        self.model.eval()
        self.model.max_seq_length = max_length

    def get_token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self.model.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )
        return [int(length) for length in encoded["length"]]

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


def fetch_bge_embeddings(
    responses: pd.DataFrame,
    cache: EmbeddingCache,
    encoder: Any,
    batch_size: int = 16,
    max_length: int = BGE_MAX_LENGTH,
    model: str = BGE_MODEL,
    revision: str = BGE_REVISION,
    dimensions: int = BGE_DIMENSIONS,
    normalize_embeddings: bool = True,
) -> tuple[dict[str, np.ndarray], str]:
    fingerprint = get_bge_fingerprint(
        model,
        revision,
        max_length,
        normalize_embeddings,
    )
    records = get_keyed_embedding_texts(responses, fingerprint, dimensions)
    cached = cache.get_many([key for key, _, _ in records])
    missing = [record for record in records if record[0] not in cached]
    print(f"BGE-M3 cache: {len(cached)} hits, {len(missing)} misses")

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        texts = [text for _, _, text in batch]
        lengths = encoder.get_token_lengths(texts)
        overlength = [length for length in lengths if length > max_length]
        if overlength:
            raise ValueError(
                f"BGE-M3 input exceeds {max_length} tokens; maximum observed: {max(overlength)}"
            )
        vectors = encoder.encode(texts, batch_size=len(texts))
        if vectors.shape != (len(batch), dimensions):
            raise ValueError(f"BGE-M3 returned unexpected shape: {vectors.shape}")
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("BGE-M3 returned invalid embeddings")
        cache_rows = [
            (key, text_hash, fingerprint, dimensions, vector)
            for (key, text_hash, _), vector in zip(batch, vectors, strict=True)
        ]
        cache.put_many(cache_rows)
        cached.update({key: vector for key, _, _, _, vector in cache_rows})
        print(f"Embedded {min(start + batch_size, len(missing))}/{len(missing)} missing responses")

    embeddings = resolve_response_embeddings(
        responses, cache, cached, fingerprint, dimensions
    )
    return embeddings, fingerprint
