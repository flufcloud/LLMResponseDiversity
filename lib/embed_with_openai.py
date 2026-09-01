from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from lib.store_embeddings import (
    EmbeddingCache,
    get_keyed_embedding_texts,
    resolve_response_embeddings,
)


OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSIONS = 1536


def get_openai_client(client: Any | None = None) -> Any:
    return client or OpenAI()


def fetch_openai_embeddings(
    responses: pd.DataFrame,
    cache: EmbeddingCache,
    client: Any | None,
    model: str = OPENAI_EMBEDDING_MODEL,
    dimensions: int = OPENAI_EMBEDDING_DIMENSIONS,
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    keyed_texts = get_keyed_embedding_texts(responses, model, dimensions)
    cached = cache.get_many([key for key, _, _ in keyed_texts])
    missing = [record for record in keyed_texts if record[0] not in cached]
    print(f"Embedding cache: {len(cached)} hits, {len(missing)} misses")
    if missing and client is None:
        raise PermissionError("Embedding API access requires the explicit --confirm-api flag")

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        response = client.embeddings.create(  # type: ignore[union-attr]
            model=model,
            dimensions=dimensions,
            input=[text for _, _, text in batch],
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in ordered]
        if len(vectors) != len(batch) or any(vector.size != dimensions for vector in vectors):
            raise ValueError("Embedding API returned an unexpected batch shape")
        records = [
            (key, text_hash, model, dimensions, vector)
            for (key, text_hash, _), vector in zip(batch, vectors, strict=True)
        ]
        cache.put_many(records)
        cached.update({key: vector for key, _, _, _, vector in records})
        completed = min(start + batch_size, len(missing))
        print(f"Embedded {completed}/{len(missing)} missing responses")

    return resolve_response_embeddings(responses, cache, cached, model, dimensions)
