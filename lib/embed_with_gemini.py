from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

from lib.compute_identifiers import compute_stable_hash
from lib.store_embeddings import (
    EmbeddingCache,
    get_keyed_embedding_texts,
    resolve_response_embeddings,
)


def get_gemini_embedding_client(client: Any | None = None) -> Any:
    return client or genai.Client()


def get_gemini_cache_model(model: str, task_prefix: str) -> str:
    prefix_id = compute_stable_hash(task_prefix)[:12]
    return f"{model}|task-prefix-{prefix_id}"


def fetch_gemini_embeddings(
    responses: pd.DataFrame,
    cache: EmbeddingCache,
    client: Any | None,
    model: str,
    dimensions: int,
    task_prefix: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    cache_model = get_gemini_cache_model(model, task_prefix)
    keyed_texts = get_keyed_embedding_texts(responses, cache_model, dimensions)
    cached = cache.get_many([key for key, _, _ in keyed_texts])
    missing = [record for record in keyed_texts if record[0] not in cached]
    print(f"Embedding database: {len(cached)} hits, {len(missing)} misses")
    if missing and client is None:
        raise PermissionError("Embedding API access requires the explicit --confirm-api flag")

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=task_prefix.format(content=text))],
            )
            for _, _, text in batch
        ]
        result = client.models.embed_content(  # type: ignore[union-attr]
            model=model,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=dimensions),
        )
        vectors = [
            np.asarray(embedding.values, dtype=np.float32)
            for embedding in result.embeddings
        ]
        if len(vectors) != len(batch) or any(vector.size != dimensions for vector in vectors):
            raise ValueError("Embedding API returned an unexpected batch shape")
        records = [
            (key, text_hash, cache_model, dimensions, vector)
            for (key, text_hash, _), vector in zip(batch, vectors, strict=True)
        ]
        cache.put_many(records)
        cached.update({key: vector for key, _, _, _, vector in records})
        print(f"Embedded {min(start + batch_size, len(missing))}/{len(missing)}")

    return resolve_response_embeddings(
        responses, cache, cached, cache_model, dimensions
    )
