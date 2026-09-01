from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def compute_embedding_key(text: str, model: str, dimensions: int) -> str:
    payload = json.dumps(
        [model, dimensions, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get_keyed_embedding_texts(
    responses: pd.DataFrame,
    model: str,
    dimensions: int,
) -> list[tuple[str, str, str]]:
    unique = responses[["response_sha256", "response"]].drop_duplicates("response_sha256")
    return [
        (
            compute_embedding_key(row.response, model, dimensions),
            row.response_sha256,
            row.response,
        )
        for row in unique.itertuples(index=False)
    ]


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                text_sha256 TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_embeddings (
                generation_id TEXT NOT NULL,
                input_kind TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                PRIMARY KEY (generation_id, input_kind, cache_key),
                FOREIGN KEY (cache_key) REFERENCES embeddings(cache_key)
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS embedding_text_idx ON embeddings(text_sha256)"
        )

    def close(self) -> None:
        self.connection.close()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 900):
            chunk = keys[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT cache_key, dimensions, embedding FROM embeddings "
                f"WHERE cache_key IN ({placeholders})"
            )
            rows = self.connection.execute(
                query,
                chunk,
            )
            for cache_key, dimensions, blob in rows:
                vector = np.frombuffer(blob, dtype="<f4").copy()
                if vector.size != dimensions:
                    raise ValueError(f"Corrupt cached embedding: {cache_key}")
                found[cache_key] = vector
        return found

    def put_many(
        self,
        records: list[tuple[str, str, str, int, np.ndarray]],
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                key,
                text_hash,
                model,
                dimensions,
                vector.astype("<f4").tobytes(),
                created_at,
            )
            for key, text_hash, model, dimensions, vector in records
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (cache_key, text_sha256, model, dimensions, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

    def link_many(self, rows: list[tuple[str, str, str]]) -> None:
        self.connection.executemany(
            """
            INSERT OR IGNORE INTO generation_embeddings
            (generation_id, input_kind, cache_key)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def resolve_response_embeddings(
    responses: pd.DataFrame,
    cache: EmbeddingCache,
    cached: dict[str, np.ndarray],
    model: str,
    dimensions: int,
) -> dict[str, np.ndarray]:
    if "generation_id" in responses:
        input_kinds = responses.get("response_input_kind", "model_response")
        if isinstance(input_kinds, str):
            input_kinds = pd.Series(input_kinds, index=responses.index)
        cache.link_many(
            [
                (
                    generation_id,
                    input_kind,
                    compute_embedding_key(text, model, dimensions),
                )
                for generation_id, input_kind, text in zip(
                    responses["generation_id"],
                    input_kinds,
                    responses["response"],
                    strict=True,
                )
            ]
        )
    return {
        row.response_id: cached[compute_embedding_key(row.response, model, dimensions)]
        for row in responses.itertuples(index=False)
    }


def load_response_embeddings(
    responses: pd.DataFrame,
    cache: EmbeddingCache,
    model: str,
    dimensions: int,
) -> dict[str, np.ndarray]:
    keyed_texts = get_keyed_embedding_texts(responses, model, dimensions)
    cached = cache.get_many([key for key, _, _ in keyed_texts])
    missing = [key for key, _, _ in keyed_texts if key not in cached]
    if missing:
        raise ValueError(
            f"Embedding database is missing {len(missing)} response embeddings; "
            "run embed_data.py first"
        )
    return resolve_response_embeddings(responses, cache, cached, model, dimensions)
