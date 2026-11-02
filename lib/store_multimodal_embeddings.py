from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class MultimodalEmbeddingStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                input_id TEXT NOT NULL,
                input_kind TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (input_id, input_kind, model)
            )
            """
        )

    def close(self) -> None:
        self.connection.close()

    def get_many(
        self,
        input_kind: str,
        model: str,
        expected_hashes: dict[str, str],
    ) -> dict[str, np.ndarray]:
        found = {}
        input_ids = list(expected_hashes)
        for start in range(0, len(input_ids), 900):
            chunk = input_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                "SELECT input_id, content_sha256, dimensions, embedding "
                "FROM embeddings WHERE input_kind = ? AND model = ? "
                f"AND input_id IN ({placeholders})",
                [input_kind, model, *chunk],
            )
            for input_id, content_hash, dimensions, blob in rows:
                if content_hash != expected_hashes[input_id]:
                    continue
                vector = np.frombuffer(blob, dtype="<f4").copy()
                if vector.size != dimensions:
                    raise ValueError(f"Corrupt embedding: {input_id}")
                found[input_id] = vector
        return found

    def put_many(
        self,
        input_kind: str,
        model: str,
        records: list[tuple[str, str, np.ndarray]],
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                input_id,
                input_kind,
                content_hash,
                model,
                vector.size,
                vector.astype("<f4").tobytes(),
                created_at,
            )
            for input_id, content_hash, vector in records
        ]
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (input_id, input_kind, content_sha256, model, dimensions, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()

    def __enter__(self) -> MultimodalEmbeddingStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
