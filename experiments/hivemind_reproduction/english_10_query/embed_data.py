from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.hivemind_reproduction.english_10_query.load_generation_records import (
    EMBEDDINGS_PATH,
    GENERATIONS_PATH,
    get_analysis_responses,
    load_generations,
    select_monolingual_queries,
)
from lib.prepare_responses import (
    normalize_responses,
)
from lib.store_embeddings import EmbeddingCache
from lib.embed_with_openai import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
    fetch_openai_embeddings,
    get_openai_client,
)


def prepare_responses(path: Path, config_id: str | None) -> pd.DataFrame:
    generations = load_generations(path)
    raw = select_monolingual_queries(get_analysis_responses(generations))
    normalized = normalize_responses(raw)
    if config_id:
        normalized = normalized[normalized["config_id"] == config_id].copy()
    return normalized


def embed_data(args: argparse.Namespace, client: Any | None = None) -> int:
    responses = prepare_responses(Path(args.input), args.config_id)
    if client is None and args.confirm_api:
        client = get_openai_client()
    with EmbeddingCache(Path(args.cache)) as cache:
        fetch_openai_embeddings(
            responses,
            cache,
            client,
            model=args.embedding_model,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
        )
    return len(responses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the 10-query English responses")
    parser.add_argument("--input", default=str(GENERATIONS_PATH))
    parser.add_argument("--cache", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--embedding-model", default=OPENAI_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=OPENAI_EMBEDDING_DIMENSIONS)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--config-id")
    parser.add_argument("--confirm-api", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    count = embed_data(parse_args())
    print(f"Embedded {count} responses")
