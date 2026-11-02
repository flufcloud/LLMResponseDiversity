from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .process_analysis_data import (
    DEFAULT_CONFIG,
    get_data_path,
    load_config,
    load_responses,
)
from lib.embed_with_gemini import fetch_gemini_embeddings, get_gemini_embedding_client
from lib.store_embeddings import EmbeddingCache


def embed_responses(
    config_path: Path, confirm_api: bool, client: Any | None = None
) -> int:
    config = load_config(config_path)
    embedding = config["embedding"]
    if not confirm_api:
        raise PermissionError("Embedding API access requires --confirm-api")
    if not embedding["user_approved_for_api"]:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {config_path}"
        )

    responses = load_responses(config)
    client = client or get_gemini_embedding_client()
    with EmbeddingCache(get_data_path(config, "embeddings.sqlite")) as cache:
        fetch_gemini_embeddings(
            responses=responses,
            cache=cache,
            client=client,
            model=embedding["model"],
            dimensions=embedding["output_dimensionality"],
            task_prefix=embedding["task_prefix"],
            batch_size=embedding["batch_size"],
        )
    return len(responses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed adjective-poetry responses")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur charges",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = embed_responses(arguments.config, arguments.confirm_api)
    print(f"Embedded {count} poems")
