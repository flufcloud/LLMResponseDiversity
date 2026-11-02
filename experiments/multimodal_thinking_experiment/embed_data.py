from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.multimodal_thinking_experiment.prepare_poems import (
    DEFAULT_CONFIG,
    DEFAULT_RESULTS_ROOT,
    EXPERIMENT_ID,
    load_analysis_config,
    load_embedding_responses,
)
from lib.embed_with_gemini import fetch_gemini_embeddings, get_gemini_embedding_client
from lib.store_embeddings import EmbeddingCache
from lib.write_experiment_metadata import write_experiment_metadata


def embed_method(args: argparse.Namespace, client: Any | None = None) -> int:
    config = load_analysis_config(Path(args.config))
    if args.method not in config["methods"]:
        raise ValueError(f"Unknown method: {args.method}")
    embedding = config["embedding"]
    if args.confirm_api:
        if not embedding["user_approved_for_api"]:
            raise PermissionError(
                f"The user must set user_approved_for_api to true in {args.config}"
            )
        client = get_gemini_embedding_client(client)

    results_root = Path(args.results_root)
    method_data = results_root / "data" / args.method
    responses = load_embedding_responses(method_data / "responses.csv")
    with EmbeddingCache(method_data / "embeddings.sqlite") as cache:
        fetch_gemini_embeddings(
            responses=responses,
            cache=cache,
            client=client,
            model=embedding["model"],
            dimensions=embedding["output_dimensionality"],
            task_prefix=embedding["task_prefix"],
            batch_size=embedding["batch_size"],
        )
    metadata_path = results_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    write_experiment_metadata(
        metadata_path,
        EXPERIMENT_ID,
        embedding["model"],
        max_token_length=metadata["max_token_length"],
    )
    return len(responses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed poems for one analysis method")
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur API charges",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = embed_method(arguments)
    print(f"Embedded {count} poems for {arguments.method}")
