from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from lib.compute_identifiers import compute_stable_hash
from lib.embed_locally import (
    LocalSentenceEncoder,
    fetch_local_embeddings,
    load_local_embedding_settings,
)
from lib.store_embeddings import EmbeddingCache


EXPERIMENT_DIR = Path("experiments/embedding_validation/cross_lingual")
EXPERIMENT_ID = "cross_lingual_embedding_validation"
RESULTS_DIR = Path("results/embedding_validation/cross_lingual")
DATA_DIR = RESULTS_DIR / "data"


def prepare_embedding_inputs(translated_queries: pd.DataFrame) -> pd.DataFrame:
    inputs = translated_queries.rename(columns={"query": "response"}).copy()
    inputs["response_id"] = inputs["query_id"] + "_" + inputs["lang"]
    inputs["response_sha256"] = inputs["response"].map(compute_stable_hash)
    return inputs[["response_id", "query_id", "lang", "response", "response_sha256"]]


def embed_queries(args: argparse.Namespace, encoder: Any | None = None) -> int:
    settings = load_local_embedding_settings(Path(args.config))
    inputs = prepare_embedding_inputs(pd.read_csv(args.input))
    encoder = encoder or LocalSentenceEncoder(
        settings,
        allow_model_download=args.allow_model_download,
    )
    embeddings_path = Path(args.embeddings) if args.embeddings else (
        RESULTS_DIR / "models" / settings.result_name / "data" / "embeddings.sqlite"
    )
    with EmbeddingCache(embeddings_path) as cache:
        fetch_local_embeddings(inputs, cache, encoder, settings)
    return len(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the translated Hivemind queries"
    )
    parser.add_argument(
        "--config", default=str(EXPERIMENT_DIR / "embedding_config_bge_m3.json")
    )
    parser.add_argument("--input", default=str(DATA_DIR / "translated_queries.csv"))
    parser.add_argument("--embeddings")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    count = embed_queries(parse_args())
    print(f"Embedded {count} translated queries")
