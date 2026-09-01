from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from lib.compute_identifiers import compute_stable_hash
from lib.embed_with_bge import BGEM3Encoder, fetch_bge_embeddings, load_bge_settings
from lib.store_embeddings import EmbeddingCache


EXPERIMENT_DIR = Path("experiments/cross_lingual_embedding_validation")
EXPERIMENT_ID = "cross_lingual_embedding_validation"
RESULTS_DIR = Path("results/experiments") / EXPERIMENT_ID
DATA_DIR = RESULTS_DIR / "data"
METADATA_PATH = RESULTS_DIR / "metadata.json"


def prepare_embedding_inputs(translated_queries: pd.DataFrame) -> pd.DataFrame:
    inputs = translated_queries.rename(columns={"query": "response"}).copy()
    inputs["response_id"] = inputs["query_id"] + "_" + inputs["lang"]
    inputs["response_sha256"] = inputs["response"].map(compute_stable_hash)
    return inputs[["response_id", "query_id", "lang", "response", "response_sha256"]]


def embed_queries(args: argparse.Namespace, encoder: Any | None = None) -> int:
    settings = load_bge_settings(Path(args.config))
    inputs = prepare_embedding_inputs(pd.read_csv(args.input))
    encoder = encoder or BGEM3Encoder(
        model=settings.model,
        revision=settings.revision,
        device=settings.device,
        max_length=settings.max_length,
        normalize_embeddings=settings.normalize_embeddings,
        allow_model_download=args.allow_model_download,
    )
    with EmbeddingCache(Path(args.embeddings)) as cache:
        fetch_bge_embeddings(
            inputs,
            cache,
            encoder,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
            model=settings.model,
            revision=settings.revision,
            dimensions=settings.dimensions,
            normalize_embeddings=settings.normalize_embeddings,
        )
    return len(inputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the translated Hivemind queries")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "embedding_config.json"))
    parser.add_argument("--input", default=str(DATA_DIR / "translated_queries.csv"))
    parser.add_argument("--embeddings", default=str(DATA_DIR / "embeddings.sqlite"))
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    count = embed_queries(parse_args())
    print(f"Embedded {count} translated queries")
