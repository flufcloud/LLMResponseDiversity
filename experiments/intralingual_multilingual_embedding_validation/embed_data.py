from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from lib.compute_identifiers import compute_stable_hash
from lib.embed_with_bge import BGEM3Encoder, fetch_bge_embeddings, load_bge_settings
from lib.store_embeddings import EmbeddingCache


EXPERIMENT_ID = "intralingual_multilingual_embedding_validation"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_ID
RESULTS_DIR = Path("results/experiments") / EXPERIMENT_ID
DATA_DIR = RESULTS_DIR / "data"
TABLES_DIR = RESULTS_DIR / "tables"
METADATA_PATH = RESULTS_DIR / "metadata.json"


def prepare_embedding_inputs(translated: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in translated.itertuples(index=False):
        for sentence_kind in ("base", "paraphrase"):
            text = getattr(row, f"translated_{sentence_kind}")
            records.append(
                {
                    "response_id": f"mrpc_{row.index}_{sentence_kind}",
                    "response": text,
                    "response_sha256": compute_stable_hash(text),
                }
            )
    return pd.DataFrame(records)


def embed_mrpc(args: argparse.Namespace, encoder: Any | None = None) -> int:
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
    parser = argparse.ArgumentParser(description="Embed translated MRPC sentence pairs")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "embedding_config.json"))
    parser.add_argument("--input", default=str(DATA_DIR / "mrpc_translated.csv"))
    parser.add_argument("--embeddings", default=str(DATA_DIR / "embeddings.sqlite"))
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Embedded {embed_mrpc(parse_args())} MRPC sentences")
