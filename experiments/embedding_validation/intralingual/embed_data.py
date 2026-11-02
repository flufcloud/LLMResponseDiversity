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


EXPERIMENT_ID = "intralingual_multilingual_embedding_validation"
EXPERIMENT_DIR = Path("experiments/embedding_validation/intralingual")
RESULTS_DIR = Path("results/embedding_validation/intralingual")
DATA_DIR = RESULTS_DIR / "data"


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
    parser = argparse.ArgumentParser(description="Embed translated MRPC sentence pairs")
    parser.add_argument(
        "--config", default=str(EXPERIMENT_DIR / "embedding_config_bge_m3.json")
    )
    parser.add_argument("--input", default=str(DATA_DIR / "mrpc_translated.csv"))
    parser.add_argument("--embeddings")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Embedded {embed_mrpc(parse_args())} MRPC sentences")
