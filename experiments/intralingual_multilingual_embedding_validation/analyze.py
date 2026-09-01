from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.intralingual_multilingual_embedding_validation.embed_data import (
    DATA_DIR,
    EXPERIMENT_DIR,
    EXPERIMENT_ID,
    METADATA_PATH,
    TABLES_DIR,
    prepare_embedding_inputs,
)
from lib.embed_with_bge import get_bge_fingerprint, load_bge_settings
from lib.store_embeddings import EmbeddingCache, load_response_embeddings
from lib.write_experiment_metadata import write_experiment_metadata


def compute_intralingual_scores(
    translated: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for row in translated.itertuples(index=False):
        base = embeddings[f"mrpc_{row.index}_base"]
        paraphrase = embeddings[f"mrpc_{row.index}_paraphrase"]
        similarity = base @ paraphrase
        similarity /= np.linalg.norm(base) * np.linalg.norm(paraphrase)
        records.append(
            {
                "index": row.index,
                "language_identifier": row.target_lang_identifier,
                "similarity_score": float(similarity),
            }
        )
    return pd.DataFrame(records)


def analyze_embeddings(args: argparse.Namespace) -> dict[str, Path]:
    settings = load_bge_settings(Path(args.config))
    translated = pd.read_csv(args.input)
    inputs = prepare_embedding_inputs(translated)
    fingerprint = get_bge_fingerprint(
        settings.model,
        settings.revision,
        settings.max_length,
        settings.normalize_embeddings,
    )
    with EmbeddingCache(Path(args.embeddings)) as cache:
        embeddings = load_response_embeddings(
            inputs,
            cache,
            fingerprint,
            settings.dimensions,
        )
    raw = compute_intralingual_scores(translated, embeddings)
    mean = raw.groupby("language_identifier", as_index=False, sort=False)[
        "similarity_score"
    ].mean().rename(columns={"similarity_score": "mean_similarity_score"})
    by_query = (
        raw.merge(translated[["index", "mrpc_sentence_id"]], on="index")
        .groupby("mrpc_sentence_id", as_index=False, sort=False)["similarity_score"]
        .mean()
        .rename(columns={"similarity_score": "mean_similarity_score"})
    )
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": tables_dir / "intralingual_embedding_scores_raw.csv",
        "mean": tables_dir / "intralingual_embedding_scores_mean.csv",
        "query": tables_dir / "intralingual_embedding_scores_mean_by_query.csv",
    }
    raw.to_csv(outputs["raw"], index=False)
    mean.to_csv(outputs["mean"], index=False)
    by_query.to_csv(outputs["query"], index=False)
    write_experiment_metadata(Path(args.metadata), EXPERIMENT_ID, settings.model)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze translated MRPC embeddings")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "embedding_config.json"))
    parser.add_argument("--input", default=str(DATA_DIR / "mrpc_translated.csv"))
    parser.add_argument("--embeddings", default=str(DATA_DIR / "embeddings.sqlite"))
    parser.add_argument("--tables-dir", default=str(TABLES_DIR))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    for name, path in analyze_embeddings(parse_args()).items():
        print(f"{name}: {path}")
