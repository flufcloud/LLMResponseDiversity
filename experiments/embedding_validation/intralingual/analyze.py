from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.embedding_validation.intralingual.embed_data import (
    DATA_DIR,
    EXPERIMENT_DIR,
    EXPERIMENT_ID,
    RESULTS_DIR,
    prepare_embedding_inputs,
)
from lib.embed_locally import (
    get_local_embedding_fingerprint,
    load_local_embedding_settings,
)
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
    settings = load_local_embedding_settings(Path(args.config))
    translated = pd.read_csv(args.input)
    inputs = prepare_embedding_inputs(translated)
    model_dir = RESULTS_DIR / "models" / settings.result_name
    embeddings_path = (
        Path(args.embeddings)
        if args.embeddings
        else model_dir / "data/embeddings.sqlite"
    )
    fingerprint = get_local_embedding_fingerprint(settings)
    with EmbeddingCache(embeddings_path) as cache:
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
    tables_dir = Path(args.tables_dir) if args.tables_dir else model_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": tables_dir / "intralingual_embedding_scores_raw.csv",
        "mean": tables_dir / "intralingual_embedding_scores_mean.csv",
        "query": tables_dir / "intralingual_embedding_scores_mean_by_query.csv",
    }
    raw.to_csv(outputs["raw"], index=False)
    mean.to_csv(outputs["mean"], index=False)
    by_query.to_csv(outputs["query"], index=False)
    metadata_path = (
        Path(args.metadata) if args.metadata else model_dir / "metadata.json"
    )
    write_experiment_metadata(metadata_path, EXPERIMENT_ID, settings.model)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze translated MRPC embeddings")
    parser.add_argument(
        "--config", default=str(EXPERIMENT_DIR / "embedding_config_bge_m3.json")
    )
    parser.add_argument("--input", default=str(DATA_DIR / "mrpc_translated.csv"))
    parser.add_argument("--embeddings")
    parser.add_argument("--tables-dir")
    parser.add_argument("--metadata")
    return parser.parse_args()


if __name__ == "__main__":
    for name, path in analyze_embeddings(parse_args()).items():
        print(f"{name}: {path}")
