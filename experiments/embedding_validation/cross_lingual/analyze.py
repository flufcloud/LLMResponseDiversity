from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.embedding_validation.cross_lingual.embed_data import (
    DATA_DIR,
    EXPERIMENT_ID,
    EXPERIMENT_DIR,
    RESULTS_DIR,
    prepare_embedding_inputs,
)
from lib.embed_locally import (
    get_local_embedding_fingerprint,
    load_local_embedding_settings,
)
from lib.store_embeddings import EmbeddingCache, load_response_embeddings
from lib.write_experiment_metadata import write_experiment_metadata


def compute_pairwise_similarities(
    inputs: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for query_id, group in inputs.groupby("query_id", sort=False):
        rows = list(group.itertuples(index=False))
        for left, right in combinations(rows, 2):
            left_vector = embeddings[left.response_id]
            right_vector = embeddings[right.response_id]
            score = left_vector @ right_vector
            score /= np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
            records.append(
                {
                    "query_id": query_id,
                    "lang_pair": "_".join(sorted((left.lang, right.lang))),
                    "similarity_score": float(score),
                }
            )
    return pd.DataFrame(records)


def summarize_similarities(
    raw: pd.DataFrame,
    queries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_query = raw.groupby("query_id", as_index=False)["similarity_score"].mean()
    by_query["query"] = by_query["query_id"].map(queries.set_index("query_id")["query"])
    by_query = by_query.rename(
        columns={"similarity_score": "mean_pairwise_similarity"}
    )[["query", "mean_pairwise_similarity"]].sort_values(
        "mean_pairwise_similarity", ascending=False, ignore_index=True
    )

    by_pair = raw.groupby("lang_pair", as_index=False)["similarity_score"].mean()
    by_pair = by_pair.rename(
        columns={"similarity_score": "mean_pairwise_similarity"}
    ).sort_values("mean_pairwise_similarity", ascending=False, ignore_index=True)

    language_pairs = by_pair["lang_pair"].str.split("_", expand=True)
    by_language = pd.concat(
        [
            pd.DataFrame(
                {
                    "lang": language_pairs[column],
                    "mean_pairwise_similarity": by_pair["mean_pairwise_similarity"],
                }
            )
            for column in language_pairs
        ],
        ignore_index=True,
    )
    by_language = (
        by_language.groupby("lang", as_index=False)["mean_pairwise_similarity"]
        .mean()
        .sort_values("mean_pairwise_similarity", ascending=False, ignore_index=True)
    )
    return by_query, by_pair, by_language


def analyze_embeddings(args: argparse.Namespace) -> dict[str, Path]:
    settings = load_local_embedding_settings(Path(args.config))
    translated_queries = pd.read_csv(args.input)
    inputs = prepare_embedding_inputs(translated_queries)
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
    raw = compute_pairwise_similarities(inputs, embeddings)
    source_queries = translated_queries.loc[
        translated_queries["lang"] == "en", ["query_id", "query"]
    ]
    summaries = summarize_similarities(raw, source_queries)
    data_dir = Path(args.data_dir) if args.data_dir else model_dir / "data"
    tables_dir = Path(args.tables_dir) if args.tables_dir else model_dir / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw": data_dir / "raw_multilingual_similarity.csv",
        "query": tables_dir / "cross_lingual_mean_similarity_by_query.csv",
        "lang_pair": tables_dir / "cross_lingual_mean_similarity_by_lang_pair.csv",
        "lang": tables_dir / "cross_lingual_mean_similarity_by_lang.csv",
    }
    for frame, path in zip((raw, *summaries), outputs.values(), strict=True):
        frame.to_csv(path, index=False)
    write_experiment_metadata(
        Path(args.metadata) if args.metadata else model_dir / "metadata.json",
        EXPERIMENT_ID,
        settings.model,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cross-lingual query embeddings"
    )
    parser.add_argument(
        "--config", default=str(EXPERIMENT_DIR / "embedding_config_bge_m3.json")
    )
    parser.add_argument("--input", default=str(DATA_DIR / "translated_queries.csv"))
    parser.add_argument("--embeddings")
    parser.add_argument("--data-dir")
    parser.add_argument("--tables-dir")
    parser.add_argument("--metadata")
    return parser.parse_args()


if __name__ == "__main__":
    for name, path in analyze_embeddings(parse_args()).items():
        print(f"{name}: {path}")
