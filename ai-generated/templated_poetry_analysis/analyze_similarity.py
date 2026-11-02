from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.compute_similarity import normalize_matrix
from lib.embed_with_gemini import get_gemini_cache_model
from lib.measure_syntactic_similarity import compute_poem_syntactic_similarity
from lib.store_embeddings import EmbeddingCache, load_response_embeddings

from .process_analysis_data import (
    DEFAULT_CONFIG,
    GROUP_LEVELS,
    METRICS,
    get_data_path,
    get_group_seed,
    get_table_path,
    load_config,
    load_responses,
    load_text_features,
)


def sample_pairs(
    indices: Iterable[int], maximum_pairs: int, seed: int
) -> list[tuple[int, int]]:
    values = tuple(int(index) for index in indices)
    total_pairs = len(values) * (len(values) - 1) // 2
    if total_pairs <= maximum_pairs:
        return list(itertools.combinations(values, 2))

    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < maximum_pairs:
        left, right = sorted(rng.choice(values, size=2, replace=False))
        pairs.add((int(left), int(right)))
    return sorted(pairs)


def compute_lexical_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def load_embeddings(
    responses: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    embedding = config["embedding"]
    cache_model = get_gemini_cache_model(
        embedding["model"], embedding["task_prefix"]
    )
    with EmbeddingCache(get_data_path(config, "embeddings.sqlite")) as cache:
        vectors = load_response_embeddings(
            responses,
            cache,
            cache_model,
            embedding["output_dimensionality"],
        )
    return normalize_matrix(responses, vectors)


def score_pairs(
    responses: pd.DataFrame,
    pairs: list[tuple[int, int]],
    embeddings: np.ndarray,
    lemma_sets: list[set[str]],
    syntax_features: list[list[Any]],
    group_values: dict[str, Any],
    progress: tqdm,
) -> list[dict[str, Any]]:
    records = []
    for left, right in pairs:
        records.append(
            {
                **group_values,
                "poem_id_1": responses.iloc[left]["poem_id"],
                "poem_id_2": responses.iloc[right]["poem_id"],
                "cosine_similarity": float(embeddings[left] @ embeddings[right]),
                "lexical_jaccard": compute_lexical_jaccard(
                    lemma_sets[left], lemma_sets[right]
                ),
                "syntactic_similarity": compute_poem_syntactic_similarity(
                    syntax_features[left], syntax_features[right]
                ),
            }
        )
        progress.update()
    return records


def summarize_scores(
    pair_scores: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    aggregations = {
        "n_pairs": ("poem_id_1", "size"),
        **{f"mean_{metric}": (metric, "mean") for metric in METRICS},
        **{f"std_{metric}": (metric, "std") for metric in METRICS},
    }
    if not group_columns:
        return pd.DataFrame(
            [
                {
                    "n_pairs": len(pair_scores),
                    **{
                        f"mean_{metric}": pair_scores[metric].mean()
                        for metric in METRICS
                    },
                    **{
                        f"std_{metric}": pair_scores[metric].std()
                        for metric in METRICS
                    },
                }
            ]
        )
    return (
        pair_scores.groupby(group_columns, sort=True)
        .agg(**aggregations)
        .reset_index()
    )


def get_group_pair_total(
    responses: pd.DataFrame, group_columns: list[str], maximum_pairs: int
) -> int:
    sizes = responses.groupby(group_columns, sort=False).size()
    return int(
        sum(min(size * (size - 1) // 2, maximum_pairs) for size in sizes)
    )


def analyze_level(
    level: str,
    responses: pd.DataFrame,
    embeddings: np.ndarray,
    lemma_sets: list[set[str]],
    syntax_features: list[list[Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampling = config["pair_sampling"]
    if level == "corpus":
        maximum_pairs = sampling["corpus_pairs"]
        pairs = sample_pairs(
            responses.index,
            maximum_pairs,
            sampling["random_seed"],
        )
        progress = tqdm(total=len(pairs), desc="Scoring corpus", unit="pair")
        records = score_pairs(
            responses,
            pairs,
            embeddings,
            lemma_sets,
            syntax_features,
            {},
            progress,
        )
        progress.close()
        scores = pd.DataFrame(records)
        return scores, summarize_scores(scores, [])

    group_columns = GROUP_LEVELS[level]
    maximum_pairs = sampling["maximum_pairs_per_group"]
    total = get_group_pair_total(responses, group_columns, maximum_pairs)
    progress = tqdm(total=total, desc=f"Scoring {level} groups", unit="pair")
    records = []
    for keys, group in responses.groupby(group_columns, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        group_values = dict(zip(group_columns, key_tuple, strict=True))
        pairs = sample_pairs(
            group.index,
            maximum_pairs,
            get_group_seed(sampling["random_seed"], level, key_tuple),
        )
        records.extend(
            score_pairs(
                responses,
                pairs,
                embeddings,
                lemma_sets,
                syntax_features,
                group_values,
                progress,
            )
        )
    progress.close()
    scores = pd.DataFrame(records)
    return scores, summarize_scores(scores, group_columns)


def create_comparison(summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for metric in METRICS:
        record = {
            "metric_family": "pair_similarity",
            "metric": metric,
            "unit": "similarity",
            "corpus_value": summaries["corpus"].loc[0, f"mean_{metric}"],
        }
        for level in GROUP_LEVELS:
            values = summaries[level][f"mean_{metric}"]
            record.update(
                {
                    f"mean_{level}_value": values.mean(),
                    f"{level}_std": values.std(),
                    f"{level}_min": values.min(),
                    f"{level}_max": values.max(),
                    f"n_{level}_groups": len(values),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def analyze_similarity(config_path: Path) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    responses = load_responses(config).reset_index(drop=True)
    lemma_sets, syntax_features = load_text_features(config, responses)
    embeddings = load_embeddings(responses, config)
    outputs = {}
    summaries = {}
    for level in ("corpus", *GROUP_LEVELS):
        scores, summary = analyze_level(
            level,
            responses,
            embeddings,
            lemma_sets,
            syntax_features,
            config,
        )
        outputs[f"{level}_pair_scores"] = scores
        summaries[level] = summary
        summary_path = get_table_path(config, f"similarity_summary_{level}.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False, float_format="%.6f")

    for name, table in outputs.items():
        path = get_data_path(config, f"{name}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6f")
    comparison = create_comparison(summaries)
    comparison.to_csv(
        get_table_path(config, "similarity_comparisons.csv"),
        index=False,
        float_format="%.6f",
    )
    return {**outputs, **summaries, "comparison": comparison}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze adjective-poetry similarity at four levels"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    tables = analyze_similarity(parse_args().config)
    print(f"Wrote {len(tables)} similarity tables")
