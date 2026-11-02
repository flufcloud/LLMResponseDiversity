from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.multimodal_thinking_experiment.analyze_words import (
    extract_content_terms,
    load_spacy_model,
)
from experiments.multimodal_thinking_experiment.prepare_poems import (
    load_embedding_responses,
)
from lib.compute_similarity import normalize_matrix
from lib.embed_with_gemini import get_gemini_cache_model
from lib.measure_syntactic_similarity import (
    compute_poem_syntactic_similarity,
    load_stanza_pipeline,
    parse_poem_line_syntax,
)
from lib.store_embeddings import EmbeddingCache, load_response_embeddings


DEFAULT_CONFIG = Path(__file__).with_name("homogeneity_analysis_config.json")
METRIC_COLUMNS = (
    "cosine_similarity",
    "lexical_jaccard",
    "syntactic_similarity",
)


def sample_random_pairs(
    num_poems: int,
    num_pairs: int,
    random_seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(random_seed)
    pairs = set()
    while len(pairs) < num_pairs:
        left, right = sorted(rng.choice(num_poems, size=2, replace=False))
        pairs.add((int(left), int(right)))
    return sorted(pairs)


def get_within_topic_pairs(responses: pd.DataFrame) -> list[tuple[int, int]]:
    pairs = []
    for _, group in responses.groupby("topic", sort=True):
        pairs.extend(itertools.combinations(group.index, 2))
    return pairs


def compute_lexical_jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def compute_pair_scores(
    responses: pd.DataFrame,
    pairs: list[tuple[int, int]],
    embedding_matrix: np.ndarray,
    lemma_sets: list[set[str]],
    syntax_features: list[list[Any]],
    description: str,
) -> pd.DataFrame:
    records = []
    for left, right in tqdm(pairs, desc=description, unit="pair"):
        left_row = responses.iloc[left]
        right_row = responses.iloc[right]
        syntactic_similarity = compute_poem_syntactic_similarity(
            syntax_features[left], syntax_features[right]
        )
        records.append(
            {
                "poem_id_1": left_row["poem_id"],
                "poem_id_2": right_row["poem_id"],
                "topic_1": left_row["topic"],
                "topic_2": right_row["topic"],
                "thinking_level_1": left_row["thinking_level"],
                "thinking_level_2": right_row["thinking_level"],
                "cosine_similarity": float(
                    embedding_matrix[left] @ embedding_matrix[right]
                ),
                "lexical_jaccard": compute_lexical_jaccard(
                    lemma_sets[left], lemma_sets[right]
                ),
                "syntactic_similarity": syntactic_similarity,
            }
        )
    return pd.DataFrame(records)


def summarize_topic_homogeneity(pair_scores: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "n_pairs": ("poem_id_1", "size"),
        **{f"mean_{column}": (column, "mean") for column in METRIC_COLUMNS},
    }
    summary = (
        pair_scores.groupby("topic_1", sort=True)
        .agg(**aggregations)
        .reset_index()
        .rename(columns={"topic_1": "topic"})
    )
    return summary


def summarize_corpus_homogeneity(pair_scores: pd.DataFrame) -> pd.DataFrame:
    record = {"n_pairs": len(pair_scores)}
    record.update(
        {f"mean_{column}": pair_scores[column].mean() for column in METRIC_COLUMNS}
    )
    return pd.DataFrame([record])


def compare_homogeneity(
    corpus_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for column in METRIC_COLUMNS:
        metric = f"mean_{column}"
        corpus_mean = corpus_summary.loc[0, metric]
        topic_mean = topic_summary[metric].mean()
        records.append(
            {
                "metric": column,
                "corpus_random_pair_mean": corpus_mean,
                "mean_topic_homogeneity": topic_mean,
            }
        )
    return pd.DataFrame(records)


def load_embeddings(
    responses: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    embedding_config = json.loads(
        Path(config["embedding_config"]).read_text(encoding="utf-8")
    )["embedding"]
    cache_model = get_gemini_cache_model(
        embedding_config["model"], embedding_config["task_prefix"]
    )
    with EmbeddingCache(Path(config["embeddings_sqlite"])) as cache:
        return load_response_embeddings(
            responses,
            cache,
            cache_model,
            embedding_config["output_dimensionality"],
        )


def analyze_homogeneity(
    config_path: Path,
    *,
    spacy_nlp: Any | None = None,
    stanza_nlp: Any | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    responses = load_embedding_responses(Path(config["responses_csv"])).reset_index(
        drop=True
    )
    lexical_config = json.loads(
        Path(config["lexical_config"]).read_text(encoding="utf-8")
    )
    spacy_nlp = spacy_nlp or load_spacy_model(lexical_config)
    _, _, lemma_terms = extract_content_terms(
        responses["response"], spacy_nlp, lexical_config
    )
    lemma_sets = [set(terms) for terms in lemma_terms]

    syntax_config = config["syntax"]
    stanza_nlp = stanza_nlp or load_stanza_pipeline(syntax_config)
    syntax_features = parse_poem_line_syntax(
        responses["response"].tolist(),
        stanza_nlp,
        syntax_config["wl_iterations"],
    )
    embeddings = embeddings or load_embeddings(responses, config)
    embedding_matrix = normalize_matrix(responses, embeddings)

    random_pairs = sample_random_pairs(
        len(responses), config["corpus_random_pairs"], config["random_seed"]
    )
    within_pairs = get_within_topic_pairs(responses)
    random_scores = compute_pair_scores(
        responses,
        random_pairs,
        embedding_matrix,
        lemma_sets,
        syntax_features,
        "Scoring corpus pairs",
    )
    within_scores = compute_pair_scores(
        responses,
        within_pairs,
        embedding_matrix,
        lemma_sets,
        syntax_features,
        "Scoring within-topic pairs",
    )
    topic_summary = summarize_topic_homogeneity(within_scores)
    corpus_summary = summarize_corpus_homogeneity(random_scores)
    comparison = compare_homogeneity(corpus_summary, topic_summary)
    outputs = {
        "random_pair_scores": random_scores,
        "within_topic_pair_scores": within_scores,
        "corpus_homogeneity": corpus_summary,
        "topic_homogeneity": topic_summary,
        "homogeneity_comparison": comparison,
    }
    for name, table in outputs.items():
        output_path = Path(config["outputs"][name])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False, float_format="%.6f")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure corpus and within-topic poetry homogeneity"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    tables = analyze_homogeneity(parse_args().config)
    print(f"Wrote {len(tables)} homogeneity tables")
