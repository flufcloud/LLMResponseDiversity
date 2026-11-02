from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.multimodal_thinking_experiment.prepare_poems import (
    DEFAULT_CONFIG,
    DEFAULT_RESULTS_ROOT,
    load_analysis_config,
    load_embedding_responses,
)
from lib.embed_with_gemini import get_gemini_cache_model
from lib.store_embeddings import EmbeddingCache, load_response_embeddings


RHYME_SCHEMES = ("No end rhymes", "AABB", "ABAB")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
VOWEL_TAIL_PATTERN = re.compile(r"[aeiouy]+[^aeiouy]*$")


def split_stanzas(poem: str) -> list[list[str]]:
    return [
        [line.strip() for line in block.splitlines() if line.strip()]
        for block in re.split(r"\n\s*\n+", poem.strip())
        if block.strip()
    ]


def get_line_end_word(line: str) -> str:
    words = WORD_PATTERN.findall(line)
    return words[-1].lower().replace("’", "'") if words else ""


def get_rhyme_key(word: str) -> str:
    word = word.removesuffix("'s")
    if len(word) > 3 and word.endswith("e") and not word.endswith("le"):
        word = word[:-1]
    match = VOWEL_TAIL_PATTERN.search(word)
    return match.group(0) if match else word[-3:]


def lines_rhyme(left: str, right: str) -> bool:
    left_word = get_line_end_word(left)
    right_word = get_line_end_word(right)
    return bool(
        left_word
        and right_word
        and left_word != right_word
        and get_rhyme_key(left_word) == get_rhyme_key(right_word)
    )


def classify_rhyme_scheme(stanzas: list[list[str]]) -> str:
    counts = {"AABB": 0, "ABAB": 0}
    for lines in stanzas:
        if len(lines) != 4:
            continue
        aabb = (
            lines_rhyme(lines[0], lines[1])
            and lines_rhyme(lines[2], lines[3])
            and get_rhyme_key(get_line_end_word(lines[0]))
            != get_rhyme_key(get_line_end_word(lines[2]))
        )
        abab = (
            lines_rhyme(lines[0], lines[2])
            and lines_rhyme(lines[1], lines[3])
            and get_rhyme_key(get_line_end_word(lines[0]))
            != get_rhyme_key(get_line_end_word(lines[1]))
        )
        counts["AABB"] += int(aabb)
        counts["ABAB"] += int(abab)
    if not any(counts.values()):
        return "No end rhymes"
    return max(counts, key=counts.get)


def compute_poem_metrics(responses: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in responses.itertuples(index=False):
        stanzas = split_stanzas(row.response)
        stanza_lengths = [len(lines) for lines in stanzas]
        records.append(
            {
                "poem_id": row.poem_id,
                "generation_id": row.generation_id,
                "method_id": row.method_id,
                "topic_id": row.topic_id,
                "topic": row.topic,
                "thinking_level": row.thinking_level,
                "source_sample_idx": row.source_sample_idx,
                "response_idx": row.response_idx,
                "line_count": sum(stanza_lengths),
                "stanza_count": len(stanzas),
                "four_line_stanza_count": stanza_lengths.count(4),
                "non_four_line_stanza_count": sum(length != 4 for length in stanza_lengths),
                "rhyme_scheme": classify_rhyme_scheme(stanzas),
            }
        )
    return pd.DataFrame(records)


def normalize_embeddings(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    matrix = np.stack([embeddings[poem_id] for poem_id in responses["poem_id"]])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def compute_within_condition_similarity(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    for keys, group in responses.groupby(
        ["topic_id", "topic", "thinking_level"], sort=False
    ):
        matrix = normalize_embeddings(group, embeddings)
        for left, right in itertools.combinations(range(len(group)), 2):
            records.append(
                {
                    "topic_id": keys[0],
                    "topic": keys[1],
                    "thinking_level": keys[2],
                    "poem_id_1": group.iloc[left]["poem_id"],
                    "poem_id_2": group.iloc[right]["poem_id"],
                    "similarity": float(matrix[left] @ matrix[right]),
                }
            )
    return pd.DataFrame(records)


def compute_cross_thinking_similarity(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    grouped = responses.groupby(["topic_id", "topic"], sort=False)
    for (topic_id, topic), group in grouped:
        minimal = group[group["thinking_level"] == "minimal"].reset_index(drop=True)
        high = group[group["thinking_level"] == "high"].reset_index(drop=True)
        minimal_matrix = normalize_embeddings(minimal, embeddings)
        high_matrix = normalize_embeddings(high, embeddings)
        similarities = minimal_matrix @ high_matrix.T
        for left, right in itertools.product(range(len(minimal)), range(len(high))):
            records.append(
                {
                    "topic_id": topic_id,
                    "topic": topic,
                    "minimal_poem_id": minimal.iloc[left]["poem_id"],
                    "high_poem_id": high.iloc[right]["poem_id"],
                    "similarity": float(similarities[left, right]),
                }
            )
    return pd.DataFrame(records)


def summarize_similarity(
    within: pd.DataFrame,
    cross: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    within_summary = (
        within.groupby(["topic_id", "topic", "thinking_level"], sort=False)["similarity"]
        .agg(n_pairs="size", mean_similarity="mean", std_similarity="std")
        .reset_index()
    )
    within_summary["n_responses"] = np.rint(
        (1 + np.sqrt(1 + 8 * within_summary["n_pairs"])) / 2
    ).astype(int)
    cross_summary = (
        cross.groupby(["topic_id", "topic"], sort=False)["similarity"]
        .agg(n_pairs="size", mean_similarity="mean", std_similarity="std")
        .reset_index()
    )
    cross_counts = (
        cross.groupby(["topic_id", "topic"], sort=False)
        .agg(
            n_minimal_responses=("minimal_poem_id", "nunique"),
            n_high_responses=("high_poem_id", "nunique"),
        )
        .reset_index()
    )
    cross_summary = cross_summary.merge(cross_counts, on=["topic_id", "topic"])
    return within_summary, cross_summary


def create_rhyme_table(metrics: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(metrics["rhyme_scheme"], metrics["thinking_level"])
    return counts.reindex(index=RHYME_SCHEMES, columns=["minimal", "high"], fill_value=0)


def create_quatrain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for thinking_level, group in itertools.chain(
        metrics.groupby("thinking_level", sort=False), [("all", metrics)]
    ):
        four_line = int(group["four_line_stanza_count"].sum())
        other = int(group["non_four_line_stanza_count"].sum())
        records.extend(
            [
                {
                    "thinking_level": thinking_level,
                    "stanza_type": "Four lines",
                    "count": four_line,
                },
                {
                    "thinking_level": thinking_level,
                    "stanza_type": "Other length",
                    "count": other,
                },
            ]
        )
    return pd.DataFrame(records)


def load_embeddings(
    responses: pd.DataFrame,
    database_path: Path,
    embedding: dict[str, Any],
) -> dict[str, np.ndarray]:
    cache_model = get_gemini_cache_model(
        embedding["model"], embedding["task_prefix"]
    )
    with EmbeddingCache(database_path) as cache:
        return load_response_embeddings(
            responses,
            cache,
            cache_model,
            embedding["output_dimensionality"],
        )


def analyze_method(args: argparse.Namespace) -> None:
    config = load_analysis_config(Path(args.config))
    if args.method not in config["methods"]:
        raise ValueError(f"Unknown method: {args.method}")
    results_root = Path(args.results_root)
    data_dir = results_root / "data" / args.method
    tables_dir = results_root / "tables" / args.method
    tables_dir.mkdir(parents=True, exist_ok=True)

    responses = load_embedding_responses(data_dir / "responses.csv")
    embeddings = load_embeddings(
        responses, data_dir / "embeddings.sqlite", config["embedding"]
    )
    metrics = compute_poem_metrics(responses)
    pii_path = data_dir / "pii_judgments.csv"
    if pii_path.exists():
        pii = pd.read_csv(pii_path, usecols=["poem_id", "count"])
        metrics = metrics.merge(
            pii.rename(columns={"count": "pii_count"}), on="poem_id", how="left"
        )

    within = compute_within_condition_similarity(responses, embeddings)
    cross = compute_cross_thinking_similarity(responses, embeddings)
    within_summary, cross_summary = summarize_similarity(within, cross)

    metrics.to_csv(data_dir / "poem_metrics.csv", index=False)
    within.to_csv(data_dir / "within_condition_similarity.csv", index=False)
    cross.to_csv(data_dir / "cross_thinking_similarity.csv", index=False)
    within_summary.to_csv(
        tables_dir / "mean_pairwise_similarity_by_topic.csv", index=False
    )
    cross_summary.to_csv(
        tables_dir / "mean_cross_thinking_similarity_by_topic.csv", index=False
    )
    create_rhyme_table(metrics).to_csv(tables_dir / "rhyme_scheme_counts.csv")
    create_quatrain_table(metrics).to_csv(
        tables_dir / "quatrain_stanza_counts.csv", index=False
    )
    print(f"Wrote analysis for {args.method} under {data_dir} and {tables_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one poetry generation method")
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    analyze_method(parse_args())
