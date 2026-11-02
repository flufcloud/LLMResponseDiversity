from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.summarize_words import calculate_shannon_entropy


def group_poem_lemmas(
    generations: pd.DataFrame,
    lemma_terms: list[list[str]],
) -> dict[str, list[list[str]]]:
    grouped: dict[str, list[list[str]]] = {}
    for topic, terms in zip(generations["topic"], lemma_terms, strict=True):
        grouped.setdefault(topic, []).append(terms)
    return grouped


def calculate_lexical_metrics(poems: Iterable[list[str]]) -> tuple[int, float]:
    counts = Counter(lemma for poem in poems for lemma in poem)
    entropy = calculate_shannon_entropy(pd.Series(list(counts.values()), dtype=float))
    return len(counts), entropy


def resample_lexical_metrics(
    poems: list[list[str]],
    sample_size: int,
    num_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    metrics = np.empty((num_resamples, 2), dtype=float)
    for index in range(num_resamples):
        selected = rng.choice(len(poems), size=sample_size, replace=False)
        metrics[index] = calculate_lexical_metrics(poems[item] for item in selected)
    return metrics


def compare_matched_topic_metrics(
    basic_terms: dict[str, list[list[str]]],
    candidate_terms: dict[str, list[list[str]]],
    *,
    sample_size: int,
    num_resamples: int,
    interval_level: float,
    random_seed: int,
    candidate_name: str,
) -> pd.DataFrame:
    tail_probability = (1 - interval_level) / 2
    rng = np.random.default_rng(random_seed)
    records = []
    topics = sorted(basic_terms)
    for topic in tqdm(
        topics,
        desc=f"Comparing {candidate_name} diversity",
        unit="topic",
    ):
        basic_unique, basic_entropy = calculate_lexical_metrics(basic_terms[topic])
        candidate_metrics = resample_lexical_metrics(
            candidate_terms[topic],
            sample_size,
            num_resamples,
            rng,
        )
        unique_differences = candidate_metrics[:, 0] - basic_unique
        entropy_differences = candidate_metrics[:, 1] - basic_entropy
        records.append(
            {
                "topic": topic,
                "sample_size": sample_size,
                "num_resamples": num_resamples,
                "interval_level": interval_level,
                "basic_num_unique_lemmas": basic_unique,
                f"{candidate_name}_mean_num_unique_lemmas": candidate_metrics[
                    :, 0
                ].mean(),
                f"{candidate_name}_num_unique_lemmas_interval_lower": np.quantile(
                    candidate_metrics[:, 0], tail_probability
                ),
                f"{candidate_name}_num_unique_lemmas_interval_upper": np.quantile(
                    candidate_metrics[:, 0], 1 - tail_probability
                ),
                "mean_num_unique_lemmas_difference": unique_differences.mean(),
                "num_unique_lemmas_difference_interval_lower": np.quantile(
                    unique_differences, tail_probability
                ),
                "num_unique_lemmas_difference_interval_upper": np.quantile(
                    unique_differences, 1 - tail_probability
                ),
                "basic_entropy": basic_entropy,
                f"{candidate_name}_mean_entropy": candidate_metrics[:, 1].mean(),
                f"{candidate_name}_entropy_interval_lower": np.quantile(
                    candidate_metrics[:, 1], tail_probability
                ),
                f"{candidate_name}_entropy_interval_upper": np.quantile(
                    candidate_metrics[:, 1], 1 - tail_probability
                ),
                "mean_entropy_difference": entropy_differences.mean(),
                "entropy_difference_interval_lower": np.quantile(
                    entropy_differences, tail_probability
                ),
                "entropy_difference_interval_upper": np.quantile(
                    entropy_differences, 1 - tail_probability
                ),
            }
        )
    return pd.DataFrame(records)
