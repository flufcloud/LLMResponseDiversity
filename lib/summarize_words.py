from __future__ import annotations

import math

import pandas as pd


def calculate_shannon_entropy(counts: pd.Series) -> float:
    total = counts.sum()
    probabilities = counts / total
    entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    return 0.0 if entropy == 0 else float(entropy)


def count_unique_lemmas(
    lemma_table: pd.DataFrame,
    group_columns: tuple[str, ...],
    lemma_column: str = "lemma",
    count_column: str = "count",
) -> pd.DataFrame:
    unique_counts = (
        lemma_table.groupby(list(group_columns), as_index=False)[lemma_column]
        .nunique()
        .rename(columns={lemma_column: "num_unique_lemmas"})
    )
    entropies = (
        lemma_table.groupby(list(group_columns))[count_column]
        .apply(calculate_shannon_entropy)
        .rename("entropy")
        .reset_index()
    )
    return unique_counts.merge(entropies, on=list(group_columns)).sort_values(
        list(group_columns),
        ignore_index=True,
    )
