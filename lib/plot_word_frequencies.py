from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def select_top_words(frequencies: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    top_words = frequencies.sort_values(
        ["count", "decimal_frequency", "lemma"],
        ascending=[False, False, True],
        ignore_index=True,
    ).head(limit)
    top_words.insert(0, "rank", range(1, len(top_words) + 1))
    return top_words[["rank", "lemma", "count", "decimal_frequency"]]


def plot_top_words(
    top_words: pd.DataFrame,
    title: str,
    count_label: str = "Total occurrences",
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5.2))
    axis.axis("off")
    axis.set_title(title, pad=14)
    rows = [
        [
            int(row.rank),
            row.lemma,
            int(row.count),
            f"{row.decimal_frequency:.2f}",
        ]
        for row in top_words.itertuples(index=False)
    ]
    table = axis.table(
        cellText=rows,
        colLabels=["Rank", "Lemma", count_label, "Poem frequency"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for column in range(4):
        table[(0, column)].set_facecolor("#D9E6F2")
        table[(0, column)].set_text_props(weight="bold")
    figure.tight_layout()
    return figure
