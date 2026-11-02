from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from tqdm import tqdm


DEFAULT_CONFIG = Path(__file__).with_name("lexical_analysis_config.json")


def get_level(
    adjective_class: str | None,
    adjective: str | None,
) -> tuple[str, tuple[str, ...]]:
    if adjective and not adjective_class:
        raise ValueError("--adjective requires --adjective-class")
    if adjective:
        return "topic and adjective", ("topic", "adjective_class", "adjective")
    if adjective_class:
        return "topic and adjective class", ("topic", "adjective_class")
    return "topic", ("topic",)


def load_word_frequencies(
    config: dict[str, Any],
    topic: str,
    adjective_class: str | None,
    adjective: str | None,
) -> tuple[pd.DataFrame, str]:
    level_name, group_columns = get_level(adjective_class, adjective)
    level = next(item for item in config["levels"] if item["name"] == level_name)
    frequencies = pd.read_csv(level["output_csv"])
    selections = {
        "topic": topic,
        "adjective_class": adjective_class,
        "adjective": adjective,
    }
    for column in group_columns:
        frequencies = frequencies[frequencies[column] == selections[column]]
    if frequencies.empty:
        selection = ", ".join(
            f"{column}={selections[column]}" for column in group_columns
        )
        raise ValueError(f"No lemma frequencies found for {selection}")
    return frequencies.reset_index(drop=True), level_name


def select_example_words(frequencies: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    common = frequencies.sort_values(
        ["count", "decimal_frequency", "lemma"],
        ascending=[False, False, True],
    ).iloc[0]
    remaining = frequencies[frequencies["lemma"] != common["lemma"]].copy()
    examples = [("Most used", common)]
    if not remaining.empty:
        median = frequencies["count"].median()
        remaining["median_distance"] = (remaining["count"] - median).abs()
        typical = remaining.sort_values(
            ["median_distance", "decimal_frequency", "lemma"],
            ascending=[True, False, True],
        ).iloc[0]
        examples.append(("Typical", typical))
        remaining = remaining[remaining["lemma"] != typical["lemma"]]
    if not remaining.empty:
        uncommon = remaining.sort_values(
            ["count", "decimal_frequency", "lemma"],
            ascending=[True, True, True],
        ).iloc[0]
        examples.append(("Uncommon", uncommon))
    return examples


def format_selection(
    topic: str,
    adjective_class: str | None,
    adjective: str | None,
) -> str:
    parts = [topic]
    if adjective_class:
        parts.append(adjective_class)
    if adjective:
        parts.append(adjective)
    return " / ".join(parts)


def plot_frequency_bars(
    frequencies: pd.DataFrame,
    topic: str,
    adjective_class: str | None,
    adjective: str | None,
) -> plt.Figure:
    ranked = frequencies.sort_values(
        ["count", "decimal_frequency", "lemma"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    examples = select_example_words(ranked)
    example_colors = {
        "Most used": "#E45756",
        "Typical": "#F2CF5B",
        "Uncommon": "#72B7B2",
    }
    lemma_colors = {
        row["lemma"]: example_colors[label] for label, row in examples
    }
    colors = [lemma_colors.get(lemma, "#4C78A8") for lemma in ranked["lemma"]]

    figure, axis = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(ranked))
    axis.bar(positions, ranked["count"], width=0.9, color=colors)
    axis.set_xticks([])
    axis.set_xlim(-0.5, len(ranked) - 0.5)
    axis.set_xlabel("Content lemmas ranked by total occurrences")
    axis.set_ylabel("Total occurrences")
    axis.set_title(
        "Content-lemma usage\n"
        + format_selection(topic, adjective_class, adjective)
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        handles=[
            Patch(
                color=example_colors[label],
                label=f"{label}: {row['lemma']} ({int(row['count'])})",
            )
            for label, row in examples
        ],
        title="Example lemmas",
    )
    figure.tight_layout()
    return figure


def plot_word_frequencies(
    config_path: Path,
    topic: str,
    adjective_class: str | None = None,
    adjective: str | None = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    progress = tqdm(total=3, desc="Plotting word frequencies", unit="stage")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress.update()
    frequencies, _ = load_word_frequencies(
        config,
        topic,
        adjective_class,
        adjective,
    )
    progress.update()
    figure = plot_frequency_bars(
        frequencies,
        topic,
        adjective_class,
        adjective,
    )
    progress.update()
    progress.close()
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return frequencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ranked content-lemma occurrence counts"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--adjective-class")
    parser.add_argument("--adjective")
    arguments = parser.parse_args()
    if arguments.adjective and not arguments.adjective_class:
        parser.error("--adjective requires --adjective-class")
    return arguments


if __name__ == "__main__":
    args = parse_args()
    from experiments.templated_prompts_compositionality.browse_word_frequencies import (
        browse_word_frequencies,
    )

    browse_word_frequencies(
        args.config,
        args.topic,
        adjective_class=args.adjective_class,
        adjective=args.adjective,
        view="plot",
    )
