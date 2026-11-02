from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from experiments.templated_prompts_compositionality.plot_word_frequencies import (
    DEFAULT_CONFIG,
    format_selection,
    load_word_frequencies,
)
from lib.plot_word_frequencies import plot_top_words, select_top_words


def show_top_words(
    config_path: Path,
    topic: str,
    adjective_class: str | None = None,
    adjective: str | None = None,
    show_plot: bool = True,
) -> pd.DataFrame:
    progress = tqdm(total=3, desc="Showing top words", unit="stage")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress.update()
    frequencies, _ = load_word_frequencies(
        config,
        topic,
        adjective_class,
        adjective,
    )
    top_words = select_top_words(frequencies)
    progress.update()
    figure = plot_top_words(
        top_words,
        "Most common content lemmas\n"
        + format_selection(topic, adjective_class, adjective),
    )
    progress.update()
    progress.close()
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return top_words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the 10 most common content lemmas"
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
        view="table",
    )
