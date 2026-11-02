from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from lib.plot_word_frequencies import plot_top_words, select_top_words


DEFAULT_CONFIG = Path(__file__).with_name("basic_lexical_browser_config.json")
THINKING_LEVELS = ("minimal", "high", "combined")


def load_usage_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    paths = {
        **config["thinking_tables"],
        "combined": config["combined_table"],
    }
    tables = {}
    for thinking, path in tqdm(paths.items(), desc="Loading tables", unit="table"):
        tables[thinking] = pd.read_csv(path)
    return tables


def get_topics(tables: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(tables["combined"]["topic"].unique())


def select_topic_words(
    tables: dict[str, pd.DataFrame],
    topic: str,
    thinking: str,
    responses_per_topic: int,
) -> pd.DataFrame:
    table = tables[thinking]
    count_column = (
        "num_usages_within_topic_generations"
        if thinking == "combined"
        else "num_usages"
    )
    selected = table.loc[
        table["topic"] == topic,
        ["word", count_column],
    ].rename(columns={"word": "lemma", count_column: "count"})
    condition_count = len(tables) - 1 if thinking == "combined" else 1
    denominator = responses_per_topic * condition_count
    selected["decimal_frequency"] = (selected["count"] / denominator).round(2)
    return selected.reset_index(drop=True)


class BasicLexicalBrowser:
    def __init__(
        self,
        root: Any,
        tables: dict[str, pd.DataFrame],
        responses_per_topic: int,
        topic: str,
        thinking: str,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.root = root
        self.tables = tables
        self.responses_per_topic = responses_per_topic
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.figure = None
        self.canvas = None

        root.title("Basic poetry words")
        root.minsize(700, 600)
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill=tk.X)
        self.content = ttk.Frame(root, padding=(8, 0, 8, 8))
        self.content.pack(fill=tk.BOTH, expand=True)

        self.topic = tk.StringVar(value=topic)
        self.thinking = tk.StringVar(value=thinking.title())
        fields = [
            ("Topic", self.topic, get_topics(tables), 28),
            ("Thinking", self.thinking, [item.title() for item in THINKING_LEVELS], 14),
        ]
        for column, (label, variable, values, width) in enumerate(fields):
            ttk.Label(controls, text=label).grid(
                row=0,
                column=column,
                padx=(0, 8),
                sticky="w",
            )
            selector = ttk.Combobox(
                controls,
                textvariable=variable,
                values=values,
                state="readonly",
                width=width,
            )
            selector.grid(row=1, column=column, padx=(0, 8), sticky="ew")
            selector.bind("<<ComboboxSelected>>", self.render)
        controls.columnconfigure(0, weight=1)
        self.render()

    def render(self, event: Any | None = None) -> None:
        del event
        thinking = self.thinking.get().lower()
        frequencies = select_topic_words(
            self.tables,
            self.topic.get(),
            thinking,
            self.responses_per_topic,
        )
        figure = plot_top_words(
            select_top_words(frequencies),
            f"Most common content lemmas\n{self.topic.get()} / {thinking}",
            count_label="Poems containing lemma",
        )
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            plt.close(self.figure)
        self.figure = figure
        self.canvas = self.FigureCanvasTkAgg(figure, master=self.content)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


def browse_basic_words(
    config_path: Path,
    topic: str,
    thinking: str = "combined",
) -> None:
    import tkinter as tk

    config = json.loads(config_path.read_text(encoding="utf-8"))
    generation_config = json.loads(
        Path(config["generation_config"]).read_text(encoding="utf-8")
    )
    tables = load_usage_tables(config)
    if topic not in get_topics(tables):
        raise ValueError(f"Unknown topic: {topic}")

    root = tk.Tk()
    BasicLexicalBrowser(
        root,
        tables,
        generation_config["responses_per_topic"],
        topic,
        thinking,
    )
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse basic-poetry lemmas")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--thinking", choices=THINKING_LEVELS, default="combined")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    browse_basic_words(args.config, args.topic, thinking=args.thinking)
