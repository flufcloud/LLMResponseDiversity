from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from experiments.templated_prompts_compositionality.plot_word_frequencies import (
    DEFAULT_CONFIG,
    plot_frequency_bars,
)
from experiments.templated_prompts_compositionality.show_top_words import format_selection
from lib.plot_word_frequencies import plot_top_words, select_top_words


ALL = "All"


def load_frequency_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables = {}
    progress = tqdm(config["levels"], desc="Loading tables", unit="table")
    for level in progress:
        tables[level["name"]] = pd.read_csv(level["output_csv"])
    return tables


def get_topics(tables: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(tables["topic"]["topic"].unique())


def get_adjective_classes(
    tables: dict[str, pd.DataFrame],
    topic: str,
) -> list[str]:
    frequencies = tables["topic and adjective class"]
    return sorted(frequencies.loc[frequencies["topic"] == topic, "adjective_class"].unique())


def get_adjectives(
    tables: dict[str, pd.DataFrame],
    topic: str,
    adjective_class: str,
) -> list[str]:
    frequencies = tables["topic and adjective"]
    selected = frequencies[
        (frequencies["topic"] == topic)
        & (frequencies["adjective_class"] == adjective_class)
    ]
    return sorted(selected["adjective"].unique())


def select_frequencies(
    tables: dict[str, pd.DataFrame],
    topic: str,
    adjective_class: str | None,
    adjective: str | None,
) -> pd.DataFrame:
    if adjective:
        frequencies = tables["topic and adjective"]
        mask = (
            (frequencies["topic"] == topic)
            & (frequencies["adjective_class"] == adjective_class)
            & (frequencies["adjective"] == adjective)
        )
    elif adjective_class:
        frequencies = tables["topic and adjective class"]
        mask = (frequencies["topic"] == topic) & (
            frequencies["adjective_class"] == adjective_class
        )
    else:
        frequencies = tables["topic"]
        mask = frequencies["topic"] == topic
    return frequencies.loc[mask].reset_index(drop=True)


class WordFrequencyBrowser:
    def __init__(
        self,
        root: Any,
        tables: dict[str, pd.DataFrame],
        topic: str,
        adjective_class: str | None,
        adjective: str | None,
        view: str,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.root = root
        self.tables = tables
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.figure = None
        self.canvas = None

        root.title("Word frequencies")
        root.minsize(800, 600)
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill=tk.X)
        self.content = ttk.Frame(root, padding=(8, 0, 8, 8))
        self.content.pack(fill=tk.BOTH, expand=True)

        self.topic = tk.StringVar(value=topic)
        self.adjective_class = tk.StringVar(value=adjective_class or ALL)
        self.adjective = tk.StringVar(value=adjective or ALL)
        self.view = tk.StringVar(value=view.title())
        fields = [
            ("Topic", self.topic, get_topics(tables)),
            (
                "Adjective class",
                self.adjective_class,
                [ALL, *get_adjective_classes(tables, topic)],
            ),
            (
                "Adjective",
                self.adjective,
                [ALL, *get_adjectives(tables, topic, adjective_class)]
                if adjective_class
                else [ALL],
            ),
            ("View", self.view, ["Plot", "Table"]),
        ]
        self.selectors = {}
        for column, (label, variable, values) in enumerate(fields):
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
                state=(
                    "disabled"
                    if label == "Adjective" and not adjective_class
                    else "readonly"
                ),
                width=24 if label == "Topic" else 16,
            )
            selector.grid(row=1, column=column, padx=(0, 8), sticky="ew")
            selector.bind("<<ComboboxSelected>>", self.update_selection)
            self.selectors[label] = selector
        controls.columnconfigure(0, weight=1)
        self.render()

    def update_selection(self, event: Any) -> None:
        del event
        topic = self.topic.get()
        classes = [ALL, *get_adjective_classes(self.tables, topic)]
        class_selector = self.selectors["Adjective class"]
        class_selector.configure(values=classes)
        if self.adjective_class.get() not in classes:
            self.adjective_class.set(ALL)

        adjective_class = self.adjective_class.get()
        adjectives = (
            [ALL, *get_adjectives(self.tables, topic, adjective_class)]
            if adjective_class != ALL
            else [ALL]
        )
        adjective_selector = self.selectors["Adjective"]
        adjective_selector.configure(values=adjectives)
        if self.adjective.get() not in adjectives:
            self.adjective.set(ALL)
        adjective_selector.configure(
            state="readonly" if adjective_class != ALL else "disabled"
        )
        self.render()

    def render(self) -> None:
        adjective_class = self.adjective_class.get()
        adjective = self.adjective.get()
        adjective_class = None if adjective_class == ALL else adjective_class
        adjective = None if adjective == ALL else adjective
        frequencies = select_frequencies(
            self.tables,
            self.topic.get(),
            adjective_class,
            adjective,
        )
        if self.view.get() == "Table":
            figure = plot_top_words(
                select_top_words(frequencies),
                "Most common content lemmas\n"
                + format_selection(
                    self.topic.get(),
                    adjective_class,
                    adjective,
                ),
            )
        else:
            figure = plot_frequency_bars(
                frequencies,
                self.topic.get(),
                adjective_class,
                adjective,
            )
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            plt.close(self.figure)
        self.figure = figure
        self.canvas = self.FigureCanvasTkAgg(figure, master=self.content)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


def browse_word_frequencies(
    config_path: Path,
    topic: str,
    adjective_class: str | None = None,
    adjective: str | None = None,
    view: str = "plot",
) -> None:
    import tkinter as tk

    config = json.loads(config_path.read_text(encoding="utf-8"))
    tables = load_frequency_tables(config)
    topics = get_topics(tables)
    if topic not in topics:
        raise ValueError(f"Unknown topic: {topic}")
    classes = get_adjective_classes(tables, topic)
    if adjective_class and adjective_class not in classes:
        raise ValueError(f"Unknown adjective class: {adjective_class}")
    if adjective and not adjective_class:
        raise ValueError("--adjective requires --adjective-class")
    if adjective and adjective not in get_adjectives(tables, topic, adjective_class):
        raise ValueError(f"Unknown adjective: {adjective}")

    root = tk.Tk()
    WordFrequencyBrowser(
        root,
        tables,
        topic,
        adjective_class,
        adjective,
        view,
    )
    root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse lemma-frequency views")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--adjective-class")
    parser.add_argument("--adjective")
    parser.add_argument("--view", choices=("plot", "table"), default="plot")
    arguments = parser.parse_args()
    if arguments.adjective and not arguments.adjective_class:
        parser.error("--adjective requires --adjective-class")
    return arguments


if __name__ == "__main__":
    args = parse_args()
    browse_word_frequencies(
        args.config,
        args.topic,
        adjective_class=args.adjective_class,
        adjective=args.adjective,
        view=args.view,
    )
