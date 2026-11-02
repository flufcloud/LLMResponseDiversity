from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm-response-diversity-matplotlib")

import matplotlib
import numpy as np
import pandas as pd

from experiments.multimodal_thinking_experiment.prepare_poems import (
    DEFAULT_CONFIG,
    DEFAULT_RESULTS_ROOT,
    load_analysis_config,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THINKING_COLORS = {"minimal": "#4C78A8", "high": "#F58518"}


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_count_histogram(
    metrics: pd.DataFrame,
    column: str,
    xlabel: str,
    title: str,
    output_dir: Path,
    filename: str,
) -> None:
    maximum = int(metrics[column].max())
    bins = np.arange(-0.5, maximum + 1.5)
    figure, axis = plt.subplots(figsize=(8, 5))
    for thinking_level in ("minimal", "high"):
        values = metrics.loc[metrics["thinking_level"] == thinking_level, column]
        axis.hist(
            values,
            bins=bins,
            alpha=0.55,
            label=thinking_level.title(),
            color=THINKING_COLORS[thinking_level],
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Poems")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output_dir, filename)


def plot_similarity_bins(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    edges = np.linspace(0, 1, 11)
    values = np.column_stack(
        [
            np.histogram(
                summary.loc[
                    summary["thinking_level"] == thinking_level, "mean_similarity"
                ].clip(0, 1),
                edges,
            )[0]
            for thinking_level in ("minimal", "high")
        ]
    ).astype(float)
    values = 100 * values / np.maximum(values.sum(axis=0, keepdims=True), 1)
    values = values[::-1]
    labels = [
        f"{edges[index]:.1f}–{edges[index + 1]:.1f}" for index in range(9, -1, -1)
    ]

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    axis.set_xticks([0, 1], labels=["Minimal", "High"])
    axis.set_yticks(range(10), labels=labels)
    axis.set_xlabel("Thinking level")
    axis.set_ylabel("Topic-level mean cosine similarity")
    axis.set_title("Intra-model repetition by topic")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value >= 55 else "black",
            )
    figure.colorbar(image, ax=axis, label="Topics (%)", shrink=0.85)
    figure.tight_layout()
    save_figure(figure, output_dir, "intra_model_similarity_bins")


def plot_rhyme_table(rhyme_counts: pd.DataFrame, output_dir: Path) -> None:
    displayed = rhyme_counts.set_index("rhyme_scheme")[["minimal", "high"]]
    figure, axis = plt.subplots(figsize=(6, 2.4))
    axis.axis("off")
    table = axis.table(
        cellText=displayed.to_numpy(),
        rowLabels=displayed.index,
        colLabels=["Minimal", "High"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axis.set_title("Rhyme-scheme counts", pad=12)
    figure.tight_layout()
    save_figure(figure, output_dir, "rhyme_scheme_table")


def plot_quatrain_share(quatrains: pd.DataFrame, output_dir: Path) -> None:
    overall = quatrains[quatrains["thinking_level"] == "all"]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.pie(
        overall["count"],
        labels=overall["stanza_type"],
        autopct="%1.1f%%",
        colors=["#4C78A8", "#BAB0AC"],
        startangle=90,
    )
    axis.set_title("Four-line stanzas versus other stanza lengths")
    figure.tight_layout()
    save_figure(figure, output_dir, "quatrain_stanza_share")


def plot_cross_thinking_similarity(cross: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(cross["similarity"], bins=np.linspace(0, 1, 51), color="#72B7B2")
    mean = cross["similarity"].mean()
    axis.axvline(mean, color="#E45756", label=f"Mean: {mean:.3f}")
    axis.set_xlabel("Minimal–high cosine similarity")
    axis.set_ylabel("Poem pairs")
    axis.set_title("Cross-thinking response similarity")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    save_figure(figure, output_dir, "cross_thinking_similarity")


def plot_pii_counts(metrics: pd.DataFrame, output_dir: Path) -> None:
    if "pii_count" not in metrics:
        return
    plot_count_histogram(
        metrics,
        "pii_count",
        "PII elements",
        "PII elements per poem",
        output_dir,
        "pii_count_histogram",
    )


def create_method_plots(args: argparse.Namespace) -> None:
    config = load_analysis_config(Path(args.config))
    if args.method not in config["methods"]:
        raise ValueError(f"Unknown method: {args.method}")
    results_root = Path(args.results_root)
    data_dir = results_root / "data" / args.method
    tables_dir = results_root / "tables" / args.method
    output_dir = results_root / "plots" / args.method

    metrics = pd.read_csv(data_dir / "poem_metrics.csv")
    cross = pd.read_csv(data_dir / "cross_thinking_similarity.csv")
    similarity = pd.read_csv(tables_dir / "mean_pairwise_similarity_by_topic.csv")
    rhyme_counts = pd.read_csv(tables_dir / "rhyme_scheme_counts.csv")
    quatrains = pd.read_csv(tables_dir / "quatrain_stanza_counts.csv")

    plot_similarity_bins(similarity, output_dir)
    plot_count_histogram(
        metrics,
        "line_count",
        "Lines per poem",
        "Poem line-count distribution",
        output_dir,
        "line_count_histogram",
    )
    plot_count_histogram(
        metrics,
        "stanza_count",
        "Stanzas per poem",
        "Poem stanza-count distribution",
        output_dir,
        "stanza_count_histogram",
    )
    plot_rhyme_table(rhyme_counts, output_dir)
    plot_quatrain_share(quatrains, output_dir)
    plot_cross_thinking_similarity(cross, output_dir)
    plot_pii_counts(metrics, output_dir)
    print(f"Wrote plots for {args.method} under {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one poetry analysis method")
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    create_method_plots(parse_args())
