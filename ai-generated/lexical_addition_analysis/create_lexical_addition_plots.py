from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CONFIG = Path(__file__).with_name("analysis_config.json")
LABELS = {"basic": "Basic", "adjective": "Adjective-balanced"}


def plot_metric(
    axis,
    curves: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for corpus, group in curves.groupby("corpus", sort=True):
        x = group["generation_number"]
        axis.plot(x, group[f"mean_{metric}"], label=LABELS[corpus])
        axis.fill_between(
            x,
            group[f"{metric}_p025"],
            group[f"{metric}_p975"],
            alpha=0.15,
        )
    axis.set(title=title, xlabel="Poems sampled", ylabel=ylabel)
    axis.spines[["top", "right"]].set_visible(False)


def create_plots(config_path: Path) -> tuple[Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    curves = pd.read_csv(config["outputs"]["dataset_curve"])
    accumulation_path = Path(config["plots"]["accumulation"])
    normalized_path = Path(config["plots"]["normalized_addition"])
    accumulation_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_metric(
        axes[0],
        curves,
        "cumulative_lemmas",
        "Cumulative lexical variety",
        "Mean unique lemmas",
    )
    plot_metric(
        axes[1],
        curves,
        "new_lemmas",
        "Lexical additions",
        "Mean new lemmas",
    )
    plot_metric(
        axes[2],
        curves,
        "no_new_lemmas",
        "No-new-lemma probability",
        "Probability",
    )
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(accumulation_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    plot_metric(
        axis,
        curves,
        "new_lemma_rate",
        "Normalized lexical additions",
        "New lemmas / distinct lemmas in incoming poem",
    )
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(normalized_path, dpi=180)
    plt.close(figure)
    return accumulation_path, normalized_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot lexical-addition curves")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    paths = create_plots(parse_args().config)
    print(f"Wrote {len(paths)} lexical-addition plots")
