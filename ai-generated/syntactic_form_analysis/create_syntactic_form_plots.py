from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def create_accumulation_plot(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    curves = pd.read_csv(config["outputs"]["dataset_accumulation_curve"])
    output_path = Path(config["plots"]["accumulation"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for threshold, group in curves.groupby("similarity_threshold", sort=True):
        label = f"threshold {threshold:.2f}"
        x = group["generation_number"]
        axes[0].plot(x, group["mean_cumulative_forms"], label=label)
        axes[0].fill_between(
            x,
            group["cumulative_forms_p025"],
            group["cumulative_forms_p975"],
            alpha=0.12,
        )
        axes[1].plot(x, group["mean_new_forms"], label=label)
        axes[2].plot(
            x,
            group["mean_probability_no_new_forms"],
            label=label,
        )

    axes[0].set(title="Cumulative line forms", ylabel="Mean line forms")
    axes[1].set(title="Line forms added", ylabel="Mean new line forms")
    axes[2].set(
        title="No-new-line-form probability",
        ylabel="Probability",
        ylim=(0, 1),
    )
    for axis in axes:
        axis.set_xlabel("Poems sampled")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote plot to {output_path}")
    return output_path


def create_line_normalized_plot(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    curves = pd.read_csv(config["outputs"]["dataset_accumulation_curve"])
    output_path = Path(config["plots"]["line_normalized_new_forms"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))

    for threshold, group in curves.groupby("similarity_threshold", sort=True):
        axis.plot(
            group["generation_number"],
            group["mean_new_forms_per_line"],
            label=f"threshold {threshold:.2f}",
        )
        axis.fill_between(
            group["generation_number"],
            group["new_forms_per_line_p025"],
            group["new_forms_per_line_p975"],
            alpha=0.12,
        )

    axis.set(
        title="Effective new line forms added",
        xlabel="Poems sampled",
        ylabel="Mean new line forms / nonempty line",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Wrote plot to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot syntactic-form accumulation summaries"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_accumulation_plot(arguments.config)
    create_line_normalized_plot(arguments.config)
