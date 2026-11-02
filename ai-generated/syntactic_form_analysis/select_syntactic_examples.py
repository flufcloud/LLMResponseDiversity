from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .analyze_syntactic_forms import (
    DEFAULT_CONFIG,
    load_line_features,
)
from lib.measure_syntactic_similarity import compute_normalized_wl_kernel


def select_examples(
    lines: pd.DataFrame,
    targets: list[float],
    pair_samples: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    topic_groups = [
        (topic, group.reset_index(drop=True))
        for topic, group in lines.groupby("topic", sort=True)
        if group["poem_id"].nunique() > 1
    ]
    pair_counts = np.array(
        [len(group) * (len(group) - 1) // 2 for _, group in topic_groups],
        dtype=float,
    )
    topic_probabilities = pair_counts / pair_counts.sum()
    best: dict[float, tuple[float, dict] | None] = {
        target: None for target in targets
    }
    progress = tqdm(total=pair_samples, desc="Selecting line pairs", unit="pair")
    completed = 0
    while completed < pair_samples:
        group_index = rng.choice(len(topic_groups), p=topic_probabilities)
        topic, group = topic_groups[group_index]
        left, right = rng.choice(len(group), size=2, replace=False)
        left_row = group.iloc[left]
        right_row = group.iloc[right]
        if (
            left_row["poem_id"] == right_row["poem_id"]
            or left_row["line_text"] == right_row["line_text"]
        ):
            continue
        similarity = compute_normalized_wl_kernel(
            left_row["features"], right_row["features"]
        )
        for target in targets:
            difference = abs(similarity - target)
            prior = best[target]
            if prior is None or difference < prior[0]:
                best[target] = (
                    difference,
                    {
                        "target_similarity": target,
                        "syntactic_similarity": similarity,
                        "topic": topic,
                        "line_1": left_row["line_text"],
                        "line_2": right_row["line_text"],
                    },
                )
        completed += 1
        progress.update()
    progress.close()
    return pd.DataFrame([best[target][1] for target in targets])


def select_syntactic_examples(config_path: Path) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lines = load_line_features(Path(config["line_features_jsonl"]))
    examples = select_examples(
        lines,
        config["example_similarity_targets"],
        config["example_pair_samples"],
        config["random_seed"],
    )
    output = Path(config["outputs"]["line_similarity_examples"])
    output.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output, index=False, float_format="%.6f")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative line-level syntax pairs"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    table = select_syntactic_examples(parse_args().config)
    print(f"Wrote {len(table)} line-pair examples")
