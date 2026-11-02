from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.summarize_words import count_unique_lemmas


DEFAULT_CONFIG = Path(__file__).with_name("lexical_analysis_config.json")


def summarize_lemmas(config_path: Path) -> dict[str, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress = tqdm(
        config["levels"],
        desc="Summarizing compositional lemmas",
        unit="table",
    )
    output_counts = {}
    for level in progress:
        lemma_table = pd.read_csv(level["output_csv"])
        summary = count_unique_lemmas(
            lemma_table,
            tuple(level["summary_group_columns"]),
            count_column="count",
        )
        output_path = Path(level["summary_output_csv"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False, float_format="%.6f")
        output_counts[level["name"]] = len(summary)
    return output_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique lemmas for compositional groups"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    counts = summarize_lemmas(parse_args().config)
    for level, count in counts.items():
        print(f"Wrote {count} rows for {level}")
