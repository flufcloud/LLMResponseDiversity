from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lib.summarize_words import count_unique_lemmas


DEFAULT_CONFIG = Path(__file__).with_name("lexical_analysis_config.json")


def summarize_lemmas(config_path: Path) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    progress = tqdm(total=2, desc="Summarizing basic lemmas", unit="stage")
    lemma_table = pd.read_csv(config["lemma_topic_sample_usage_output"])
    progress.update()
    summary = count_unique_lemmas(
        lemma_table,
        ("topic",),
        lemma_column="word",
        count_column="num_usages_within_topic_generations",
    )
    output_path = Path(config["num_unique_lemmas_by_topic_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, float_format="%.6f")
    progress.update()
    progress.close()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count unique lemmas by basic-poetry topic"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    result = summarize_lemmas(parse_args().config)
    print(f"Wrote {len(result)} topic rows")
