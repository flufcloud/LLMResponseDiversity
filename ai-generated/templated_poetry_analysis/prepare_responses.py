from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .process_analysis_data import (
    DEFAULT_CONFIG,
    get_data_path,
    load_config,
)


OUTPUT_COLUMNS = [
    "poem_id",
    "generation_id",
    "topic_id",
    "topic",
    "adjective_class",
    "adjective",
    "sample_idx",
    "response",
]


def prepare_responses(config_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    generations = pd.read_csv(config["input_csv"], keep_default_na=False)
    records = []
    for row in tqdm(
        generations.itertuples(index=False),
        total=len(generations),
        desc="Preparing poems",
        unit="poem",
    ):
        records.append(
            {
                "poem_id": row.generation_id,
                "generation_id": row.generation_id,
                "topic_id": row.topic_id,
                "topic": row.topic,
                "adjective_class": row.adjective_group,
                "adjective": row.adjective,
                "sample_idx": row.sample_idx,
                "response": row.response,
            }
        )
    responses = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    output = get_data_path(config, "responses.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    responses.to_csv(output, index=False)
    return responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare adjective-poetry responses")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    table = prepare_responses(parse_args().config)
    print(f"Wrote {len(table)} poems")
