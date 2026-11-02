from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .process_analysis_data import DEFAULT_CONFIG, get_table_path, load_config


def combine_results(config_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    tables = []
    inputs = (
        "similarity_comparisons.csv",
        "form_comparisons.csv",
    )
    for name in tqdm(inputs, desc="Combining tables", unit="table"):
        tables.append(pd.read_csv(get_table_path(config, name)))
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_csv(
        get_table_path(config, "homogeneity_comparisons.csv"),
        index=False,
        float_format="%.6f",
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine adjective-poetry comparisons")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    table = combine_results(parse_args().config)
    print(f"Wrote {len(table)} comparison rows")
