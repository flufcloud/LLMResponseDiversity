from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from experiments.multimodal_thinking_experiment.analyze_words import (
    extract_content_terms,
    load_spacy_model,
)


DEFAULT_CONFIG = Path(__file__).with_name("lexical_analysis_config.json")


def count_grouped_lemmas(
    generations: pd.DataFrame,
    lemma_terms: list[list[str]],
    group_columns: tuple[str, ...],
    description: str,
) -> pd.DataFrame:
    total_counts: Counter[tuple[str, ...]] = Counter()
    poem_counts: Counter[tuple[str, ...]] = Counter()
    group_poem_counts: Counter[tuple[str, ...]] = Counter()
    groups = generations[list(group_columns)].itertuples(index=False, name=None)
    rows = zip(groups, lemma_terms, strict=True)
    for group, terms in tqdm(
        rows,
        total=len(generations),
        desc=f"Counting {description}",
        unit="poem",
    ):
        group_poem_counts[group] += 1
        total_counts.update(group + (lemma,) for lemma in terms)
        poem_counts.update(group + (lemma,) for lemma in set(terms))

    records = []
    for key, count in total_counts.items():
        group = key[:-1]
        record = dict(zip(group_columns, group, strict=True))
        record.update(
            {
                "lemma": key[-1],
                "count": count,
                "decimal_frequency": round(
                    poem_counts[key] / group_poem_counts[group],
                    2,
                ),
            }
        )
        records.append(record)
    columns = [*group_columns, "count", "lemma"]
    ascending = [True] * len(group_columns) + [False, True]
    return pd.DataFrame(records).sort_values(
        columns,
        ascending=ascending,
        ignore_index=True,
    )


def analyze_words(config_path: Path, nlp: Any | None = None) -> dict[str, int]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    generations = pd.read_csv(
        config["input_csv"],
        usecols=["topic", "adjective_group", "adjective", "response"],
    ).rename(columns={"adjective_group": "adjective_class"})
    nlp = nlp or load_spacy_model(config)
    _, _, lemma_terms = extract_content_terms(
        generations["response"],
        nlp,
        config,
    )

    output_counts = {}
    for level in config["levels"]:
        table = count_grouped_lemmas(
            generations,
            lemma_terms,
            tuple(level["group_columns"]),
            level["name"],
        )
        output_path = Path(level["output_csv"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False, float_format="%.2f")
        output_counts[level["name"]] = len(table)
    return output_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute grouped lemma frequencies for templated poems"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    counts = analyze_words(parse_args().config)
    for level, count in counts.items():
        print(f"Wrote {count} rows for {level}")
