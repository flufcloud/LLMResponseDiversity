from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.compute_identifiers import compute_stable_hash


DEFAULT_CONFIG = Path(__file__).with_name("analysis_config.json")
METRICS = (
    "cumulative_lemmas",
    "new_lemmas",
    "new_lemma_rate",
    "no_new_lemmas",
)


def load_features(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            record["lemma_set"] = frozenset(record.pop("lemmas"))
            records.append(record)
    return pd.DataFrame(records)


def get_group_seed(seed: int, corpus: str, topic: str) -> int:
    offset = int(compute_stable_hash(corpus, topic)[:8], 16)
    return seed + offset


def sample_balanced_adjectives(
    group: pd.DataFrame,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    adjective_indices = {
        adjective: values.index.to_numpy()
        for adjective, values in group.groupby("adjective", sort=True)
    }
    adjectives = np.asarray(sorted(adjective_indices))
    base, remainder = divmod(sample_size, len(adjectives))
    extra = set(rng.choice(adjectives, size=remainder, replace=False))
    selected = []
    for adjective in adjectives:
        count = base + int(adjective in extra)
        selected.extend(
            rng.choice(adjective_indices[adjective], size=count, replace=False)
        )
    return rng.permutation(selected)


def measure_order(
    order: np.ndarray,
    lemma_sets: pd.Series,
) -> dict[str, np.ndarray]:
    size = len(order)
    cumulative = np.zeros(size, dtype=np.int32)
    additions = np.zeros(size, dtype=np.int32)
    rates = np.zeros(size, dtype=np.float64)
    no_new = np.zeros(size, dtype=np.float64)
    observed: set[str] = set()
    for step, index in enumerate(order):
        poem_lemmas = lemma_sets.loc[index]
        new_lemmas = poem_lemmas - observed
        observed.update(poem_lemmas)
        cumulative[step] = len(observed)
        additions[step] = len(new_lemmas)
        rates[step] = len(new_lemmas) / len(poem_lemmas) if poem_lemmas else 0.0
        no_new[step] = not new_lemmas
    return {
        "cumulative_lemmas": cumulative,
        "new_lemmas": additions,
        "new_lemma_rate": rates,
        "no_new_lemmas": no_new,
    }


def summarize_repetitions(
    values: dict[str, np.ndarray],
    corpus: str,
    topic: str | None = None,
) -> pd.DataFrame:
    records = []
    sample_size = next(iter(values.values())).shape[1]
    for step in range(sample_size):
        record: dict[str, Any] = {
            "corpus": corpus,
            "generation_number": step + 1,
        }
        if topic is not None:
            record["topic"] = topic
        for metric, matrix in values.items():
            column = matrix[:, step]
            record[f"mean_{metric}"] = column.mean()
            record[f"{metric}_p025"] = np.quantile(column, 0.025)
            record[f"{metric}_p975"] = np.quantile(column, 0.975)
        records.append(record)
    return pd.DataFrame(records)


def analyze_corpus(
    corpus: str,
    features: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repetitions = config["random_orderings"]
    sample_size = config["sample_size"]
    topic_records = []
    dataset_values = {
        metric: np.zeros((repetitions, sample_size), dtype=np.float64)
        for metric in METRICS
    }
    groups = features.groupby("topic", sort=True)
    progress = tqdm(
        groups,
        total=groups.ngroups,
        desc=f"Analyzing {corpus} topics",
        unit="topic",
    )
    for topic, group in progress:
        rng = np.random.default_rng(
            get_group_seed(config["random_seed"], corpus, topic)
        )
        topic_values = {
            metric: np.zeros((repetitions, sample_size), dtype=np.float64)
            for metric in METRICS
        }
        for repetition in range(repetitions):
            if corpus == "adjective":
                order = sample_balanced_adjectives(group, sample_size, rng)
            else:
                order = rng.choice(group.index, size=sample_size, replace=False)
            measured = measure_order(order, features["lemma_set"])
            for metric in METRICS:
                topic_values[metric][repetition] = measured[metric]
                dataset_values[metric][repetition] += measured[metric]
        topic_records.append(
            summarize_repetitions(topic_values, corpus, topic)
        )

    num_topics = groups.ngroups
    for metric in METRICS:
        dataset_values[metric] /= num_topics
    return (
        pd.concat(topic_records, ignore_index=True),
        summarize_repetitions(dataset_values, corpus),
    )


def create_summary(dataset_curve: pd.DataFrame) -> pd.DataFrame:
    final_generation = dataset_curve["generation_number"].max()
    final = dataset_curve[dataset_curve["generation_number"] == final_generation]
    columns = [
        "corpus",
        "generation_number",
        "mean_cumulative_lemmas",
        "mean_new_lemmas",
        "mean_new_lemma_rate",
        "mean_no_new_lemmas",
    ]
    return final[columns].rename(
        columns={
            "generation_number": "sample_size",
            "mean_no_new_lemmas": "probability_no_new_lemmas",
        }
    )


def analyze_lexical_addition(config_path: Path) -> dict[str, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    topic_tables = {}
    dataset_tables = []
    for corpus, corpus_config in config["corpora"].items():
        features = load_features(Path(corpus_config["features_jsonl"]))
        topic_curve, dataset_curve = analyze_corpus(corpus, features, config)
        topic_tables[corpus] = topic_curve
        dataset_tables.append(dataset_curve)

    dataset_curve = pd.concat(dataset_tables, ignore_index=True)
    summary = create_summary(dataset_curve)
    outputs = config["outputs"]
    for path in outputs.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    topic_tables["basic"].to_csv(outputs["basic_topic_curve"], index=False)
    topic_tables["adjective"].to_csv(
        outputs["adjective_topic_curve"], index=False
    )
    dataset_curve.to_csv(outputs["dataset_curve"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    return {
        "basic_topic_curve": topic_tables["basic"],
        "adjective_topic_curve": topic_tables["adjective"],
        "dataset_curve": dataset_curve,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure lexical additions from successive poems"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    tables = analyze_lexical_addition(parse_args().config)
    print(f"Wrote {len(tables)} lexical-addition tables")
