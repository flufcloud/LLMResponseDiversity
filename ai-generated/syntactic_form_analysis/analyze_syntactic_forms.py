from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from lib.compute_identifiers import compute_stable_hash
from lib.measure_syntactic_similarity import compute_normalized_wl_kernel


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_line_features(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            record["features"] = Counter(record["features"])
            records.append(record)
    return pd.DataFrame(records)


def compute_line_similarity_matrix(features: list[Counter]) -> np.ndarray:
    size = len(features)
    matrix = np.eye(size, dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            similarity = compute_normalized_wl_kernel(
                features[left], features[right]
            )
            matrix[left, right] = similarity
            matrix[right, left] = similarity
    return matrix


def cluster_line_forms(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(similarity_matrix) == 1:
        return np.array([1], dtype=int)
    distances = np.clip(1 - similarity_matrix, 0, 1)
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=1 - threshold,
    ).fit_predict(distances)
    label_map: dict[int, int] = {}
    normalized = []
    for label in labels:
        if int(label) not in label_map:
            label_map[int(label)] = len(label_map) + 1
        normalized.append(label_map[int(label)])
    return np.array(normalized, dtype=int)


def summarize_similarity_matrix(
    similarity_matrix: np.ndarray,
) -> dict[str, float | int]:
    values = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)]
    return {
        "num_line_pairs": len(values),
        "mean_similarity": values.mean(),
        "similarity_p10": np.quantile(values, 0.10),
        "similarity_p25": np.quantile(values, 0.25),
        "similarity_p50": np.quantile(values, 0.50),
        "similarity_p75": np.quantile(values, 0.75),
        "similarity_p90": np.quantile(values, 0.90),
        "similarity_p95": np.quantile(values, 0.95),
        "similarity_p99": np.quantile(values, 0.99),
    }


def create_group_seed(
    random_seed: int,
    group_values: dict[str, Any],
    threshold: float,
) -> int:
    payload = json.dumps(
        {"group": group_values, "threshold": threshold},
        sort_keys=True,
    )
    offset = int(compute_stable_hash(payload)[:8], 16)
    return random_seed + offset


def count_nonempty_lines(text: str) -> int:
    lines = (line for line in text.splitlines() if line.strip())
    return sum(1 for _ in lines)


def summarize_accumulation(
    lines: pd.DataFrame,
    labels: np.ndarray,
    poem_line_counts: dict[str, int],
    repetitions: int,
    random_seed: int,
) -> pd.DataFrame:
    poems = lines["poem_id"].drop_duplicates().to_numpy()
    poem_forms = {
        poem_id: set(labels[lines["poem_id"].to_numpy() == poem_id])
        for poem_id in poems
    }
    cumulative = np.zeros((repetitions, len(poems)), dtype=np.int32)
    new_forms = np.zeros_like(cumulative)
    new_forms_per_line = np.zeros(
        (repetitions, len(poems)), dtype=np.float64
    )
    rng = np.random.default_rng(random_seed)

    for repetition in range(repetitions):
        seen: set[int] = set()
        for step, poem_id in enumerate(rng.permutation(poems)):
            introduced = poem_forms[poem_id] - seen
            seen.update(poem_forms[poem_id])
            cumulative[repetition, step] = len(seen)
            new_forms[repetition, step] = len(introduced)
            new_forms_per_line[repetition, step] = (
                len(introduced) / poem_line_counts[poem_id]
            )

    records = []
    for step in range(len(poems)):
        records.append(
            {
                "generation_number": step + 1,
                "mean_cumulative_forms": cumulative[:, step].mean(),
                "cumulative_forms_p025": np.quantile(cumulative[:, step], 0.025),
                "cumulative_forms_p975": np.quantile(cumulative[:, step], 0.975),
                "mean_new_forms": new_forms[:, step].mean(),
                "new_forms_p025": np.quantile(new_forms[:, step], 0.025),
                "new_forms_p975": np.quantile(new_forms[:, step], 0.975),
                "mean_new_forms_per_line": new_forms_per_line[:, step].mean(),
                "new_forms_per_line_p025": np.quantile(
                    new_forms_per_line[:, step], 0.025
                ),
                "new_forms_per_line_p975": np.quantile(
                    new_forms_per_line[:, step], 0.975
                ),
                "probability_no_new_forms": np.mean(new_forms[:, step] == 0),
            }
        )
    return pd.DataFrame(records)


def summarize_form_distribution(
    lines: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, float | int]:
    counts = np.array(list(Counter(labels).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )
    singleton_count = int(np.sum(counts == 1))
    return {
        "num_poems": int(lines["poem_id"].nunique()),
        "num_lines": len(lines),
        "num_forms": len(counts),
        "form_line_ratio": len(counts) / len(lines),
        "singleton_form_count": singleton_count,
        "singleton_form_fraction": singleton_count / len(counts),
        "largest_form_line_share": probabilities.max(),
        "top_three_form_line_share": np.sort(probabilities)[-3:].sum(),
        "shannon_entropy": entropy,
        "effective_forms_shannon": math.exp(entropy),
        "effective_form_line_ratio": math.exp(entropy) / len(lines),
        "simpson_concentration": np.square(probabilities).sum(),
        "effective_forms_simpson": 1 / np.square(probabilities).sum(),
        "observed_sample_coverage": 1 - singleton_count / len(lines),
    }


def create_occupancy_records(
    lines: pd.DataFrame,
    labels: np.ndarray,
    similarity_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    records = []
    for form_id in sorted(set(labels)):
        indices = np.flatnonzero(labels == form_id)
        within_form = similarity_matrix[np.ix_(indices, indices)]
        representative_index = indices[np.argmax(within_form.mean(axis=1))]
        representative = lines.iloc[representative_index]
        records.append(
            {
                "form_id": int(form_id),
                "line_count": len(indices),
                "poem_count": lines.iloc[indices]["poem_id"].nunique(),
                "line_share": len(indices) / len(lines),
                "representative_line_id": representative["line_id"],
                "representative_line": representative["line_text"],
            }
        )
    return records


def add_group_values(
    records: list[dict[str, Any]],
    group_values: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    return [
        {**group_values, "similarity_threshold": threshold, **record}
        for record in records
    ]


def summarize_dataset(
    distributions: pd.DataFrame,
    curves: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distribution_summary = (
        distributions.groupby("similarity_threshold", sort=True)
        .agg(
            num_topics=("topic", "nunique"),
            mean_num_forms=("num_forms", "mean"),
            median_num_forms=("num_forms", "median"),
            mean_singleton_form_fraction=("singleton_form_fraction", "mean"),
            mean_largest_form_line_share=(
                "largest_form_line_share",
                "mean",
            ),
            mean_effective_forms_shannon=("effective_forms_shannon", "mean"),
            mean_form_line_ratio=("form_line_ratio", "mean"),
            mean_effective_form_line_ratio=(
                "effective_form_line_ratio",
                "mean",
            ),
            mean_effective_forms_simpson=("effective_forms_simpson", "mean"),
            mean_final_new_forms=(
                "mean_new_forms_at_final_generation",
                "mean",
            ),
            mean_final_new_forms_per_line=(
                "mean_new_forms_per_line_at_final_generation",
                "mean",
            ),
            mean_final_no_new_probability=(
                "probability_no_new_forms_at_final_generation",
                "mean",
            ),
        )
        .reset_index()
    )
    curve_summary = (
        curves.groupby(
            ["similarity_threshold", "generation_number"],
            sort=True,
        )
        .agg(
            mean_cumulative_forms=("mean_cumulative_forms", "mean"),
            cumulative_forms_p025=(
                "mean_cumulative_forms",
                lambda values: values.quantile(0.025),
            ),
            cumulative_forms_p975=(
                "mean_cumulative_forms",
                lambda values: values.quantile(0.975),
            ),
            mean_new_forms=("mean_new_forms", "mean"),
            new_forms_p025=(
                "mean_new_forms",
                lambda values: values.quantile(0.025),
            ),
            new_forms_p975=(
                "mean_new_forms",
                lambda values: values.quantile(0.975),
            ),
            mean_new_forms_per_line=("mean_new_forms_per_line", "mean"),
            new_forms_per_line_p025=(
                "mean_new_forms_per_line",
                lambda values: values.quantile(0.025),
            ),
            new_forms_per_line_p975=(
                "mean_new_forms_per_line",
                lambda values: values.quantile(0.975),
            ),
            mean_probability_no_new_forms=(
                "probability_no_new_forms",
                "mean",
            ),
        )
        .reset_index()
    )
    return distribution_summary, curve_summary


def analyze_syntactic_forms(config_path: Path) -> dict[str, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lines = load_line_features(Path(config["line_features_jsonl"]))
    responses = pd.read_csv(config["input_csv"], keep_default_na=False)
    poem_line_counts = {
        row.poem_id: count_nonempty_lines(row.response)
        for row in responses.itertuples(index=False)
    }
    group_columns = config["group_columns"]
    assignments = []
    occupancies = []
    distributions = []
    curves = []
    similarity_summaries = []
    groups = list(lines.groupby(group_columns, sort=True))
    progress = tqdm(
        total=len(groups) * len(config["similarity_thresholds"]),
        desc="Analyzing syntactic forms",
        unit="group-threshold",
    )

    for group_key, group in groups:
        group = group.reset_index(drop=True)
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_columns, key_values, strict=True))
        similarity_matrix = compute_line_similarity_matrix(
            group["features"].tolist()
        )
        similarity_summary = summarize_similarity_matrix(similarity_matrix)
        similarity_summaries.append({**group_values, **similarity_summary})
        for threshold in config["similarity_thresholds"]:
            labels = cluster_line_forms(similarity_matrix, threshold)
            assignment_columns = [
                "line_id",
                "poem_id",
                "generation_id",
                "thinking_level",
                "source_sample_idx",
                "line_idx",
                "line_text",
            ]
            assignment = group[assignment_columns].copy()
            for column, value in group_values.items():
                assignment[column] = value
            assignment["similarity_threshold"] = threshold
            assignment["form_id"] = labels
            assignments.append(assignment)

            occupancy = create_occupancy_records(group, labels, similarity_matrix)
            occupancies.extend(add_group_values(occupancy, group_values, threshold))
            distribution = summarize_form_distribution(group, labels)
            seed = create_group_seed(
                config["random_seed"], group_values, threshold
            )
            curve = summarize_accumulation(
                group,
                labels,
                poem_line_counts,
                config["random_orderings"],
                seed,
            )
            curve_records = curve.to_dict("records")
            curves.extend(add_group_values(curve_records, group_values, threshold))
            distribution.update(
                {
                    "mean_new_forms_at_final_generation": curve.iloc[-1][
                        "mean_new_forms"
                    ],
                    "mean_new_forms_per_line_at_final_generation": curve.iloc[-1][
                        "mean_new_forms_per_line"
                    ],
                    "probability_no_new_forms_at_final_generation": curve.iloc[-1][
                        "probability_no_new_forms"
                    ],
                }
            )
            distributions.extend(
                add_group_values([distribution], group_values, threshold)
            )
            progress.update()
    progress.close()

    distribution_table = pd.DataFrame(distributions)
    curve_table = pd.DataFrame(curves)
    dataset_distribution, dataset_curve = summarize_dataset(
        distribution_table,
        curve_table,
    )
    tables = {
        "line_form_assignments": pd.concat(assignments, ignore_index=True),
        "line_similarity_summary": pd.DataFrame(similarity_summaries),
        "form_occupancy": pd.DataFrame(occupancies),
        "form_distribution_summary": distribution_table,
        "form_accumulation_curve": curve_table,
        "dataset_distribution_summary": dataset_distribution,
        "dataset_accumulation_curve": dataset_curve,
    }
    for name, table in tables.items():
        output_path = Path(config["outputs"][name])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False, float_format="%.6f")
        print(f"Wrote {len(table)} rows to {output_path}")
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure syntactic-form discovery and occupancy"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze_syntactic_forms(arguments.config)
