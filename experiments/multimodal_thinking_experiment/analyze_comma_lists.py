from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


RESULT_ROOT = Path("results/multimodal_thinking_experiment")
DEFAULT_RESPONSES = RESULT_ROOT / "data/poetry_basic/responses.csv"
DEFAULT_OCCURRENCES = RESULT_ROOT / "data/poetry_basic/three_comma_lines.csv"
DEFAULT_METRICS = RESULT_ROOT / "data/poetry_basic/three_comma_line_metrics.csv"
DEFAULT_SUMMARY = RESULT_ROOT / "tables/poetry_basic/three_comma_line_summary.csv"
DEFAULT_DISTRIBUTION = (
    RESULT_ROOT / "tables/poetry_basic/three_comma_line_count_distribution.csv"
)


def find_three_comma_lines(text: str) -> list[tuple[int, str]]:
    matches = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped and stripped.count(",") == 3:
            matches.append((line_number, stripped))
    return matches


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    counts = metrics["three_comma_line_count"]
    return pd.DataFrame(
        [
            {
                "num_poems": len(metrics),
                "total_lines": int(metrics["line_count"].sum()),
                "total_three_comma_lines": int(counts.sum()),
                "poems_with_three_comma_line": int(counts.gt(0).sum()),
                "fraction_poems_with_three_comma_line": counts.gt(0).mean(),
                "mean_three_comma_lines_per_poem": counts.mean(),
                "median_three_comma_lines_per_poem": counts.median(),
                "maximum_three_comma_lines_in_poem": int(counts.max()),
            }
        ]
    )


def create_count_distribution(metrics: pd.DataFrame) -> pd.DataFrame:
    distribution = (
        metrics["three_comma_line_count"]
        .value_counts(sort=False)
        .sort_index()
        .rename_axis("three_comma_line_count")
        .reset_index(name="poem_count")
    )
    distribution["poem_fraction"] = distribution["poem_count"] / len(metrics)
    return distribution


def analyze_three_comma_lines(
    responses_csv: Path,
    occurrences_csv: Path,
    metrics_csv: Path,
    summary_csv: Path,
    distribution_csv: Path,
) -> dict[str, pd.DataFrame]:
    responses = pd.read_csv(responses_csv, keep_default_na=False)
    occurrence_records = []
    metric_records = []

    rows = tqdm(
        responses.itertuples(index=False),
        total=len(responses),
        desc="Searching three-comma lines",
        unit="poem",
    )
    for row in rows:
        nonempty_lines = [line for line in row.response.splitlines() if line.strip()]
        matches = find_three_comma_lines(row.response)
        metric_records.append(
            {
                "poem_id": row.poem_id,
                "generation_id": row.generation_id,
                "topic": row.topic,
                "thinking_level": row.thinking_level,
                "source_sample_idx": row.source_sample_idx,
                "line_count": len(nonempty_lines),
                "three_comma_line_count": len(matches),
            }
        )
        for line_number, line_text in matches:
            occurrence_records.append(
                {
                    "poem_id": row.poem_id,
                    "generation_id": row.generation_id,
                    "topic": row.topic,
                    "thinking_level": row.thinking_level,
                    "source_sample_idx": row.source_sample_idx,
                    "line_number": line_number,
                    "line_text": line_text,
                    "comma_count": 3,
                }
            )

    metrics = pd.DataFrame(metric_records)
    occurrences = pd.DataFrame(
        occurrence_records,
        columns=[
            "poem_id",
            "generation_id",
            "topic",
            "thinking_level",
            "source_sample_idx",
            "line_number",
            "line_text",
            "comma_count",
        ],
    )
    summary = summarize_metrics(metrics)
    distribution = create_count_distribution(metrics)
    outputs = {
        "occurrences": (occurrences, occurrences_csv),
        "metrics": (metrics, metrics_csv),
        "summary": (summary, summary_csv),
        "distribution": (distribution, distribution_csv),
    }
    for table, path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6f")
    return {name: table for name, (table, _) in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count basic-poetry lines containing exactly three commas"
    )
    parser.add_argument("--responses-csv", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--occurrences-csv", type=Path, default=DEFAULT_OCCURRENCES)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--distribution-csv", type=Path, default=DEFAULT_DISTRIBUTION
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    tables = analyze_three_comma_lines(
        arguments.responses_csv,
        arguments.occurrences_csv,
        arguments.metrics_csv,
        arguments.summary_csv,
        arguments.distribution_csv,
    )
    print(
        f"Found {len(tables['occurrences'])} three-comma lines "
        f"across {len(tables['metrics'])} poems"
    )
