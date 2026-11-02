from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from experiments.multimodal_thinking_experiment.analyze_poetic_form import (
    RHYME_SCHEMES,
    compute_poetic_form_metrics,
)

from .process_analysis_data import (
    DEFAULT_CONFIG,
    GROUP_LEVELS,
    get_data_path,
    get_table_path,
    load_config,
    load_responses,
)


FORM_UNITS = {
    "mean_line_count": "lines_per_poem",
    "mean_stanza_count": "stanzas_per_poem",
    "mean_stanza_length": "lines_per_stanza",
    "four_line_stanza_fraction": "fraction",
    "rhymed_line_fraction": "fraction",
    "iambic_pentameter_line_fraction": "fraction",
    "dominant_iambic_pentameter_poem_fraction": "fraction",
    "mean_iambic_alignment_score": "similarity",
}


def summarize_form_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    total_lines = frame["line_count"].sum()
    total_stanzas = frame["stanza_count"].sum()
    total_meter_lines = frame["scorable_meter_line_count"].sum()
    return {
        "num_poems": len(frame),
        "mean_line_count": frame["line_count"].mean(),
        "std_line_count": frame["line_count"].std(),
        "median_line_count": frame["line_count"].median(),
        "mean_stanza_count": frame["stanza_count"].mean(),
        "std_stanza_count": frame["stanza_count"].std(),
        "median_stanza_count": frame["stanza_count"].median(),
        "mean_stanza_length": (
            total_lines / total_stanzas if total_stanzas else 0.0
        ),
        "four_line_stanza_fraction": (
            frame["four_line_stanza_count"].sum() / total_stanzas
            if total_stanzas
            else 0.0
        ),
        "rhymed_line_fraction": (
            frame["rhymed_line_count"].sum() / total_lines if total_lines else 0.0
        ),
        "end_word_pronunciation_coverage": (
            frame["known_end_word_count"].sum() / total_lines
            if total_lines
            else 0.0
        ),
        "iambic_pentameter_line_fraction": (
            frame["iambic_pentameter_line_count"].sum() / total_meter_lines
            if total_meter_lines
            else 0.0
        ),
        "dominant_iambic_pentameter_poem_fraction": frame[
            "dominant_iambic_pentameter"
        ].mean(),
        "mean_iambic_alignment_score": (
            frame["iambic_alignment_score_sum"].sum() / total_meter_lines
            if total_meter_lines
            else 0.0
        ),
    }


def add_adjective_columns(
    frame: pd.DataFrame, responses: pd.DataFrame
) -> pd.DataFrame:
    metadata = responses[["poem_id", "adjective_class", "adjective"]]
    return frame.drop(columns="thinking_level").merge(
        metadata, on="poem_id", how="left", validate="many_to_one"
    )


def create_group_summary(
    metrics: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame([summarize_form_metrics(metrics)])
    records = []
    groups = metrics.groupby(group_columns, sort=True)
    for keys, group in tqdm(groups, total=groups.ngroups, desc="Summarizing groups"):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        records.append(
            {
                **dict(zip(group_columns, key_tuple, strict=True)),
                **summarize_form_metrics(group),
            }
        )
    return pd.DataFrame(records)


def create_distribution(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    value_name: str,
    required_values: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if group_columns:
        group_sizes = frame.groupby(group_columns, sort=True).size().rename("total")
        counts = (
            frame.groupby([*group_columns, value_column], sort=True)
            .size()
            .rename("count")
        )
        if required_values:
            group_index = group_sizes.index
            if not isinstance(group_index, pd.MultiIndex):
                group_index = pd.MultiIndex.from_arrays(
                    [group_index], names=group_columns
                )
            full_index = pd.MultiIndex.from_tuples(
                [(*keys, value) for keys in group_index for value in required_values],
                names=[*group_columns, value_column],
            )
            counts = counts.reindex(full_index, fill_value=0)
        distribution = counts.reset_index()
        distribution["fraction"] = distribution["count"] / distribution.merge(
            group_sizes.reset_index(), on=group_columns, how="left"
        )["total"]
    else:
        counts = frame[value_column].value_counts().sort_index()
        if required_values:
            counts = counts.reindex(required_values, fill_value=0)
        distribution = counts.rename("count").rename_axis(value_column).reset_index()
        distribution["fraction"] = distribution["count"] / len(frame)
    return distribution.rename(columns={value_column: value_name})


def create_form_comparison(
    summaries: dict[str, pd.DataFrame],
    rhyme_distributions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for metric, unit in FORM_UNITS.items():
        record = {
            "metric_family": "poetic_form",
            "metric": metric,
            "unit": unit,
            "corpus_value": summaries["corpus"].loc[0, metric],
        }
        for level in GROUP_LEVELS:
            values = summaries[level][metric]
            record.update(
                {
                    f"mean_{level}_value": values.mean(),
                    f"{level}_std": values.std(),
                    f"{level}_min": values.min(),
                    f"{level}_max": values.max(),
                    f"n_{level}_groups": len(values),
                }
            )
        records.append(record)

    for scheme in RHYME_SCHEMES:
        name = re.sub(r"[^a-z]+", "_", scheme.lower()).strip("_")
        corpus = rhyme_distributions["corpus"]
        record = {
            "metric_family": "rhyme_scheme",
            "metric": f"rhyme_{name}_poem_fraction",
            "unit": "fraction",
            "corpus_value": corpus.loc[
                corpus["rhyme_scheme"] == scheme, "fraction"
            ].iloc[0],
        }
        for level in GROUP_LEVELS:
            values = rhyme_distributions[level].loc[
                rhyme_distributions[level]["rhyme_scheme"] == scheme,
                "fraction",
            ]
            record.update(
                {
                    f"mean_{level}_value": values.mean(),
                    f"{level}_std": values.std(),
                    f"{level}_min": values.min(),
                    f"{level}_max": values.max(),
                    f"n_{level}_groups": len(values),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def write_distributions(
    metrics: pd.DataFrame,
    stanzas: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    outputs = {}
    specifications = (
        ("rhyme_scheme", metrics, "rhyme_scheme", "rhyme_scheme", RHYME_SCHEMES),
        ("line_count", metrics, "line_count", "line_count", None),
        ("stanza_count", metrics, "stanza_count", "stanza_count", None),
        ("stanza_length", stanzas, "stanza_length", "stanza_length", None),
    )
    for level, columns in {"corpus": [], **GROUP_LEVELS}.items():
        for name, frame, value_column, value_name, required in specifications:
            table = create_distribution(
                frame, columns, value_column, value_name, required
            )
            outputs[f"{name}_distribution_{level}"] = table
            table.to_csv(
                get_table_path(config, f"{name}_distribution_{level}.csv"),
                index=False,
                float_format="%.6f",
            )
    return outputs


def analyze_form(config_path: Path) -> dict[str, pd.DataFrame]:
    config = load_config(config_path)
    responses = load_responses(config)
    form_input = responses.assign(thinking_level="all")
    metrics, stanzas = compute_poetic_form_metrics(form_input, config)
    metrics = add_adjective_columns(metrics, responses)
    stanzas = add_adjective_columns(stanzas, responses)

    data_dir = get_data_path(config, "poetic_form_metrics.csv").parent
    table_dir = get_table_path(config, "form_summary_corpus.csv").parent
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(
        data_dir / "poetic_form_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    stanzas.to_csv(data_dir / "stanza_metrics.csv", index=False)

    summaries = {}
    for level, columns in {"corpus": [], **GROUP_LEVELS}.items():
        summary = create_group_summary(metrics, columns)
        summaries[level] = summary
        summary.to_csv(
            table_dir / f"form_summary_{level}.csv",
            index=False,
            float_format="%.6f",
        )
    distributions = write_distributions(metrics, stanzas, config)
    rhyme_distributions = {
        level: distributions[f"rhyme_scheme_distribution_{level}"]
        for level in ("corpus", *GROUP_LEVELS)
    }
    comparison = create_form_comparison(summaries, rhyme_distributions)
    comparison.to_csv(
        table_dir / "form_comparisons.csv", index=False, float_format="%.6f"
    )
    return {
        "metrics": metrics,
        "stanzas": stanzas,
        **summaries,
        **distributions,
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rhyme, meter, lines, and stanzas for adjective poems"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    tables = analyze_form(parse_args().config)
    print(f"Wrote {len(tables)} form tables")
