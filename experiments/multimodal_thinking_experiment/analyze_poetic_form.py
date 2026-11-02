from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pronouncing
from tqdm import tqdm

from experiments.multimodal_thinking_experiment.analyze import (
    get_line_end_word,
    split_stanzas,
)
from experiments.multimodal_thinking_experiment.prepare_poems import (
    load_embedding_responses,
)


DEFAULT_CONFIG = Path(__file__).with_name("poetic_form_analysis_config.json")
RHYME_SCHEMES = ("AA", "AB", "ABC", "Other / non-rhyming")
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+")
VOWEL_TAIL_PATTERN = re.compile(r"[aeiouy]+[^aeiouy]*$")


@lru_cache(maxsize=None)
def get_phones(word: str) -> tuple[str, ...]:
    normalized = word.lower().replace("’", "'").removesuffix("'s")
    return tuple(pronouncing.phones_for_word(normalized))


@lru_cache(maxsize=None)
def get_rhyme_keys(word: str) -> frozenset[str]:
    phones = get_phones(word)
    if phones:
        return frozenset(pronouncing.rhyming_part(value) for value in phones)
    normalized = word.lower().replace("’", "'").removesuffix("'s")
    match = VOWEL_TAIL_PATTERN.search(normalized)
    fallback = match.group(0) if match else normalized[-3:]
    return frozenset({f"orth:{fallback}"}) if fallback else frozenset()


def words_rhyme(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left == right or bool(get_rhyme_keys(left) & get_rhyme_keys(right))


def get_scheme_pairs(line_count: int, cycle_length: int) -> list[tuple[int, int]]:
    pairs = []
    block_length = cycle_length * 2
    for start in range(0, line_count, block_length):
        for offset in range(cycle_length):
            left = start + offset
            right = left + cycle_length
            if right < line_count:
                pairs.append((left, right))
    return pairs


def score_rhyme_scheme(
    stanzas: list[list[str]], cycle_length: int
) -> tuple[float, int, int]:
    matched_lines: set[tuple[int, int]] = set()
    eligible_lines: set[tuple[int, int]] = set()
    for stanza_index, lines in enumerate(stanzas):
        end_words = [get_line_end_word(line) for line in lines]
        for left, right in get_scheme_pairs(len(lines), cycle_length):
            eligible_lines.update({(stanza_index, left), (stanza_index, right)})
            if words_rhyme(end_words[left], end_words[right]):
                matched_lines.update({(stanza_index, left), (stanza_index, right)})
    score = len(matched_lines) / len(eligible_lines) if eligible_lines else 0.0
    return score, len(matched_lines), len(eligible_lines)


def count_rhymed_lines(
    stanzas: list[list[str]], maximum_distance: int
) -> int:
    rhymed_lines: set[tuple[int, int]] = set()
    for stanza_index, lines in enumerate(stanzas):
        end_words = [get_line_end_word(line) for line in lines]
        for left in range(len(lines)):
            stop = min(len(lines), left + maximum_distance + 1)
            for right in range(left + 1, stop):
                if words_rhyme(end_words[left], end_words[right]):
                    rhymed_lines.update({(stanza_index, left), (stanza_index, right)})
    return len(rhymed_lines)


def classify_rhyme_scheme(
    scores: dict[str, tuple[float, int, int]], config: dict[str, Any]
) -> str:
    eligible = [
        scheme
        for scheme in RHYME_SCHEMES[:3]
        if scores[scheme][0] >= config["minimum_scheme_score"]
        and scores[scheme][1] >= config["minimum_matched_lines"][scheme]
    ]
    if not eligible:
        return RHYME_SCHEMES[-1]
    return max(
        eligible,
        key=lambda scheme: (
            scores[scheme][0],
            scores[scheme][1],
            -RHYME_SCHEMES.index(scheme),
        ),
    )


def estimate_syllables(word: str) -> int:
    normalized = re.sub(r"[^a-z]", "", word.lower())
    count = len(VOWEL_GROUP_PATTERN.findall(normalized))
    if count > 1 and normalized.endswith("e") and not normalized.endswith(("le", "ye")):
        count -= 1
    return max(count, 1)


@lru_cache(maxsize=None)
def get_stress_options(word: str) -> tuple[str, ...]:
    options = {
        pronouncing.stresses(phones).replace("2", "1") for phones in get_phones(word)
    }
    return tuple(sorted(option for option in options if option))


def compute_edit_distance(left: str, right: str) -> float:
    previous = [float(index) for index in range(len(right) + 1)]
    for left_index, left_value in enumerate(left, start=1):
        current = [float(left_index)]
        for right_index, right_value in enumerate(right, start=1):
            substitution = 0.0 if left_value == right_value else 1.0
            if "?" in (left_value, right_value):
                substitution = 0.5
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution,
                )
            )
        previous = current
    return previous[-1]


def score_stress_pattern(pattern: str, target: str) -> float:
    denominator = max(len(pattern), len(target))
    if denominator == 0:
        return 0.0
    return max(0.0, 1 - compute_edit_distance(pattern, target) / denominator)


def prune_stress_candidates(
    candidates: Iterable[str], target: str, beam_size: int
) -> set[str]:
    unique = set(candidates)
    return set(
        sorted(
            unique,
            key=lambda value: (
                compute_edit_distance(value, target[: len(value)]),
                abs(len(value) - len(target)),
            ),
        )[:beam_size]
    )


def analyze_meter_line(line: str, config: dict[str, Any]) -> dict[str, Any]:
    words = WORD_PATTERN.findall(line)
    candidates = {""}
    known_words = 0
    for word in words:
        options = get_stress_options(word)
        if options:
            known_words += 1
        else:
            options = ("?" * estimate_syllables(word),)
        candidates = prune_stress_candidates(
            (prefix + option for prefix in candidates for option in options),
            config["target_stress_pattern"],
            config["beam_size"],
        )
    pattern = max(
        candidates,
        key=lambda value: (
            score_stress_pattern(value, config["target_stress_pattern"]),
            -abs(len(value) - len(config["target_stress_pattern"])),
        ),
    )
    coverage = known_words / len(words) if words else 0.0
    similarity = score_stress_pattern(pattern, config["target_stress_pattern"])
    is_scorable = coverage >= config["minimum_pronunciation_coverage"]
    is_iambic = (
        is_scorable
        and config["minimum_syllables"] <= len(pattern) <= config["maximum_syllables"]
        and similarity >= config["minimum_pattern_similarity"]
    )
    return {
        "pattern_similarity": similarity,
        "syllable_count": len(pattern),
        "pronunciation_coverage": coverage,
        "is_scorable": is_scorable,
        "is_iambic_pentameter": is_iambic,
    }


def compute_poetic_form_metrics(
    responses: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    poem_records = []
    stanza_records = []
    for row in tqdm(
        responses.itertuples(index=False),
        total=len(responses),
        desc="Analyzing poetic form",
        unit="poem",
    ):
        stanzas = split_stanzas(row.response)
        stanza_lengths = [len(lines) for lines in stanzas]
        lines = [line for stanza in stanzas for line in stanza]
        rhyme_scores = {
            scheme: score_rhyme_scheme(stanzas, cycle_length)
            for scheme, cycle_length in zip(RHYME_SCHEMES[:3], (1, 2, 3), strict=True)
        }
        rhyme_scheme = classify_rhyme_scheme(rhyme_scores, config["rhyme"])
        rhymed_line_count = count_rhymed_lines(
            stanzas, config["rhyme"]["maximum_other_rhyme_distance"]
        )
        end_words = [get_line_end_word(line) for line in lines]
        known_end_words = sum(bool(get_phones(word)) for word in end_words)
        meter_lines = [analyze_meter_line(line, config["meter"]) for line in lines]
        scorable_meter_lines = [line for line in meter_lines if line["is_scorable"]]
        iambic_line_count = sum(
            line["is_iambic_pentameter"] for line in scorable_meter_lines
        )
        iambic_line_fraction = (
            iambic_line_count / len(scorable_meter_lines)
            if scorable_meter_lines
            else 0.0
        )
        record = {
            "poem_id": row.poem_id,
            "generation_id": row.generation_id,
            "topic_id": row.topic_id,
            "topic": row.topic,
            "thinking_level": row.thinking_level,
            "line_count": len(lines),
            "stanza_count": len(stanzas),
            "mean_stanza_length": np.mean(stanza_lengths) if stanza_lengths else 0.0,
            "four_line_stanza_count": stanza_lengths.count(4),
            "rhyme_scheme": rhyme_scheme,
            "rhymed_line_count": rhymed_line_count,
            "rhymed_line_fraction": rhymed_line_count / len(lines) if lines else 0.0,
            "known_end_word_count": known_end_words,
            "end_word_pronunciation_coverage": (
                known_end_words / len(lines) if lines else 0.0
            ),
            "scorable_meter_line_count": len(scorable_meter_lines),
            "iambic_pentameter_line_count": iambic_line_count,
            "iambic_pentameter_line_fraction": iambic_line_fraction,
            "dominant_iambic_pentameter": (
                bool(scorable_meter_lines)
                and iambic_line_fraction
                >= config["meter"]["minimum_poem_line_fraction"]
            ),
            "iambic_alignment_score_sum": sum(
                line["pattern_similarity"] for line in scorable_meter_lines
            ),
            "mean_iambic_alignment_score": (
                np.mean([line["pattern_similarity"] for line in scorable_meter_lines])
                if scorable_meter_lines
                else 0.0
            ),
        }
        for scheme in RHYME_SCHEMES[:3]:
            score, matched, eligible = rhyme_scores[scheme]
            key = scheme.lower()
            record[f"{key}_score"] = score
            record[f"{key}_matched_line_count"] = matched
            record[f"{key}_eligible_line_count"] = eligible
        poem_records.append(record)
        for stanza_index, stanza_length in enumerate(stanza_lengths, start=1):
            stanza_records.append(
                {
                    "poem_id": row.poem_id,
                    "topic_id": row.topic_id,
                    "topic": row.topic,
                    "thinking_level": row.thinking_level,
                    "stanza_index": stanza_index,
                    "stanza_length": stanza_length,
                }
            )
    return pd.DataFrame(poem_records), pd.DataFrame(stanza_records)


def iter_thinking_groups(frame: pd.DataFrame):
    yield "all", frame
    for thinking_level, group in frame.groupby("thinking_level", sort=True):
        yield thinking_level, group


def summarize_metrics(frame: pd.DataFrame) -> dict[str, Any]:
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
        "mean_stanza_length": total_lines / total_stanzas,
        "four_line_stanza_fraction": frame["four_line_stanza_count"].sum()
        / total_stanzas,
        "rhymed_line_fraction": frame["rhymed_line_count"].sum() / total_lines,
        "end_word_pronunciation_coverage": frame["known_end_word_count"].sum()
        / total_lines,
        "iambic_pentameter_line_fraction": frame[
            "iambic_pentameter_line_count"
        ].sum()
        / total_meter_lines,
        "dominant_iambic_pentameter_poem_fraction": frame[
            "dominant_iambic_pentameter"
        ].mean(),
        "mean_iambic_alignment_score": frame["iambic_alignment_score_sum"].sum()
        / total_meter_lines,
    }


def create_summary_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = [
        {"thinking_level": level, **summarize_metrics(group)}
        for level, group in iter_thinking_groups(metrics)
    ]
    by_topic = []
    for (topic_id, topic), group in metrics.groupby(["topic_id", "topic"], sort=True):
        for level, subset in iter_thinking_groups(group):
            by_topic.append(
                {
                    "topic_id": topic_id,
                    "topic": topic,
                    "thinking_level": level,
                    **summarize_metrics(subset),
                }
            )
    return pd.DataFrame(overall), pd.DataFrame(by_topic)


def create_distribution(
    frame: pd.DataFrame, value_column: str, value_name: str
) -> pd.DataFrame:
    records = []
    for level, group in iter_thinking_groups(frame):
        counts = group[value_column].value_counts().sort_index()
        for value, count in counts.items():
            records.append(
                {
                    "thinking_level": level,
                    value_name: value,
                    "count": count,
                    "fraction": count / len(group),
                }
            )
    return pd.DataFrame(records)


def create_topic_distribution(
    frame: pd.DataFrame, value_column: str, value_name: str
) -> pd.DataFrame:
    records = []
    for (topic_id, topic), group in frame.groupby(["topic_id", "topic"], sort=True):
        for level, subset in iter_thinking_groups(group):
            counts = subset[value_column].value_counts().sort_index()
            for value, count in counts.items():
                records.append(
                    {
                        "topic_id": topic_id,
                        "topic": topic,
                        "thinking_level": level,
                        value_name: value,
                        "count": count,
                        "fraction": count / len(subset),
                    }
                )
    return pd.DataFrame(records)


def create_rhyme_distributions(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_overall = create_distribution(metrics, "rhyme_scheme", "rhyme_scheme")
    thinking_levels = ["all", *sorted(metrics["thinking_level"].unique())]
    overall_index = pd.MultiIndex.from_product(
        [thinking_levels, RHYME_SCHEMES],
        names=["thinking_level", "rhyme_scheme"],
    )
    overall = (
        observed_overall.set_index(["thinking_level", "rhyme_scheme"])
        .reindex(overall_index, fill_value=0)
        .reset_index()
    )
    observed_by_topic = create_topic_distribution(
        metrics, "rhyme_scheme", "rhyme_scheme"
    )
    full_index = pd.MultiIndex.from_product(
        [metrics["topic_id"].unique(), thinking_levels, RHYME_SCHEMES],
        names=["topic_id", "thinking_level", "rhyme_scheme"],
    )
    topic_names = metrics[["topic_id", "topic"]].drop_duplicates().set_index("topic_id")
    by_topic = (
        observed_by_topic.set_index(["topic_id", "thinking_level", "rhyme_scheme"])[
            ["count", "fraction"]
        ]
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    by_topic.insert(1, "topic", by_topic["topic_id"].map(topic_names["topic"]))
    mean_topics = (
        by_topic.groupby(["thinking_level", "rhyme_scheme"], sort=False)["fraction"]
        .agg(
            n_topics="size",
            mean_topic_fraction="mean",
            std_topic_fraction="std",
            min_topic_fraction="min",
            max_topic_fraction="max",
        )
        .reset_index()
    )
    return overall, by_topic, mean_topics


def create_homogeneity_comparisons(
    config: dict[str, Any],
    summary: pd.DataFrame,
    summary_by_topic: pd.DataFrame,
    rhyme_distribution: pd.DataFrame,
    rhyme_by_topic: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    similarity_path = Path(config["similarity_comparison_csv"])
    topic_similarity_path = Path(config["topic_homogeneity_csv"])
    if similarity_path.exists() and topic_similarity_path.exists():
        similarity = pd.read_csv(similarity_path)
        topic_similarity = pd.read_csv(topic_similarity_path)
        for row in similarity.itertuples(index=False):
            topic_values = topic_similarity[f"mean_{row.metric}"]
            records.append(
                {
                    "metric_family": "pair_similarity",
                    "metric": row.metric,
                    "thinking_level": "all",
                    "unit": "similarity",
                    "corpus_value": row.corpus_random_pair_mean,
                    "mean_topic_value": row.mean_topic_homogeneity,
                    "topic_std": topic_values.std(),
                    "topic_min": topic_values.min(),
                    "topic_max": topic_values.max(),
                    "n_topics": len(topic_values),
                }
            )

    units = {
        "mean_line_count": "lines_per_poem",
        "mean_stanza_count": "stanzas_per_poem",
        "mean_stanza_length": "lines_per_stanza",
        "four_line_stanza_fraction": "fraction",
        "rhymed_line_fraction": "fraction",
        "iambic_pentameter_line_fraction": "fraction",
        "dominant_iambic_pentameter_poem_fraction": "fraction",
        "mean_iambic_alignment_score": "similarity",
    }
    for row in summary.itertuples(index=False):
        topics = summary_by_topic[
            summary_by_topic["thinking_level"] == row.thinking_level
        ]
        for metric, unit in units.items():
            values = topics[metric]
            records.append(
                {
                    "metric_family": "poetic_form",
                    "metric": metric,
                    "thinking_level": row.thinking_level,
                    "unit": unit,
                    "corpus_value": getattr(row, metric),
                    "mean_topic_value": values.mean(),
                    "topic_std": values.std(),
                    "topic_min": values.min(),
                    "topic_max": values.max(),
                    "n_topics": len(values),
                }
            )

    for row in rhyme_distribution.itertuples(index=False):
        values = rhyme_by_topic[
            (rhyme_by_topic["thinking_level"] == row.thinking_level)
            & (rhyme_by_topic["rhyme_scheme"] == row.rhyme_scheme)
        ]["fraction"]
        scheme_name = re.sub(r"[^a-z]+", "_", row.rhyme_scheme.lower()).strip("_")
        records.append(
            {
                "metric_family": "rhyme_scheme",
                "metric": f"rhyme_{scheme_name}_poem_fraction",
                "thinking_level": row.thinking_level,
                "unit": "fraction",
                "corpus_value": row.fraction,
                "mean_topic_value": values.mean(),
                "topic_std": values.std(),
                "topic_min": values.min(),
                "topic_max": values.max(),
                "n_topics": summary_by_topic["topic_id"].nunique(),
            }
        )
    return pd.DataFrame(records)


def analyze_poetic_form(config_path: Path) -> dict[str, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    responses = load_embedding_responses(Path(config["responses_csv"]))
    metrics, stanzas = compute_poetic_form_metrics(responses, config)
    summary, summary_by_topic = create_summary_tables(metrics)
    rhyme, rhyme_by_topic, rhyme_mean_topics = create_rhyme_distributions(metrics)
    tables = {
        "poem_metrics": metrics,
        "poetic_form_summary": summary,
        "poetic_form_summary_by_topic": summary_by_topic,
        "rhyme_scheme_distribution": rhyme,
        "rhyme_scheme_distribution_by_topic": rhyme_by_topic,
        "rhyme_scheme_mean_topic_percentages": rhyme_mean_topics,
        "line_count_distribution": create_distribution(
            metrics, "line_count", "line_count"
        ),
        "line_count_distribution_by_topic": create_topic_distribution(
            metrics, "line_count", "line_count"
        ),
        "stanza_count_distribution": create_distribution(
            metrics, "stanza_count", "stanza_count"
        ),
        "stanza_count_distribution_by_topic": create_topic_distribution(
            metrics, "stanza_count", "stanza_count"
        ),
        "stanza_length_distribution": create_distribution(
            stanzas, "stanza_length", "stanza_length"
        ),
        "stanza_length_distribution_by_topic": create_topic_distribution(
            stanzas, "stanza_length", "stanza_length"
        ),
    }
    tables["homogeneity_comparisons"] = create_homogeneity_comparisons(
        config, summary, summary_by_topic, rhyme, rhyme_by_topic
    )
    for name, table in tables.items():
        output_path = Path(config["outputs"][name])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False, float_format="%.6f")
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze rhyme, meter, lines, and stanzas in basic poems"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = analyze_poetic_form(parse_args().config)
    print(f"Wrote {len(outputs)} poetic-form tables")
