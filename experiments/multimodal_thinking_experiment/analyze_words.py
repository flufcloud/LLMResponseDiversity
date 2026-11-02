from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from experiments.multimodal_thinking_experiment.prepare_poems import (
    DEFAULT_RESULTS_ROOT,
)


WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
DEFAULT_INPUT = DEFAULT_RESULTS_ROOT / "data" / "poetry_basic_generations.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "tables" / "poetry_basic" / "unique_words.csv"
DEFAULT_CONFIG = Path(__file__).with_name("lexical_analysis_config.json")


def extract_words(text: str) -> list[str]:
    words = WORD_PATTERN.findall(text)
    return [word.lower().replace("’", "'") for word in words]


def count_terms(
    poem_terms: Iterable[Iterable[str]],
    term_column: str,
) -> pd.DataFrame:
    total_counts: Counter[str] = Counter()
    poem_counts: Counter[str] = Counter()
    poem_total = 0
    for terms in poem_terms:
        normalized_terms = list(terms)
        total_counts.update(normalized_terms)
        poem_counts.update(set(normalized_terms))
        poem_total += 1

    records = [
        {
            term_column: term,
            "count": count,
            "decimal_frequency": round(poem_counts[term] / poem_total, 2),
        }
        for term, count in total_counts.items()
    ]
    return pd.DataFrame(records).sort_values(
        ["count", term_column], ascending=[False, True], ignore_index=True
    )


def count_unique_words(responses: pd.Series) -> pd.DataFrame:
    return count_terms((extract_words(response) for response in responses), "word")


def extract_content_terms(
    responses: pd.Series,
    nlp: Any,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[list[str]]]:
    surface_terms = []
    lemma_terms = []
    allowed_pos = set(config["content_pos"])
    excluded_terms = {
        term.lower().replace("’", "'") for term in config["excluded_terms"]
    }
    docs = nlp.pipe(responses, batch_size=config["batch_size"])
    for doc in tqdm(docs, total=len(responses), desc="Tagging poems", unit="poem"):
        retained = [
            token
            for token in doc
            if token.is_alpha
            and token.pos_ in allowed_pos
            and not token.is_stop
            and token.lower_.replace("’", "'") not in excluded_terms
        ]
        surface_terms.append(
            [token.lower_.replace("’", "'") for token in retained]
        )
        lemma_terms.append(
            [token.lemma_.lower().replace("’", "'") for token in retained]
        )
    return (
        count_terms(surface_terms, "word"),
        count_terms(lemma_terms, "lemma"),
        lemma_terms,
    )


def create_topic_usage_tables(
    generations: pd.DataFrame,
    lemma_terms: list[list[str]],
) -> dict[str, pd.DataFrame]:
    records = []
    columns = ["topic", "thinking_level"]
    for row, terms in zip(
        generations[columns].itertuples(index=False), lemma_terms, strict=True
    ):
        records.extend(
            {
                "word": word,
                "topic": row.topic,
                "thinking_level": row.thinking_level,
            }
            for word in set(terms)
        )

    presence = pd.DataFrame(records)
    usage = (
        presence.groupby(["thinking_level", "word", "topic"], sort=False)
        .size()
        .rename("num_usages")
        .reset_index()
    )
    return {
        thinking_level: group[["word", "topic", "num_usages"]]
        .sort_values(["word", "topic"], ignore_index=True)
        for thinking_level, group in usage.groupby("thinking_level", sort=False)
    }


def create_word_topics_table(
    generations: pd.DataFrame,
    lemma_terms: list[list[str]],
) -> pd.DataFrame:
    word_topics: dict[str, set[str]] = {}
    for topic, terms in zip(generations["topic"], lemma_terms, strict=True):
        for word in set(terms):
            word_topics.setdefault(word, set()).add(topic)

    records = [
        {
            "word": word,
            "topics": json.dumps(sorted(topics), ensure_ascii=False),
        }
        for word, topics in word_topics.items()
    ]
    return pd.DataFrame(records).sort_values("word", ignore_index=True)


def create_combined_topic_tables(
    topic_usage: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = (
        pd.concat(topic_usage.values(), ignore_index=True)
        .groupby(["word", "topic"], as_index=False, sort=False)["num_usages"]
        .sum()
        .rename(
            columns={"num_usages": "num_usages_within_topic_generations"}
        )
        .sort_values(["word", "topic"], ignore_index=True)
    )
    coverage = combined[["word", "topic"]].copy()
    coverage["num_topics_covered"] = coverage.groupby("word")["topic"].transform(
        "nunique"
    )
    return combined, coverage


def load_spacy_model(config: dict[str, Any]) -> Any:
    import spacy

    return spacy.load(config["spacy_model"], disable=config["disabled_components"])


def analyze_words(args: argparse.Namespace) -> None:
    generations = pd.read_csv(
        args.input, usecols=["topic", "thinking_level", "response"]
    )
    unique_words = count_unique_words(generations["response"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_words.to_csv(output_path, index=False, float_format="%.2f")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    content_words, content_lemmas, lemma_terms = extract_content_terms(
        generations["response"], load_spacy_model(config), config
    )
    content_words_path = Path(config["content_words_output"])
    content_lemmas_path = Path(config["content_lemmas_output"])
    content_words_path.parent.mkdir(parents=True, exist_ok=True)
    content_words.to_csv(content_words_path, index=False, float_format="%.2f")
    content_lemmas.to_csv(content_lemmas_path, index=False, float_format="%.2f")
    topic_usage = create_topic_usage_tables(generations, lemma_terms)
    for thinking_level, path in config["lemma_topic_usage_outputs"].items():
        topic_usage[thinking_level].to_csv(path, index=False)
    create_word_topics_table(generations, lemma_terms).to_csv(
        config["lemma_topics_output"], index=False
    )
    combined_usage, topic_coverage = create_combined_topic_tables(topic_usage)
    combined_usage.to_csv(config["lemma_topic_sample_usage_output"], index=False)
    topic_coverage.to_csv(config["lemma_topic_coverage_output"], index=False)
    print(f"Wrote lexical tables for {len(generations)} poems under {output_path.parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count words in basic poetry generations")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


if __name__ == "__main__":
    analyze_words(parse_args())
