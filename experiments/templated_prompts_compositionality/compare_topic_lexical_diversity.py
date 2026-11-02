from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.multimodal_thinking_experiment.analyze_words import (
    extract_content_terms,
    load_spacy_model,
)
from lib.compare_lexical_diversity import (
    compare_matched_topic_metrics,
    group_poem_lemmas,
)


DEFAULT_CONFIG = Path(__file__).with_name("lexical_analysis_config.json")


def compare_topic_lexical_diversity(
    config_path: Path,
    nlp: Any | None = None,
) -> pd.DataFrame:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    comparison = config["matched_topic_comparison"]
    basic = pd.read_csv(comparison["basic_input_csv"], usecols=["topic", "response"])
    adjective = pd.read_csv(config["input_csv"], usecols=["topic", "response"])
    responses = pd.concat([basic["response"], adjective["response"]], ignore_index=True)
    nlp = nlp or load_spacy_model(config)
    _, _, lemma_terms = extract_content_terms(responses, nlp, config)
    basic_terms = group_poem_lemmas(basic, lemma_terms[: len(basic)])
    adjective_terms = group_poem_lemmas(adjective, lemma_terms[len(basic) :])

    result = compare_matched_topic_metrics(
        basic_terms,
        adjective_terms,
        sample_size=comparison["sample_size"],
        num_resamples=comparison["num_resamples"],
        interval_level=comparison["confidence_level"],
        random_seed=comparison["random_seed"],
        candidate_name="adjective",
    )
    output_path = Path(comparison["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.6f")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare topic lexical diversity using matched poem counts"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    comparison_table = compare_topic_lexical_diversity(parse_args().config)
    print(f"Wrote {len(comparison_table)} topic comparisons")
