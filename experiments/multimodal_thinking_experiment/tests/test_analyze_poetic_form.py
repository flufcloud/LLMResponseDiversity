import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.multimodal_thinking_experiment.analyze_poetic_form import (
    RHYME_SCHEMES,
    analyze_poetic_form,
    classify_rhyme_scheme,
    compute_poetic_form_metrics,
    score_rhyme_scheme,
    score_stress_pattern,
)
from experiments.multimodal_thinking_experiment.analyze import split_stanzas


def make_config(tmp_path: Path) -> dict:
    output_names = (
        "poem_metrics",
        "poetic_form_summary",
        "poetic_form_summary_by_topic",
        "rhyme_scheme_distribution",
        "rhyme_scheme_distribution_by_topic",
        "rhyme_scheme_mean_topic_percentages",
        "line_count_distribution",
        "line_count_distribution_by_topic",
        "stanza_count_distribution",
        "stanza_count_distribution_by_topic",
        "stanza_length_distribution",
        "stanza_length_distribution_by_topic",
        "homogeneity_comparisons",
    )
    return {
        "responses_csv": str(tmp_path / "responses.csv"),
        "similarity_comparison_csv": str(tmp_path / "similarity.csv"),
        "topic_homogeneity_csv": str(tmp_path / "topic_homogeneity.csv"),
        "rhyme": {
            "minimum_scheme_score": 0.6,
            "minimum_matched_lines": {"AA": 2, "AB": 4, "ABC": 6},
            "maximum_other_rhyme_distance": 3,
        },
        "meter": {
            "target_stress_pattern": "0101010101",
            "minimum_pattern_similarity": 0.8,
            "minimum_pronunciation_coverage": 0.8,
            "minimum_poem_line_fraction": 0.5,
            "minimum_syllables": 9,
            "maximum_syllables": 11,
            "beam_size": 64,
        },
        "outputs": {name: str(tmp_path / f"{name}.csv") for name in output_names},
    }


@pytest.mark.parametrize(
    ("poem", "expected"),
    [
        ("night\nlight\nsea\nme", "AA"),
        ("night\nsea\nlight\nme", "AB"),
        ("night\nsea\nmoon\nlight\nme\ntune", "ABC"),
        ("night\nsea\nmoon\nday", "Other / non-rhyming"),
    ],
)
def test_classifies_variable_length_rhyme_families(
    tmp_path: Path, poem: str, expected: str
) -> None:
    config = make_config(tmp_path)
    stanzas = split_stanzas(poem)
    scores = {
        scheme: score_rhyme_scheme(stanzas, cycle)
        for scheme, cycle in zip(RHYME_SCHEMES[:3], (1, 2, 3), strict=True)
    }

    assert classify_rhyme_scheme(scores, config["rhyme"]) == expected


def test_stress_pattern_similarity_handles_exact_and_near_matches() -> None:
    assert score_stress_pattern("0101010101", "0101010101") == 1.0
    assert score_stress_pattern("0101010100", "0101010101") == 0.9
    assert score_stress_pattern("010101010", "0101010101") == 0.9


def test_analysis_writes_row_topic_and_corpus_tables(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    responses = pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "generation_id": "g1",
                "method_id": "poetry_basic",
                "topic_id": "t1",
                "topic": "Love",
                "thinking_level": "minimal",
                "source_sample_idx": 1,
                "response_idx": 1,
                "response": "night\nlight\n\nsea\nme",
            },
            {
                "poem_id": "p2",
                "generation_id": "g2",
                "method_id": "poetry_basic",
                "topic_id": "t1",
                "topic": "Love",
                "thinking_level": "high",
                "source_sample_idx": 1,
                "response_idx": 1,
                "response": "night\nsea\nlight\nme",
            },
            {
                "poem_id": "p3",
                "generation_id": "g3",
                "method_id": "poetry_basic",
                "topic_id": "t2",
                "topic": "Nature",
                "thinking_level": "minimal",
                "source_sample_idx": 1,
                "response_idx": 1,
                "response": "night\nsea\nmoon\nlight\nme\ntune",
            },
            {
                "poem_id": "p4",
                "generation_id": "g4",
                "method_id": "poetry_basic",
                "topic_id": "t2",
                "topic": "Nature",
                "thinking_level": "high",
                "source_sample_idx": 1,
                "response_idx": 1,
                "response": "red\nblue\ngreen\ngold",
            },
        ]
    )
    responses.to_csv(config["responses_csv"], index=False)
    pd.DataFrame(
        [
            {
                "metric": "cosine_similarity",
                "corpus_random_pair_mean": 0.4,
                "mean_topic_homogeneity": 0.5,
            }
        ]
    ).to_csv(config["similarity_comparison_csv"], index=False)
    pd.DataFrame(
        [
            {"topic": "Love", "mean_cosine_similarity": 0.4},
            {"topic": "Nature", "mean_cosine_similarity": 0.6},
        ]
    ).to_csv(config["topic_homogeneity_csv"], index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outputs = analyze_poetic_form(config_path)

    assert len(outputs["poem_metrics"]) == 4
    assert set(outputs["poem_metrics"]["rhyme_scheme"]) == set(RHYME_SCHEMES)
    assert set(outputs["poetic_form_summary"]["thinking_level"]) == {
        "all",
        "high",
        "minimal",
    }
    assert len(outputs["poetic_form_summary_by_topic"]) == 6
    comparison = outputs["homogeneity_comparisons"]
    assert "cosine_similarity" in set(comparison["metric"])
    assert "rhyme_aa_poem_fraction" in set(comparison["metric"])
    assert all(Path(path).exists() for path in config["outputs"].values())


def test_poem_metrics_record_stanza_and_meter_fields(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    responses = pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "generation_id": "g1",
                "topic_id": "t1",
                "topic": "Love",
                "thinking_level": "minimal",
                "response": "Shall I compare thee to a summer's day\nnight",
            }
        ]
    )

    metrics, stanzas = compute_poetic_form_metrics(responses, config)

    assert metrics.loc[0, "line_count"] == 2
    assert metrics.loc[0, "stanza_count"] == 1
    assert len(stanzas) == 1
    assert stanzas.loc[0, "stanza_length"] == 2
    assert 0 <= metrics.loc[0, "mean_iambic_alignment_score"] <= 1
