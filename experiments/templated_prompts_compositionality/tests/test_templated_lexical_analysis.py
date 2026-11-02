import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from experiments.templated_prompts_compositionality.browse_word_frequencies import (
    get_adjective_classes,
    get_adjectives,
    get_topics,
    load_frequency_tables,
    select_frequencies,
)
from experiments.templated_prompts_compositionality.compare_topic_lexical_diversity import (
    compare_topic_lexical_diversity,
)
from experiments.templated_prompts_compositionality.analyze_words import (
    analyze_words,
)
from experiments.templated_prompts_compositionality.plot_word_frequencies import (
    load_word_frequencies,
    plot_frequency_bars,
    plot_word_frequencies,
    select_example_words,
)
from experiments.templated_prompts_compositionality.show_top_words import (
    show_top_words,
)
from experiments.templated_prompts_compositionality.summarize_lemmas import (
    summarize_lemmas as summarize_compositional_lemmas,
)
from lib.plot_word_frequencies import plot_top_words, select_top_words


def make_token(text: str, lemma: str, pos: str) -> SimpleNamespace:
    return SimpleNamespace(
        lower_=text,
        lemma_=lemma,
        pos_=pos,
        is_stop=False,
        is_alpha=True,
    )


class FakeNlp:
    def pipe(self, responses, batch_size):
        assert len(list(responses)) == 4
        assert batch_size == 8
        return iter(
            [
                [
                    make_token("dreams", "dream", "NOUN"),
                    make_token("dreaming", "dream", "VERB"),
                    make_token("stars", "star", "NOUN"),
                ],
                [make_token("stars", "star", "NOUN")],
                [make_token("dream", "dream", "NOUN")],
                [make_token("dream", "dream", "NOUN")],
            ]
        )


class TextNlp:
    def pipe(self, responses, batch_size):
        for response in responses:
            words = response.split() if response in {"sun sun", "moon"} else ["sun"]
            yield [make_token(word, word, "NOUN") for word in words]


def write_config(tmp_path: Path) -> Path:
    input_path = tmp_path / "generations.csv"
    pd.DataFrame(
        [
            {
                "topic": "Love",
                "adjective_group": "happy",
                "adjective": "joyful",
                "response": "one",
            },
            {
                "topic": "Love",
                "adjective_group": "happy",
                "adjective": "joyful",
                "response": "two",
            },
            {
                "topic": "Love",
                "adjective_group": "happy",
                "adjective": "delightful",
                "response": "three",
            },
            {
                "topic": "Love",
                "adjective_group": "sad",
                "adjective": "somber",
                "response": "four",
            },
        ]
    ).to_csv(input_path, index=False)
    levels = [
        {
            "name": "topic",
            "group_columns": ["topic"],
            "output_csv": str(tmp_path / "topic.csv"),
            "summary_group_columns": ["topic"],
            "summary_output_csv": str(tmp_path / "topic_summary.csv"),
        },
        {
            "name": "topic and adjective class",
            "group_columns": ["topic", "adjective_class"],
            "output_csv": str(tmp_path / "class.csv"),
            "summary_group_columns": ["topic", "adjective_class"],
            "summary_output_csv": str(tmp_path / "class_summary.csv"),
        },
        {
            "name": "topic and adjective",
            "group_columns": ["topic", "adjective_class", "adjective"],
            "output_csv": str(tmp_path / "adjective.csv"),
            "summary_group_columns": ["topic", "adjective"],
            "summary_output_csv": str(tmp_path / "adjective_summary.csv"),
        },
    ]
    config_path = tmp_path / "lexical_config.json"
    config_path.write_text(
        json.dumps(
            {
                "input_csv": str(input_path),
                "spacy_model": "fake",
                "batch_size": 8,
                "content_pos": ["NOUN", "VERB"],
                "excluded_terms": [],
                "disabled_components": [],
                "levels": levels,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_analysis_writes_three_lemma_frequency_levels(tmp_path: Path) -> None:
    counts = analyze_words(write_config(tmp_path), nlp=FakeNlp())
    topic = pd.read_csv(tmp_path / "topic.csv").set_index("lemma")
    adjective_class = pd.read_csv(tmp_path / "class.csv").set_index(
        ["adjective_class", "lemma"]
    )
    adjective = pd.read_csv(tmp_path / "adjective.csv").set_index(
        ["adjective", "lemma"]
    )

    assert counts == {
        "topic": 2,
        "topic and adjective class": 3,
        "topic and adjective": 4,
    }
    assert topic.loc["dream", "count"] == 4
    assert topic.loc["dream", "decimal_frequency"] == 0.75
    assert adjective_class.loc[("happy", "dream"), "count"] == 3
    assert adjective_class.loc[("happy", "dream"), "decimal_frequency"] == 0.67
    assert adjective.loc[("joyful", "dream"), "count"] == 2
    assert adjective.loc[("joyful", "dream"), "decimal_frequency"] == 0.5


def test_summary_counts_unique_lemmas_at_three_levels(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    analyze_words(config_path, nlp=FakeNlp())

    counts = summarize_compositional_lemmas(config_path)
    topic = pd.read_csv(tmp_path / "topic_summary.csv")
    adjective_class = pd.read_csv(tmp_path / "class_summary.csv")
    adjective = pd.read_csv(tmp_path / "adjective_summary.csv")

    assert counts == {
        "topic": 1,
        "topic and adjective class": 2,
        "topic and adjective": 3,
    }
    assert topic.loc[0, "topic"] == "Love"
    assert topic.loc[0, "num_unique_lemmas"] == 2
    assert topic.loc[0, "entropy"] == pytest.approx(0.918296)
    assert list(adjective_class.columns) == [
        "topic",
        "adjective_class",
        "num_unique_lemmas",
        "entropy",
    ]
    assert list(adjective.columns) == [
        "topic",
        "adjective",
        "num_unique_lemmas",
        "entropy",
    ]


def test_matched_comparison_resamples_equal_poem_counts(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    basic_path = tmp_path / "basic.csv"
    output_path = tmp_path / "matched.csv"
    pd.DataFrame(
        [
            {"topic": "Love", "response": "sun sun"},
            {"topic": "Love", "response": "moon"},
        ]
    ).to_csv(basic_path, index=False)
    config["matched_topic_comparison"] = {
        "basic_input_csv": str(basic_path),
        "sample_size": 2,
        "num_resamples": 5,
        "confidence_level": 0.95,
        "random_seed": 0,
        "output_csv": str(output_path),
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = compare_topic_lexical_diversity(config_path, nlp=TextNlp())

    assert result.loc[0, "basic_num_unique_lemmas"] == 2
    assert result.loc[0, "adjective_mean_num_unique_lemmas"] == 1
    assert result.loc[0, "mean_num_unique_lemmas_difference"] == -1
    assert result.loc[0, "interval_level"] == 0.95
    assert result.loc[0, "basic_entropy"] == pytest.approx(0.918295834)
    assert result.loc[0, "adjective_mean_entropy"] == 0.0
    assert result.loc[0, "mean_entropy_difference"] == pytest.approx(-0.918295834)
    assert pd.read_csv(output_path).loc[0, "basic_entropy"] == pytest.approx(
        0.918296
    )


def test_plot_selects_requested_level_and_examples(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    analyze_words(config_path, nlp=FakeNlp())
    config = json.loads(config_path.read_text(encoding="utf-8"))

    selections = [
        (None, None, "topic"),
        ("happy", None, "topic and adjective class"),
        ("happy", "joyful", "topic and adjective"),
    ]
    for adjective_class, adjective, expected_level in selections:
        frequencies, level = load_word_frequencies(
            config,
            "Love",
            adjective_class,
            adjective,
        )
        assert level == expected_level

    examples = select_example_words(frequencies)
    figure = plot_frequency_bars(frequencies, "Love", "happy", "joyful")
    selected = plot_word_frequencies(
        config_path,
        "Love",
        adjective_class="happy",
        adjective="joyful",
        show_plot=False,
    )

    assert set(frequencies["lemma"]) == {"dream", "star"}
    assert examples[0][0] == "Most used"
    assert examples[0][1]["lemma"] == "star"
    axis = figure.axes[0]
    assert axis.get_xlabel() == "Content lemmas ranked by total occurrences"
    assert axis.get_ylabel() == "Total occurrences"
    assert len(axis.patches) == len(frequencies)
    assert list(axis.get_xticks()) == []
    assert axis.get_legend().get_texts()[0].get_text() == "Most used: star (2)"
    assert selected.equals(frequencies)
    plt.close(figure)


def test_plot_requires_adjective_class_for_adjective(tmp_path: Path) -> None:
    config = json.loads(write_config(tmp_path).read_text(encoding="utf-8"))

    with pytest.raises(
        ValueError,
        match="--adjective requires --adjective-class",
    ):
        load_word_frequencies(config, "Love", None, "joyful")


def test_top_words_table_uses_same_selection(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    analyze_words(config_path, nlp=FakeNlp())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frequencies, _ = load_word_frequencies(config, "Love", "happy", None)

    top_words = select_top_words(frequencies)
    figure = plot_top_words(top_words, "Most common content lemmas\nLove / happy")
    selected = show_top_words(
        config_path,
        "Love",
        adjective_class="happy",
        show_plot=False,
    )

    assert top_words.to_dict("records") == [
        {"rank": 1, "lemma": "dream", "count": 3, "decimal_frequency": 0.67},
        {"rank": 2, "lemma": "star", "count": 2, "decimal_frequency": 0.67},
    ]
    assert len(figure.axes[0].tables) == 1
    assert selected.equals(top_words)
    plt.close(figure)


def test_browser_switches_between_all_frequency_levels(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    analyze_words(config_path, nlp=FakeNlp())
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tables = load_frequency_tables(config)

    assert get_topics(tables) == ["Love"]
    assert get_adjective_classes(tables, "Love") == ["happy", "sad"]
    assert get_adjectives(tables, "Love", "happy") == ["delightful", "joyful"]
    assert len(select_frequencies(tables, "Love", None, None)) == 2
    assert len(select_frequencies(tables, "Love", "happy", None)) == 2
    assert len(select_frequencies(tables, "Love", "happy", "joyful")) == 2
