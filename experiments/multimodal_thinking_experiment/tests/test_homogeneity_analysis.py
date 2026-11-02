import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.multimodal_thinking_experiment.analyze_homogeneity import (
    analyze_homogeneity,
    get_within_topic_pairs,
    sample_random_pairs,
)
from lib.measure_syntactic_similarity import (
    compute_normalized_wl_kernel,
    compute_wl_features,
    parse_poem_line_syntax,
)


def make_stanza_sentence(labels: list[str]) -> SimpleNamespace:
    words = [
        SimpleNamespace(
            id=index,
            head=0 if index == 1 else index - 1,
            upos=label,
        )
        for index, label in enumerate(labels, start=1)
    ]
    return SimpleNamespace(words=words)


class FakeSpacy:
    def pipe(self, responses, batch_size):
        assert batch_size == 8
        for response in responses:
            yield [
                SimpleNamespace(
                    lower_=word,
                    lemma_=word,
                    pos_="NOUN",
                    is_stop=False,
                    is_alpha=True,
                )
                for word in response.split()
            ]


class FakeStanza:
    def __call__(self, response):
        labels = ["NOUN"] * len(response.split())
        return SimpleNamespace(sentences=[make_stanza_sentence(labels)])


def test_poem_syntax_parses_each_nonempty_line_independently() -> None:
    pipeline = FakeStanza()

    parsed = parse_poem_line_syntax(
        ["First line,\nSecond line,\n\nThird line."], pipeline, 2
    )

    assert len(parsed) == 1
    assert len(parsed[0]) == 3


def write_homogeneity_config(tmp_path: Path) -> Path:
    responses_path = tmp_path / "responses.csv"
    pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "generation_id": "g1",
                "topic": "Love",
                "thinking_level": "minimal",
                "response": "sun",
            },
            {
                "poem_id": "p2",
                "generation_id": "g2",
                "topic": "Love",
                "thinking_level": "high",
                "response": "sun moon",
            },
            {
                "poem_id": "p3",
                "generation_id": "g3",
                "topic": "Nature",
                "thinking_level": "minimal",
                "response": "rain",
            },
            {
                "poem_id": "p4",
                "generation_id": "g4",
                "topic": "Nature",
                "thinking_level": "high",
                "response": "rain cloud",
            },
        ]
    ).to_csv(responses_path, index=False)
    lexical_path = tmp_path / "lexical.json"
    lexical_path.write_text(
        json.dumps(
            {
                "batch_size": 8,
                "content_pos": ["NOUN"],
                "excluded_terms": [],
            }
        ),
        encoding="utf-8",
    )
    output_names = (
        "random_pair_scores",
        "within_topic_pair_scores",
        "corpus_homogeneity",
        "topic_homogeneity",
        "homogeneity_comparison",
    )
    config_path = tmp_path / "homogeneity.json"
    config_path.write_text(
        json.dumps(
            {
                "responses_csv": str(responses_path),
                "embeddings_sqlite": str(tmp_path / "embeddings.sqlite"),
                "embedding_config": str(tmp_path / "embedding.json"),
                "lexical_config": str(lexical_path),
                "corpus_random_pairs": 3,
                "random_seed": 42,
                "syntax": {"wl_iterations": 2},
                "outputs": {
                    name: str(tmp_path / f"{name}.csv") for name in output_names
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_wl_kernel_is_one_for_identical_dependency_graphs() -> None:
    left = compute_wl_features(make_stanza_sentence(["NOUN", "VERB"]), 2)
    same = compute_wl_features(make_stanza_sentence(["NOUN", "VERB"]), 2)
    different = compute_wl_features(make_stanza_sentence(["ADJ", "NOUN"]), 2)

    assert compute_normalized_wl_kernel(left, same) == pytest.approx(1.0)
    assert 0 <= compute_normalized_wl_kernel(left, different) < 1


def test_pair_selection_is_unique_and_excludes_self_pairs() -> None:
    random_pairs = sample_random_pairs(20, 100, 42)
    within_pairs = get_within_topic_pairs(
        pd.DataFrame({"topic": ["Love"] * 20})
    )

    assert len(random_pairs) == len(set(random_pairs)) == 100
    assert all(left < right for left, right in random_pairs)
    assert len(within_pairs) == 190


def test_homogeneity_analysis_writes_pair_and_summary_tables(tmp_path: Path) -> None:
    embeddings = {
        "p1": np.array([1.0, 0.0]),
        "p2": np.array([1.0, 0.0]),
        "p3": np.array([0.0, 1.0]),
        "p4": np.array([0.0, 1.0]),
    }
    config_path = write_homogeneity_config(tmp_path)

    tables = analyze_homogeneity(
        config_path,
        spacy_nlp=FakeSpacy(),
        stanza_nlp=FakeStanza(),
        embeddings=embeddings,
    )

    assert len(tables["random_pair_scores"]) == 3
    assert len(tables["within_topic_pair_scores"]) == 2
    assert set(tables["topic_homogeneity"]["topic"]) == {"Love", "Nature"}
    assert set(tables["topic_homogeneity"]["n_pairs"]) == {1}
    assert set(tables["topic_homogeneity"]["mean_lexical_jaccard"]) == {0.5}
    assert set(tables["topic_homogeneity"]["mean_cosine_similarity"]) == {1.0}
    assert "mean_syntactic_diversity" not in tables["topic_homogeneity"].columns
    assert "syntactic_diversity" not in tables["within_topic_pair_scores"].columns
    comparison = tables["homogeneity_comparison"].set_index("metric")
    assert comparison.loc["lexical_jaccard", "mean_topic_homogeneity"] == 0.5
    assert "within_topic_minus_corpus" not in comparison.columns
    assert "syntactic_diversity" not in comparison.index
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert all(Path(path).exists() for path in config["outputs"].values())
