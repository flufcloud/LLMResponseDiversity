from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


analysis = importlib.import_module(
    "ai-generated.syntactic_form_analysis.analyze_syntactic_forms"
)
parser = importlib.import_module(
    "ai-generated.syntactic_form_analysis.parse_syntactic_forms"
)
selector = importlib.import_module(
    "ai-generated.syntactic_form_analysis.select_syntactic_examples"
)


def make_sentence(text: str, labels: list[str]) -> SimpleNamespace:
    words = [
        SimpleNamespace(
            id=index,
            head=0 if index == 1 else index - 1,
            upos=label,
        )
        for index, label in enumerate(labels, start=1)
    ]
    return SimpleNamespace(text=text, words=words)


def test_parse_line_features_writes_one_row_per_poetic_line(tmp_path: Path) -> None:
    input_path = tmp_path / "responses.csv"
    output_path = tmp_path / "line_features.jsonl"
    pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "generation_id": "g1",
                "topic": "Love",
                "thinking_level": "minimal",
                "source_sample_idx": 1,
                "response": "First. Second.\nThird continuation",
            }
        ]
    ).to_csv(input_path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "input_csv": str(input_path),
                "line_features_jsonl": str(output_path),
                "syntax": {"wl_iterations": 2},
            }
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def __call__(self, text):
            if text == "First. Second.":
                sentences = [
                    make_sentence("First.", ["NOUN"]),
                    make_sentence("Second.", ["VERB"]),
                ]
            else:
                assert text == "Third continuation"
                sentences = [make_sentence(text, ["ADJ", "NOUN"])]
            return SimpleNamespace(sentences=sentences)

    parser.parse_line_features(config_path, pipeline=FakePipeline())
    records = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert len(records) == 2
    assert [record["line_idx"] for record in records] == [1, 2]
    assert records[0]["line_text"] == "First. Second."
    assert records[0]["features"]["(0, 'NOUN')"] == 1
    assert records[0]["features"]["(0, 'VERB')"] == 1
    assert all(record["features"] for record in records)


def test_three_repeated_forms_plateau_at_three() -> None:
    lines = pd.DataFrame(
        [
            {"poem_id": f"p{index:03d}"}
            for index in range(300)
        ]
    )
    labels = np.repeat([1, 2, 3], 100)

    curve = analysis.summarize_accumulation(
        lines,
        labels,
        {f"p{index:03d}": 2 for index in range(300)},
        repetitions=100,
        random_seed=42,
    )
    summary = analysis.summarize_form_distribution(lines, labels)

    assert curve.iloc[-1]["mean_cumulative_forms"] == 3
    assert curve.iloc[-1]["mean_new_forms"] == 0
    assert curve.iloc[-1]["probability_no_new_forms"] == 1
    assert curve.iloc[0]["mean_new_forms_per_line"] == pytest.approx(0.5)
    assert summary["num_forms"] == 3
    assert summary["largest_form_line_share"] == pytest.approx(1 / 3)
    assert summary["effective_forms_shannon"] == pytest.approx(3)
    assert summary["form_line_ratio"] == pytest.approx(0.01)


def test_complete_linkage_separates_dissimilar_feature_groups() -> None:
    features = [
        Counter({"a": 2}),
        Counter({"a": 2}),
        Counter({"b": 2}),
        Counter({"b": 2}),
        Counter({"c": 2}),
        Counter({"c": 2}),
    ]
    matrix = analysis.compute_line_similarity_matrix(features)
    labels = analysis.cluster_line_forms(matrix, threshold=0.75)

    assert len(set(labels)) == 3
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[4] == labels[5]

    summary = analysis.summarize_similarity_matrix(matrix)
    assert summary["num_line_pairs"] == 15
    assert 0 <= summary["mean_similarity"] <= 1


def test_select_examples_returns_line_pairs_near_targets() -> None:
    lines = pd.DataFrame(
        [
            ("l1", "p1", "Love", "First", Counter({"a": 2})),
            ("l2", "p2", "Love", "Second", Counter({"a": 2})),
            ("l3", "p3", "Love", "Third", Counter({"b": 2})),
            ("l4", "p4", "Love", "Fourth", Counter({"b": 2})),
        ],
        columns=["line_id", "poem_id", "topic", "line_text", "features"],
    )

    examples = selector.select_examples(lines, [0.0, 1.0], 100, 3)

    assert examples["target_similarity"].tolist() == [0.0, 1.0]
    assert examples["syntactic_similarity"].tolist() == [0.0, 1.0]
    assert examples["topic"].tolist() == ["Love", "Love"]
    assert list(examples.columns) == [
        "target_similarity",
        "syntactic_similarity",
        "topic",
        "line_1",
        "line_2",
    ]


def test_dataset_summary_preserves_threshold_and_generation() -> None:
    distributions = pd.DataFrame(
        [
            {
                "topic": topic,
                "similarity_threshold": 0.75,
                "num_forms": forms,
                "singleton_form_fraction": 0.5,
                "largest_form_line_share": 0.2,
                "effective_forms_shannon": forms - 1,
                "form_line_ratio": 0.5,
                "effective_form_line_ratio": 0.4,
                "effective_forms_simpson": forms - 2,
                "mean_new_forms_at_final_generation": 0.1,
                "mean_new_forms_per_line_at_final_generation": 0.02,
                "probability_no_new_forms_at_final_generation": 0.9,
            }
            for topic, forms in [("Love", 10), ("Nature", 20)]
        ]
    )
    curves = pd.DataFrame(
        [
            {
                "topic": topic,
                "similarity_threshold": 0.75,
                "generation_number": generation,
                "mean_cumulative_forms": value,
                "mean_new_forms": value / 2,
                "mean_new_forms_per_line": value / 20,
                "probability_no_new_forms": 0.1,
            }
            for topic, generation, value in [
                ("Love", 1, 2),
                ("Nature", 1, 4),
                ("Love", 2, 3),
                ("Nature", 2, 5),
            ]
        ]
    )

    summary, curve = analysis.summarize_dataset(distributions, curves)

    assert summary.loc[0, "num_topics"] == 2
    assert summary.loc[0, "mean_num_forms"] == 15
    assert curve["generation_number"].tolist() == [1, 2]
    assert curve["mean_cumulative_forms"].tolist() == [3, 4]
