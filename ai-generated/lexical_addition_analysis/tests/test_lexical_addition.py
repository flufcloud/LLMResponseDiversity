from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


analysis = importlib.import_module(
    "ai-generated.lexical_addition_analysis.analyze_lexical_addition"
)


def test_balanced_sample_uses_every_adjective() -> None:
    rows = []
    for adjective_index in range(15):
        for sample_index in range(3):
            rows.append(
                {
                    "adjective": f"a{adjective_index:02d}",
                    "poem_id": f"p{adjective_index:02d}-{sample_index}",
                }
            )
    frame = pd.DataFrame(rows)

    selected = analysis.sample_balanced_adjectives(
        frame, 20, np.random.default_rng(4)
    )
    counts = frame.loc[selected, "adjective"].value_counts()

    assert len(selected) == 20
    assert len(counts) == 15
    assert sorted(counts) == [1] * 10 + [2] * 5


def test_measure_order_counts_only_unseen_lemmas() -> None:
    lemma_sets = pd.Series(
        [
            frozenset({"sun", "light"}),
            frozenset({"light", "rain"}),
            frozenset({"sun"}),
        ]
    )

    result = analysis.measure_order(np.asarray([0, 1, 2]), lemma_sets)

    assert result["cumulative_lemmas"].tolist() == [2, 3, 3]
    assert result["new_lemmas"].tolist() == [2, 1, 0]
    assert result["new_lemma_rate"].tolist() == [1.0, 0.5, 0.0]
    assert result["no_new_lemmas"].tolist() == [0.0, 0.0, 1.0]


def test_repetition_summary_reports_generation_quantiles() -> None:
    values = {
        metric: np.asarray([[1.0, 2.0], [1.0, 4.0]])
        for metric in analysis.METRICS
    }

    result = analysis.summarize_repetitions(values, "basic", "Love")

    assert result["generation_number"].tolist() == [1, 2]
    assert result.loc[1, "mean_new_lemmas"] == 3.0
    assert result["topic"].eq("Love").all()
