from pathlib import Path

import pandas as pd

from experiments.multimodal_thinking_experiment.analyze_comma_lists import (
    analyze_three_comma_lines,
    find_three_comma_lines,
)


def test_find_three_comma_lines_requires_exactly_three_commas() -> None:
    text = (
        "No need for throat, or tongue, or word,\n"
        "One, two, three\n"
        "One, two, three, four,"
    )

    assert find_three_comma_lines(text) == [
        (1, "No need for throat, or tongue, or word,")
    ]


def test_analysis_writes_occurrences_and_per_poem_counts(tmp_path: Path) -> None:
    responses = pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "generation_id": "g1",
                "topic": "Language",
                "thinking_level": "minimal",
                "source_sample_idx": 1,
                "response": (
                    "No need for throat, or tongue, or word,\n"
                    "A plain second line"
                ),
            },
            {
                "poem_id": "p2",
                "generation_id": "g2",
                "topic": "Language",
                "thinking_level": "high",
                "source_sample_idx": 1,
                "response": "One, two, three\nAnother line",
            },
        ]
    )
    responses_csv = tmp_path / "responses.csv"
    responses.to_csv(responses_csv, index=False)
    output_paths = [tmp_path / f"output_{index}.csv" for index in range(4)]

    tables = analyze_three_comma_lines(responses_csv, *output_paths)

    assert len(tables["occurrences"]) == 1
    assert tables["metrics"]["three_comma_line_count"].tolist() == [1, 0]
    assert tables["summary"].loc[0, "total_three_comma_lines"] == 1
    assert tables["summary"].loc[0, "mean_three_comma_lines_per_poem"] == 0.5
    assert tables["distribution"]["poem_count"].tolist() == [1, 1]
    assert all(path.exists() for path in output_paths)
