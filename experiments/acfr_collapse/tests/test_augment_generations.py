import csv
from pathlib import Path

from experiments.acfr_collapse.augment_generations import augment_generations


def test_augments_generations_with_all_rating_columns(tmp_path: Path) -> None:
    generations_path = tmp_path / "generations.csv"
    generations_path.write_text(
        "word,prompt,modality,sample_idx,response,thinking\n"
        "ability,Write about ability,text,1,Response,Thought\n",
        encoding="utf-8",
    )
    ratings_path = tmp_path / "ratings.csv"
    ratings_path.write_text(
        "Word,Emotion,Polarity,CNC\nability,4.52,4.85,273\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "generations_augmented.csv"

    count = augment_generations(generations_path, ratings_path, output_path)

    with output_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert count == 1
    assert list(rows[0]) == [
        "word",
        "prompt",
        "modality",
        "sample_idx",
        "response",
        "thinking",
        "Emotion",
        "Polarity",
        "CNC",
    ]
    assert rows[0]["Emotion"] == "4.52"
    assert rows[0]["CNC"] == "273"
