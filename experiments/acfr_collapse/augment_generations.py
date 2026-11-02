from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_GENERATIONS = Path("results/acfr_collapse/data/generations.csv")
DEFAULT_RATINGS = Path("datasets/ACFnorms_TrochCrutchReilly_2014.csv")
DEFAULT_OUTPUT = Path("results/acfr_collapse/data/generations_augmented.csv")


def augment_generations(
    generations_path: Path,
    ratings_path: Path,
    output_path: Path,
) -> int:
    with ratings_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rating_columns = [column for column in reader.fieldnames if column != "Word"]
        ratings = {row["Word"]: row for row in reader}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with generations_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames + rating_columns)
        writer.writeheader()
        count = 0
        for row in reader:
            rating = ratings[row["word"]]
            writer.writerow(
                {
                    **row,
                    **{column: rating[column] for column in rating_columns},
                }
            )
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join ACFr ratings to generations")
    parser.add_argument("--generations", default=str(DEFAULT_GENERATIONS))
    parser.add_argument("--ratings", default=str(DEFAULT_RATINGS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    rows_written = augment_generations(
        Path(arguments.generations),
        Path(arguments.ratings),
        Path(arguments.output),
    )
    print(f"Wrote {rows_written} augmented rows to {arguments.output}")
