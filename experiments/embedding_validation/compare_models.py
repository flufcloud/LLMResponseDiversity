from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results/embedding_validation")
TABLE_FILES = {
    "cross_lingual": (
        "cross_lingual_mean_similarity_by_query.csv",
        "cross_lingual_mean_similarity_by_lang_pair.csv",
        "cross_lingual_mean_similarity_by_lang.csv",
    ),
    "intralingual": (
        "intralingual_embedding_scores_mean.csv",
        "intralingual_embedding_scores_mean_by_query.csv",
    ),
}


def merge_model_tables(models_dir: Path, table_name: str) -> pd.DataFrame:
    frames = []
    for model_dir in sorted(path for path in models_dir.iterdir() if path.is_dir()):
        table_path = model_dir / "tables" / table_name
        if not table_path.exists():
            continue
        frame = pd.read_csv(table_path)
        score_column = frame.columns[-1]
        frame = frame.rename(
            columns={score_column: f"{model_dir.name}_similarity"}
        )
        frames.append(frame)

    combined = frames[0]
    key_columns = combined.columns[:-1].tolist()
    for frame in frames[1:]:
        combined = combined.merge(frame, on=key_columns, how="outer")
    return combined


def write_comparison_tables(results_dir: Path) -> dict[str, Path]:
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for validation_name, table_names in TABLE_FILES.items():
        models_dir = results_dir / validation_name / "models"
        for table_name in table_names:
            combined = merge_model_tables(models_dir, table_name)
            output = tables_dir / table_name
            combined.to_csv(output, index=False)
            outputs[table_name] = output
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine embedding validation mean tables"
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    for name, path in write_comparison_tables(Path(parse_args().results_dir)).items():
        print(f"{name}: {path}")
