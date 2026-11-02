from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.glasgow_concreteness_regression.prepare_data import (
    load_regression_settings,
    prepare_embedding_inputs,
    prepare_training_rows,
)
from lib.embed_locally import (
    LocalSentenceEncoder,
    fetch_local_embeddings,
    load_local_embedding_settings,
)
from lib.store_embeddings import EmbeddingCache


EXPERIMENT_DIR = Path("experiments/glasgow_concreteness_regression")
DEFAULT_CONFIG = EXPERIMENT_DIR / "experiment_config.json"


def embed_words(
    config_path: Path,
    results_dir: Path | None = None,
    encoder: Any | None = None,
    allow_model_download: bool = False,
) -> int:
    regression = load_regression_settings(config_path)
    embedding = load_local_embedding_settings(config_path)
    results_dir = results_dir or Path("results") / regression.experiment_id
    data_dir = results_dir / "data"
    for path in (data_dir, results_dir / "tables", results_dir / "plots"):
        path.mkdir(parents=True, exist_ok=True)

    rows = prepare_training_rows(regression)
    rows.to_csv(data_dir / "split_assignments.csv", index=False)
    inputs = prepare_embedding_inputs(rows)
    encoder = encoder or LocalSentenceEncoder(
        embedding,
        allow_model_download=allow_model_download,
    )
    with EmbeddingCache(data_dir / "embeddings.sqlite") as cache:
        fetch_local_embeddings(inputs, cache, encoder, embedding)

    metadata = {
        "experiment_id": regression.experiment_id,
        "embedding_model": embedding.model,
    }
    (results_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed plain Glasgow Norms words with a frozen local encoder"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    count = embed_words(
        arguments.config,
        allow_model_download=arguments.allow_model_download,
    )
    print(f"Embedded {count} Glasgow Norms words")
