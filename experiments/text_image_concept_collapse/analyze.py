from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.text_image_concept_collapse.embed_data import (
    DATA_DIR,
    DEFAULT_CONFIG,
    EXPERIMENT_DIR,
    RESULTS_ROOT,
    compute_file_hash,
    get_vision_fingerprint,
    load_analysis_config,
    prepare_cross_modal_inputs,
    prepare_sentence_inputs,
)
from lib.compute_identifiers import compute_stable_hash
from lib.embed_locally import get_local_embedding_fingerprint, load_local_embedding_settings
from lib.store_embeddings import EmbeddingCache, load_response_embeddings
from lib.store_multimodal_embeddings import MultimodalEmbeddingStore


def compute_cross_modal_scores(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_embeddings: dict[str, np.ndarray],
    right_embeddings: dict[str, np.ndarray],
    left_sample_column: str,
    right_sample_column: str,
) -> pd.DataFrame:
    rows = []
    for query_id, left_group in left.groupby("query_id", sort=False):
        right_group = right.loc[right["query_id"] == query_id]
        for left_row in left_group.itertuples(index=False):
            for right_row in right_group.itertuples(index=False):
                left_vector = left_embeddings[left_row.input_id]
                right_vector = right_embeddings[right_row.input_id]
                score = left_vector @ right_vector
                score /= np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
                rows.append(
                    {
                        "query_id": query_id,
                        left_sample_column: int(getattr(left_row, left_sample_column)),
                        right_sample_column: int(getattr(right_row, right_sample_column)),
                        "similarity_score": float(score),
                    }
                )
    return pd.DataFrame(rows)


def create_similarity_matrix(
    scores: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> pd.DataFrame:
    matrix = scores.pivot(
        index=["query_id", row_column],
        columns=column_column,
        values="similarity_score",
    )
    matrix.columns = [f"image_{int(value):03d}" for value in matrix.columns]
    return matrix.reset_index()


def summarize_scores(scores: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    summary = scores.groupby("query_id", as_index=False)["similarity_score"].mean()
    summary = summary.rename(columns={"similarity_score": "mean_pairwise_similarity"})
    return summary.merge(prompts, on="query_id", how="left").sort_values(
        "mean_pairwise_similarity", ascending=False, ignore_index=True
    )


def load_vision_embeddings(
    records: pd.DataFrame,
    input_kind: str,
    value_column: str,
    path: Path,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    if input_kind == "text":
        hashes = {
            row.input_id: compute_stable_hash(getattr(row, value_column))
            for row in records.itertuples(index=False)
        }
    else:
        hashes = {
            row.input_id: compute_file_hash(Path(getattr(row, value_column)))
            for row in records.itertuples(index=False)
        }
    with MultimodalEmbeddingStore(path) as store:
        embeddings = store.get_many(input_kind, get_vision_fingerprint(settings), hashes)
    missing = set(records["input_id"]) - embeddings.keys()
    if missing:
        raise ValueError(f"Embedding database is missing {len(missing)} {input_kind} inputs")
    return embeddings


def write_outputs(
    metric: str,
    scores: pd.DataFrame,
    prompts: pd.DataFrame,
    row_column: str,
    column_column: str,
) -> dict[str, Path]:
    data_dir = DATA_DIR / metric
    tables_dir = RESULTS_ROOT / "tables" / metric
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scores": data_dir / "pairwise_similarity.csv",
        "matrix": tables_dir / "similarity_matrix.csv",
        "query_means": tables_dir / "mean_similarity_by_query.csv",
    }
    scores.to_csv(outputs["scores"], index=False)
    create_similarity_matrix(scores, row_column, column_column).to_csv(
        outputs["matrix"], index=False
    )
    summarize_scores(scores, prompts).to_csv(outputs["query_means"], index=False)
    return outputs


def analyze_clip(
    config: dict[str, Any],
    texts: pd.DataFrame,
    images: pd.DataFrame,
    prompts: pd.DataFrame,
) -> dict[str, Path]:
    settings = config["vision_language_embedding"]
    database = DATA_DIR / "CLIPS" / "embeddings.sqlite"
    text_embeddings = load_vision_embeddings(texts, "text", "response", database, settings)
    image_embeddings = load_vision_embeddings(images, "image", "image_path", database, settings)
    scores = compute_cross_modal_scores(
        texts,
        images,
        text_embeddings,
        image_embeddings,
        "sample_idx",
        "image_sample_idx",
    )
    return write_outputs("CLIPS", scores, prompts, "sample_idx", "image_sample_idx")


def analyze_captions(
    config_path: Path,
    config: dict[str, Any],
    texts: pd.DataFrame,
    prompts: pd.DataFrame,
) -> dict[str, Path]:
    captions = pd.read_csv(DATA_DIR / "image-caption-embed" / "image_captions.csv")
    captions["input_id"] = captions.apply(
        lambda row: f"caption--{row.query_id}--{int(row.image_sample_idx):03d}", axis=1
    )
    inputs = prepare_sentence_inputs(texts, captions)
    settings = load_local_embedding_settings(
        config_path.parent / config["sentence_embedding_config"]
    )
    fingerprint = get_local_embedding_fingerprint(settings)
    with EmbeddingCache(DATA_DIR / "image-caption-embed" / "embeddings.sqlite") as store:
        embeddings = load_response_embeddings(inputs, store, fingerprint, settings.dimensions)
    scores = compute_cross_modal_scores(
        texts,
        captions,
        embeddings,
        embeddings,
        "sample_idx",
        "image_sample_idx",
    )
    return write_outputs(
        "image-caption-embed",
        scores,
        prompts,
        "sample_idx",
        "image_sample_idx",
    )


def analyze_generated_images(
    config: dict[str, Any],
    images: pd.DataFrame,
    prompts: pd.DataFrame,
) -> dict[str, Path]:
    generated = pd.read_csv(DATA_DIR / "text-to-image-embed" / "generated_images.csv")
    generated["input_id"] = generated.apply(
        lambda row: f"generated-image--{row.query_id}--{int(row.response_sample_idx):03d}",
        axis=1,
    )
    settings = config["vision_language_embedding"]
    database = DATA_DIR / "text-to-image-embed" / "embeddings.sqlite"
    original_embeddings = load_vision_embeddings(
        images, "original_image", "image_path", database, settings
    )
    generated_embeddings = load_vision_embeddings(
        generated, "generated_image", "image_path", database, settings
    )
    scores = compute_cross_modal_scores(
        generated,
        images,
        generated_embeddings,
        original_embeddings,
        "response_sample_idx",
        "image_sample_idx",
    )
    return write_outputs(
        "text-to-image-embed",
        scores,
        prompts,
        "response_sample_idx",
        "image_sample_idx",
    )


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    config_path = Path(args.config)
    config = load_analysis_config(config_path)
    generation_config = json.loads((EXPERIMENT_DIR / "generation_config.json").read_text())
    texts, images = prepare_cross_modal_inputs(generation_config)
    images = images.rename(columns={"sample_idx": "image_sample_idx"})
    prompts = pd.DataFrame(generation_config["prompts"]).rename(
        columns={"text": "text_query", "image": "image_query"}
    )
    functions = {
        "CLIPS": lambda: analyze_clip(config, texts, images, prompts),
        "image-caption-embed": lambda: analyze_captions(config_path, config, texts, prompts),
        "text-to-image-embed": lambda: analyze_generated_images(config, images, prompts),
    }
    selected = functions if args.metric == "all" else {args.metric: functions[args.metric]}
    outputs = {}
    for metric, function in selected.items():
        outputs.update({f"{metric}:{name}": path for name, path in function().items()})
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cross-modal embedding similarities")
    parser.add_argument(
        "--metric",
        choices=["CLIPS", "image-caption-embed", "text-to-image-embed", "all"],
        required=True,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


if __name__ == "__main__":
    for name, path in analyze(parse_args()).items():
        print(f"{name}: {path}")
