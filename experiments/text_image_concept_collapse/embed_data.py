from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.compute_identifiers import compute_stable_hash
from lib.embed_images_locally import LocalVisionLanguageEncoder
from lib.embed_locally import (
    LocalSentenceEncoder,
    fetch_local_embeddings,
    load_local_embedding_settings,
)
from lib.store_embeddings import EmbeddingCache
from lib.store_multimodal_embeddings import MultimodalEmbeddingStore


EXPERIMENT_ID = "text_image_concept_collapse"
EXPERIMENT_DIR = Path(__file__).parent
RESULTS_ROOT = Path("results") / EXPERIMENT_ID
DATA_DIR = RESULTS_ROOT / "data"
DEFAULT_CONFIG = EXPERIMENT_DIR / "analysis_config.json"


def load_analysis_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_cross_modal_inputs(
    config: dict[str, Any],
    text_path: Path = DATA_DIR / "text_experiment.csv",
    image_path: Path = DATA_DIR / "image_experiment.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_ids = {prompt["text"].rstrip("."): prompt["query_id"] for prompt in config["prompts"]}
    texts = pd.read_csv(text_path).drop(columns=["Unnamed: 0"], errors="ignore")
    texts["query_id"] = texts["query"].str.rstrip(".").map(query_ids)
    texts["input_id"] = texts.apply(
        lambda row: f"text--{row.query_id}--{int(row.sample_idx):03d}", axis=1
    )
    images = pd.read_csv(image_path)
    images["input_id"] = images.apply(
        lambda row: f"image--{row.query_id}--{int(row.sample_idx):03d}", axis=1
    )
    return texts, images


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_vision_fingerprint(settings: dict[str, Any]) -> str:
    return (
        f"hf:{settings['model']}@{settings['revision']}:"
        f"{settings['dtype']}:normalized"
    )


def embed_vision_inputs(
    records: pd.DataFrame,
    input_kind: str,
    value_column: str,
    store: MultimodalEmbeddingStore,
    encoder: Any,
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
    fingerprint = get_vision_fingerprint(settings)
    embeddings = store.get_many(input_kind, fingerprint, hashes)
    missing = records.loc[~records["input_id"].isin(embeddings)]
    print(f"{input_kind} embeddings: {len(embeddings)} hits, {len(missing)} misses")
    if not missing.empty:
        values = missing[value_column].tolist()
        vectors = (
            encoder.encode_texts(values)
            if input_kind == "text"
            else encoder.encode_images([Path(value) for value in values])
        )
        expected_shape = (len(missing), settings["dimensions"])
        if vectors.shape != expected_shape or not np.isfinite(vectors).all():
            raise ValueError(f"Embedding model returned unexpected output: {vectors.shape}")
        rows = [
            (input_id, hashes[input_id], vector)
            for input_id, vector in zip(missing["input_id"], vectors, strict=True)
        ]
        store.put_many(input_kind, fingerprint, rows)
        embeddings.update({input_id: vector for input_id, _, vector in rows})
    return embeddings


def prepare_sentence_inputs(
    texts: pd.DataFrame,
    captions: pd.DataFrame,
) -> pd.DataFrame:
    response_rows = texts[["input_id", "response"]].copy()
    caption_rows = captions.copy()
    caption_rows["input_id"] = caption_rows.apply(
        lambda row: f"caption--{row.query_id}--{int(row.image_sample_idx):03d}", axis=1
    )
    caption_rows = caption_rows[["input_id", "caption"]].rename(columns={"caption": "response"})
    inputs = pd.concat([response_rows, caption_rows], ignore_index=True)
    inputs["response_id"] = inputs["input_id"]
    inputs["response_sha256"] = inputs["response"].map(compute_stable_hash)
    return inputs


def embed_clip(
    config: dict[str, Any],
    texts: pd.DataFrame,
    images: pd.DataFrame,
    allow_model_download: bool,
) -> None:
    settings = config["vision_language_embedding"]
    encoder = LocalVisionLanguageEncoder(settings, allow_model_download)
    path = DATA_DIR / "CLIPS" / "embeddings.sqlite"
    with MultimodalEmbeddingStore(path) as store:
        embed_vision_inputs(texts, "text", "response", store, encoder, settings)
        embed_vision_inputs(images, "image", "image_path", store, encoder, settings)


def embed_captions(
    config_path: Path,
    config: dict[str, Any],
    texts: pd.DataFrame,
    allow_model_download: bool,
) -> None:
    caption_path = DATA_DIR / "image-caption-embed" / "image_captions.csv"
    captions = pd.read_csv(caption_path)
    inputs = prepare_sentence_inputs(texts, captions)
    settings_path = config_path.parent / config["sentence_embedding_config"]
    settings = load_local_embedding_settings(settings_path)
    encoder = LocalSentenceEncoder(settings, allow_model_download)
    with EmbeddingCache(DATA_DIR / "image-caption-embed" / "embeddings.sqlite") as store:
        fetch_local_embeddings(inputs, store, encoder, settings)


def embed_generated_images(
    config: dict[str, Any],
    images: pd.DataFrame,
    allow_model_download: bool,
) -> None:
    generated_path = DATA_DIR / "text-to-image-embed" / "generated_images.csv"
    generated = pd.read_csv(generated_path)
    generated["input_id"] = generated.apply(
        lambda row: f"generated-image--{row.query_id}--{int(row.response_sample_idx):03d}",
        axis=1,
    )
    settings = config["vision_language_embedding"]
    encoder = LocalVisionLanguageEncoder(settings, allow_model_download)
    path = DATA_DIR / "text-to-image-embed" / "embeddings.sqlite"
    with MultimodalEmbeddingStore(path) as store:
        embed_vision_inputs(images, "original_image", "image_path", store, encoder, settings)
        embed_vision_inputs(generated, "generated_image", "image_path", store, encoder, settings)


def embed_data(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    analysis_config = load_analysis_config(config_path)
    generation_config = json.loads((EXPERIMENT_DIR / "generation_config.json").read_text())
    texts, images = prepare_cross_modal_inputs(generation_config)
    if args.metric in {"CLIPS", "all"}:
        embed_clip(analysis_config, texts, images, args.allow_model_download)
    if args.metric in {"image-caption-embed", "all"}:
        embed_captions(config_path, analysis_config, texts, args.allow_model_download)
    if args.metric in {"text-to-image-embed", "all"}:
        embed_generated_images(analysis_config, images, args.allow_model_download)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed cross-modal experiment data locally")
    parser.add_argument(
        "--metric",
        choices=["CLIPS", "image-caption-embed", "text-to-image-embed", "all"],
        required=True,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow Hugging Face model downloads when files are not already local",
    )
    return parser.parse_args()


if __name__ == "__main__":
    embed_data(parse_args())
