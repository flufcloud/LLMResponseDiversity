from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    EMBEDDINGS_PATH,
    GENERATIONS_PATH,
    LANGUAGES,
    get_analysis_responses,
    load_generations,
    select_multilingual_queries,
)
from lib.prepare_responses import normalize_responses
from lib.embed_with_bge import (
    BGE_MAX_LENGTH,
    BGEM3Encoder,
    fetch_bge_embeddings,
)
from lib.store_embeddings import EmbeddingCache


def load_multilingual_responses(path: Path) -> pd.DataFrame:
    generations = load_generations(path)
    responses = get_analysis_responses(generations)
    return select_multilingual_queries(responses, list(LANGUAGES))


def load_translated_responses(path: Path) -> pd.DataFrame:
    generations = load_generations(path)
    multilingual = select_multilingual_queries(
        get_analysis_responses(generations), list(LANGUAGES)
    )
    indexed = generations.set_index("generation_id")
    translated = multilingual["generation_id"].map(indexed["response_translation"])
    non_english = multilingual["language_code"] != "en"
    multilingual.loc[non_english, "response"] = translated[non_english]
    multilingual.loc[non_english, "response_input_kind"] = "response_translation"
    return multilingual


def prepare_responses(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    normalized = normalize_responses(raw)
    if args.config_id:
        normalized = normalized[normalized["config_id"] == args.config_id]
    return normalized[normalized["language_code"].isin(args.languages)].copy()


def embed_data(args: argparse.Namespace, encoder: Any | None = None) -> dict[str, int]:
    encoder = encoder or BGEM3Encoder(
        device=args.device,
        max_length=args.max_length,
        allow_model_download=args.allow_model_download,
    )
    inputs = {"model_response": load_multilingual_responses(Path(args.input))}
    if args.include_translated:
        inputs["response_translation"] = load_translated_responses(Path(args.input))

    counts = {}
    with EmbeddingCache(Path(args.cache)) as cache:
        for input_kind, raw in inputs.items():
            selected = prepare_responses(raw, args)
            fetch_bge_embeddings(
                selected,
                cache,
                encoder,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            counts[input_kind] = len(selected)
    return counts


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=str(GENERATIONS_PATH))
    parser.add_argument("--cache", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=BGE_MAX_LENGTH)
    parser.add_argument("--config-id")
    parser.add_argument("--include-translated", action="store_true")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow downloading the pinned BGE-M3 model when it is not cached locally",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed the multilingual responses")
    add_embedding_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    embedded_counts = embed_data(parse_args())
    for kind, count in embedded_counts.items():
        print(f"{kind}: {count} responses")
