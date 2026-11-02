from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from lib.translate_locally import (
    LocalTranslationSettings,
    TRANSLATION_PROVIDER,
    get_local_translator,
    get_translation_batches,
    load_local_translation_settings,
    translate_texts,
)


OUTPUT_COLUMNS = ["query_id", "lang", "query"]
CHECKPOINT_COLUMNS = [
    "query_id",
    "base_query",
    "lang",
    "query",
    "translation_provider",
    "translation_model",
    "translation_revision",
]


@dataclass(frozen=True)
class TranslationSettings:
    source_dataset: str
    source_split: str
    query_count: int
    source_language: str
    languages: tuple[str, ...]
    local: LocalTranslationSettings


def load_translation_settings(path: Path) -> TranslationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    language_rows = raw["languages"]
    languages = tuple(row["code"] for row in language_rows)
    source_language = raw["source_language"]
    return TranslationSettings(
        raw["source_dataset"],
        raw["source_split"],
        raw["query_count"],
        source_language,
        languages,
        load_local_translation_settings(raw),
    )


def prepare_query_rows(rows: Iterable[dict[str, Any]], expected_count: int) -> pd.DataFrame:
    queries = pd.DataFrame(rows)[["query"]].copy()
    queries["query"] = queries["query"].fillna("").astype(str).str.strip()
    queries.insert(
        0,
        "query_id",
        [f"q{index:03d}" for index in range(1, expected_count + 1)],
    )
    return queries


def load_translation_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    checkpoint = pd.read_csv(path)
    if "translation_revision" not in checkpoint:
        checkpoint["translation_revision"] = None
    checkpoint = checkpoint[CHECKPOINT_COLUMNS]
    return checkpoint.loc[
        checkpoint["translation_provider"] == TRANSLATION_PROVIDER
    ].copy()


def write_translated_queries(
    queries: pd.DataFrame,
    checkpoint: pd.DataFrame,
    settings: TranslationSettings,
    path: Path,
) -> None:
    english = queries.assign(lang=settings.source_language)
    translated = checkpoint[["query_id", "lang", "query"]]
    output = pd.concat([english, translated], ignore_index=True)[OUTPUT_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.sort_values(["query_id", "lang"]).to_csv(path, index=False)


def translate_queries(
    args: argparse.Namespace,
    translator: Any | None = None,
    source_rows: Iterable[dict[str, Any]] | None = None,
) -> None:
    settings = load_translation_settings(Path(args.config))
    if source_rows is None:
        source_rows = load_dataset(
            settings.source_dataset,
            split=settings.source_split,
            cache_dir=args.cache_dir,
        )
    queries = prepare_query_rows(source_rows, settings.query_count)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_translation_checkpoint(checkpoint_path)
    local_translator = translator
    for language in settings.languages:
        if language == settings.source_language:
            continue
        checkpointed_ids = set(
            checkpoint.loc[checkpoint["lang"] == language, "query_id"]
        )
        missing = queries[~queries["query_id"].isin(checkpointed_ids)]
        with tqdm(
            total=len(queries),
            initial=len(checkpointed_ids),
            desc=f"Translate en to {language}",
            unit="query",
        ) as progress:
            for batch in get_translation_batches(
                missing,
                "query",
                settings.local.max_batch_characters,
                settings.local.max_batch_items,
            ):
                local_translator = get_local_translator(
                    settings.local, local_translator
                )
                translated = translate_texts(
                    local_translator,
                    batch["query"].tolist(),
                    language,
                    settings.source_language,
                )
                additions = batch.rename(columns={"query": "base_query"}).copy()
                additions["lang"] = language
                additions["query"] = translated
                additions["translation_provider"] = TRANSLATION_PROVIDER
                additions["translation_model"] = settings.local.model_id
                additions["translation_revision"] = settings.local.revision
                checkpoint = pd.concat(
                    [checkpoint, additions[CHECKPOINT_COLUMNS]],
                    ignore_index=True,
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.to_csv(checkpoint_path, index=False)
                progress.update(len(additions))
    output_path = Path(args.output)
    write_translated_queries(queries, checkpoint, settings, output_path)
    print(f"Wrote {len(queries) * len(settings.languages)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate Infinite Chats queries for multilingual embedding validation"
    )
    parser.add_argument(
        "--config",
        default="experiments/embedding_validation/cross_lingual/translation_config.json",
    )
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument(
        "--checkpoint",
        default=(
            ".cache/experiments/cross_lingual_embedding_validation/"
            "translation_checkpoint.csv"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/embedding_validation/cross_lingual/data/"
            "translated_queries.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    translate_queries(parse_args())
