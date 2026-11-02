import argparse
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from datasets import load_dataset
from ftlangdetect import detect
from tqdm.auto import tqdm

from lib.translate_locally import (
    MARIAN_TRANSLATION_MODEL as TRANSLATION_MODEL,
    MARIAN_TRANSLATION_PROVIDER as TRANSLATION_PROVIDER,
    MARIAN_TRANSLATION_REVISION as TRANSLATION_REVISION,
    MarianSettings,
    get_marian_translator,
    get_text_batches,
    load_marian_settings,
    split_text,
    translate_texts,
)


CONFIG_PATH = Path("experiments/wildchat_multilingual_pilot/translation_config.json")
OUTPUT_DIR = Path("results/wildchat_multilingual_pilot/data")
CHECKPOINT_DIR = Path(".cache/experiments/wildchat_multilingual_pilot")
OUTPUT_COLUMNS = ["turn_idx", "untranslated", "role", "translated"]
LEGACY_CONVERSATION_DTYPE = pl.List(
    pl.Struct(
        {
            "turn_idx": pl.Int64,
            "untranslated": pl.String,
            "role": pl.String,
            "translated": pl.String,
        }
    )
)


@dataclass(frozen=True)
class TranslationSettings:
    source_dataset: str
    source_split: str
    source_row_limit: int
    dataset_batch_size: int
    conversation_turn_limit: int
    target_language: str
    language_ids: dict[str, str]
    marian: MarianSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and translate WildChat conversations with Polars."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--max-dataset-batches",
        type=int,
        help="Stop after this many input batches for a partial test run.",
    )
    return parser.parse_args()


def load_translation_settings(path: Path) -> TranslationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TranslationSettings(
        source_dataset=raw["source_dataset"],
        source_split=raw["source_split"],
        source_row_limit=raw["source_row_limit"],
        dataset_batch_size=raw["dataset_batch_size"],
        conversation_turn_limit=raw["conversation_turn_limit"],
        target_language=raw["target_language"],
        language_ids={row["name"]: row["code"] for row in raw["languages"]},
        marian=load_marian_settings(raw),
    )


def check_run_config(
    args: argparse.Namespace, settings: TranslationSettings
) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.checkpoint_dir / "run_config.json"
    config = {
        "schema_version": 1,
        "source_dataset": settings.source_dataset,
        "source_split": settings.source_split,
        "source_row_limit": settings.source_row_limit,
        "dataset_batch_size": settings.dataset_batch_size,
        "conversation_turn_limit": settings.conversation_turn_limit,
        "target_language": settings.target_language,
        "language_ids": settings.language_ids,
        "translation_provider": TRANSLATION_PROVIDER,
        "translation_model": TRANSLATION_MODEL,
        "translation_revision": TRANSLATION_REVISION,
    }
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        base_keys = set(config) - {
            "translation_provider",
            "translation_model",
            "translation_revision",
        }
        if any(saved.get(key) != config[key] for key in base_keys):
            raise RuntimeError(
                f"Checkpoint settings differ from {config_path}; use another checkpoint directory."
            )
        if saved == config:
            return
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def open_translation_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS local_marian_translations (
            translation_provider TEXT NOT NULL,
            translation_revision TEXT NOT NULL,
            source_language TEXT NOT NULL,
            source_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            PRIMARY KEY (
                translation_provider,
                translation_revision,
                source_language,
                source_text
            )
        )
        """
    )
    return connection


def load_cached_translations(
    connection: sqlite3.Connection, source_language: str, texts: Sequence[str]
) -> dict[str, str]:
    translations: dict[str, str] = {}
    for offset in range(0, len(texts), 500):
        chunk = texts[offset : offset + 500]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"SELECT source_text, translated_text FROM local_marian_translations "
            f"WHERE translation_provider = ? AND translation_revision = ? "
            f"AND source_language = ? AND source_text IN ({placeholders})",
            [TRANSLATION_PROVIDER, TRANSLATION_REVISION, source_language, *chunk],
        )
        translations.update(rows)
    return translations


def save_translations(
    connection: sqlite3.Connection,
    source_language: str,
    translations: dict[str, str],
) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO local_marian_translations VALUES (?, ?, ?, ?, ?)",
        [
            (
                TRANSLATION_PROVIDER,
                TRANSLATION_REVISION,
                source_language,
                source_text,
                translated_text,
            )
            for source_text, translated_text in translations.items()
        ],
    )
    connection.commit()


def matches_language(text: str, language_id: str) -> bool:
    try:
        return detect(text, low_memory=False)["lang"] == language_id
    except Exception:
        return True


def filter_language(
    frame: pl.DataFrame, language: str, language_id: str
) -> pl.DataFrame:
    return (
        frame.filter(pl.col("language") == language)
        .with_columns(
            pl.col("conversation")
            .list.first()
            .struct.field("content")
            .alias("first_content")
        )
        .filter(
            pl.col("first_content").map_elements(
                lambda text: matches_language(text, language_id),
                return_dtype=pl.Boolean,
            )
        )
        .drop("first_content")
    )


def flatten_conversations(frame: pl.DataFrame, turn_limit: int) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col("conversation").list.slice(0, turn_limit))
        .with_columns(
            pl.int_ranges(0, pl.col("conversation").list.len()).alias("turn_idx")
        )
        .explode("conversation", "turn_idx", empty_as_null=True)
        .select(
            "source_row",
            "turn_idx",
            pl.col("conversation").struct.field("content").alias("untranslated"),
            pl.col("conversation").struct.field("role").alias("role"),
        )
    )


def translate_missing_texts(
    texts: Sequence[str],
    source_language: str,
    connection: sqlite3.Connection,
    request_batch: Callable[[Sequence[str]], list[str]],
    settings: TranslationSettings,
    progress_description: str = "Translation",
) -> dict[str, str]:
    translations = load_cached_translations(connection, source_language, texts)
    missing = list(dict.fromkeys(text for text in texts if text not in translations))
    split_texts = {
        text: split_text(text, settings.marian.max_chunk_characters)
        for text in missing
    }
    segments = list(
        dict.fromkeys(segment for parts in split_texts.values() for segment in parts)
    )
    segment_translations = load_cached_translations(
        connection, source_language, segments
    )
    missing_segments = [
        segment for segment in segments if segment not in segment_translations
    ]
    batches = list(
        get_text_batches(
            ((text, text) for text in missing_segments),
            settings.marian.max_batch_characters,
            settings.marian.max_batch_items,
        )
    )
    with tqdm(
        total=len(segments),
        initial=len(segments) - len(missing_segments),
        desc=progress_description,
        unit="segment",
    ) as progress:
        for batch in batches:
            batch_texts = [text for _, text in batch]
            translated = request_batch(batch_texts)
            additions = dict(zip(batch_texts, translated, strict=True))
            save_translations(connection, source_language, additions)
            segment_translations.update(additions)
            progress.update(len(additions))
    assembled = {
        text: "\n".join(segment_translations[segment] for segment in parts)
        for text, parts in split_texts.items()
    }
    save_translations(connection, source_language, assembled)
    translations.update(assembled)
    return translations


def build_translation_rows(
    turns: pl.DataFrame, translations: dict[str, str]
) -> pl.DataFrame:
    mapping = pl.DataFrame(
        {
            "untranslated": list(translations),
            "translated": list(translations.values()),
        }
    )
    return turns.join(
        mapping, on="untranslated", how="left", validate="m:1"
    ).select(
        "source_row", "turn_idx", "untranslated", "role", "translated"
    )


def get_shard_path(checkpoint_dir: Path, language: str, batch_index: int) -> Path:
    return (
        checkpoint_dir
        / "checkpoints"
        / TRANSLATION_PROVIDER
        / f"{language}-{batch_index:05d}.parquet"
    )


def read_translation_shard(path: Path) -> pl.LazyFrame:
    shard = pl.scan_parquet(path)
    if "conversation" not in shard.collect_schema():
        return shard
    return (
        shard.with_columns(
            pl.col("conversation").str.json_decode(LEGACY_CONVERSATION_DTYPE)
        )
        .explode("conversation", empty_as_null=True)
        .unnest("conversation")
        .select("source_row", *OUTPUT_COLUMNS)
    )


def process_language(
    frame: pl.DataFrame,
    language: str,
    language_id: str,
    shard_path: Path,
    connection: sqlite3.Connection,
    settings: TranslationSettings,
    load_translator: Callable[[], Any],
) -> None:
    filtered = filter_language(frame, language, language_id)
    if filtered.is_empty():
        rows = pl.DataFrame(
            schema={
                "source_row": pl.UInt32,
                "turn_idx": pl.Int64,
                "untranslated": pl.String,
                "role": pl.String,
                "translated": pl.String,
            }
        )
    else:
        turns = flatten_conversations(filtered, settings.conversation_turn_limit)
        texts = turns.get_column("untranslated").unique(maintain_order=True).to_list()

        def request_batch(batch: Sequence[str]) -> list[str]:
            return translate_texts(
                load_translator(),
                list(batch),
                settings.target_language,
                language_id,
            )

        translations = translate_missing_texts(
            texts,
            language_id,
            connection,
            request_batch,
            settings,
            f"{language} batch {shard_path.stem.rsplit('-', 1)[-1]}",
        )
        rows = build_translation_rows(turns, translations)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(shard_path, compression="zstd")


def finalize_outputs(
    checkpoint_dir: Path, output_dir: Path, languages: Sequence[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for language in languages:
        shards = sorted(
            (checkpoint_dir / "checkpoints" / TRANSLATION_PROVIDER).glob(
                f"{language}-*.parquet"
            )
        )
        if not shards:
            continue
        output_path = output_dir / f"{language}.csv"
        temporary_path = output_path.with_suffix(".csv.tmp")
        (
            pl.concat([read_translation_shard(path) for path in shards])
            .sort("source_row")
            .select(OUTPUT_COLUMNS)
            .sink_csv(temporary_path)
        )
        temporary_path.replace(output_path)
        print(f"Wrote {output_path}")


def process_dataset(
    args: argparse.Namespace,
    translator: Any | None = None,
    stream: Any | None = None,
) -> None:
    settings = load_translation_settings(args.config)
    check_run_config(args, settings)
    if stream is None:
        stream = load_dataset(
            settings.source_dataset, split=settings.source_split, streaming=True
        )
    connection = open_translation_store(args.checkpoint_dir / "translation_cache.sqlite")
    marian_translator = translator

    def load_translator() -> Any:
        nonlocal marian_translator
        marian_translator = get_marian_translator(
            settings.marian, marian_translator
        )
        return marian_translator

    try:
        for batch_index, batch in enumerate(
            stream.iter(batch_size=settings.dataset_batch_size)
        ):
            source_row = batch_index * settings.dataset_batch_size
            if source_row >= settings.source_row_limit:
                break
            if (
                args.max_dataset_batches is not None
                and batch_index >= args.max_dataset_batches
            ):
                break
            pending = {
                language: language_id
                for language, language_id in settings.language_ids.items()
                if not get_shard_path(
                    args.checkpoint_dir, language, batch_index
                ).exists()
            }
            if not pending:
                print(f"Input batch {batch_index:,}: already complete")
                continue
            frame = pl.DataFrame(
                {
                    "language": batch["language"],
                    "conversation": batch["conversation"],
                },
                strict=False,
            ).head(settings.source_row_limit - source_row).with_row_index(
                "source_row", offset=source_row
            )
            for language, language_id in pending.items():
                print(f"Input batch {batch_index:,}: processing {language}")
                process_language(
                    frame,
                    language,
                    language_id,
                    get_shard_path(args.checkpoint_dir, language, batch_index),
                    connection,
                    settings,
                    load_translator,
                )
    finally:
        connection.close()
    finalize_outputs(args.checkpoint_dir, args.output_dir, settings.language_ids)


def main() -> None:
    process_dataset(parse_args())


if __name__ == "__main__":
    main()
