from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from lib.translate_locally import (
    LocalTranslationSettings,
    TRANSLATION_PROVIDER,
    get_local_translator,
    get_translation_batches,
    load_local_translation_settings,
    translate_texts,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    GENERATION_COLUMNS,
    GENERATIONS_PATH,
    LANGUAGES,
    load_generations,
    write_csv,
)
from lib.compute_identifiers import compute_stable_hash


CHECKPOINT_COLUMNS = [
    "source_language_code",
    "source_language",
    "source_response_sha256",
    "source_response",
    "response_en",
    "translation_provider",
    "translation_model",
    "translation_revision",
]
TRANSLATION_CONFIG = Path(__file__).with_name("translation_config.json")


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    checkpoint = pd.read_csv(path)
    if "translation_model" not in checkpoint:
        checkpoint["translation_model"] = None
    if "translation_revision" not in checkpoint:
        checkpoint["translation_revision"] = None
    checkpoint = checkpoint[CHECKPOINT_COLUMNS].fillna("")
    return checkpoint.loc[
        checkpoint["translation_provider"] == TRANSLATION_PROVIDER
    ].copy()


def load_translation_settings(path: Path) -> LocalTranslationSettings:
    return load_local_translation_settings(json.loads(path.read_text()))


def get_translation_source(generation_path: Path) -> pd.DataFrame:
    generations = load_generations(generation_path)
    source = generations[generations["translated_language_code"].notna()].copy()
    source["source_response_sha256"] = source["model_response"].map(compute_stable_hash)
    source = source[
        [
            "translated_language_code",
            "translated_language",
            "source_response_sha256",
            "model_response",
        ]
    ].drop_duplicates(["translated_language_code", "source_response_sha256"])
    return source.rename(
        columns={
            "translated_language_code": "source_language_code",
            "translated_language": "source_language",
            "model_response": "source_response",
        }
    ).reset_index(drop=True)


def get_missing_translations(
    source: pd.DataFrame,
    checkpoint: pd.DataFrame,
) -> pd.DataFrame:
    checkpointed_keys = set(
        checkpoint[["source_language_code", "source_response_sha256"]].itertuples(
            index=False, name=None
        )
    )
    keys = source[["source_language_code", "source_response_sha256"]].apply(tuple, axis=1)
    return source.loc[~keys.isin(checkpointed_keys)].reset_index(drop=True)


def materialize_english_responses(
    generation_path: Path,
    checkpoint: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    generations = load_generations(generation_path)
    translation_columns = [
        "response_translation",
        "response_translation_language_code",
        "response_translation_provider",
    ]
    generations[translation_columns] = generations[translation_columns].astype(object)
    lookup = {
        (row.source_language_code, row.source_response_sha256): (
            row.response_en,
            row.translation_provider,
        )
        for row in checkpoint.itertuples(index=False)
    }
    for index, row in generations.iterrows():
        if pd.isna(row["translated_language_code"]):
            continue
        text_hash = compute_stable_hash(row["model_response"])
        key = (row["translated_language_code"], text_hash)
        response_en, provider = lookup[key]
        generations.at[index, "response_translation"] = response_en
        generations.at[index, "response_translation_language_code"] = "en"
        generations.at[index, "response_translation_provider"] = provider
    write_csv(generations, output_path, GENERATION_COLUMNS)
    return generations


def translate_responses(args: argparse.Namespace, translator: Any | None = None) -> None:
    settings = load_translation_settings(Path(args.config))
    generation_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    source = get_translation_source(generation_path)
    checkpoint = load_checkpoint(checkpoint_path)
    missing = get_missing_translations(source, checkpoint)
    print(
        f"Response translations: {len(source) - len(missing)} checkpointed, "
        f"{len(missing)} missing "
        f"({missing['source_response'].str.len().sum():,} source characters)"
    )
    if args.dry_run:
        return
    local_translator = translator
    for language_code, language_source in missing.groupby("source_language_code", sort=True):
        if language_code not in LANGUAGES or language_code == "en":
            raise ValueError(f"Unsupported response language: {language_code}")
        total = len(source.loc[source["source_language_code"] == language_code])
        with tqdm(
            total=total,
            initial=total - len(language_source),
            desc=f"Translate {language_code} to en",
            unit="response",
        ) as progress:
            for batch in get_translation_batches(
                language_source,
                "source_response",
                settings.max_batch_characters,
                settings.max_batch_items,
            ):
                local_translator = get_local_translator(settings, local_translator)
                response_en = translate_texts(
                    local_translator,
                    batch["source_response"].tolist(),
                    "en",
                    language_code,
                )
                rows = batch.copy()
                rows["response_en"] = response_en
                rows["translation_provider"] = TRANSLATION_PROVIDER
                rows["translation_model"] = settings.model_id
                rows["translation_revision"] = settings.revision
                checkpoint = pd.concat(
                    [checkpoint, rows[CHECKPOINT_COLUMNS]],
                    ignore_index=True,
                )
                write_csv(checkpoint, checkpoint_path, CHECKPOINT_COLUMNS)
                progress.update(len(rows))
    materialized = materialize_english_responses(
        generation_path, checkpoint, Path(args.output)
    )
    print(f"Wrote {len(materialized)} English-normalized responses to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate non-English Gemini responses to English locally"
    )
    parser.add_argument(
        "--input",
        default=str(GENERATIONS_PATH),
    )
    parser.add_argument(
        "--checkpoint",
        default=".cache/experiments/en_te_yo_30_query/response_translation_checkpoint.csv",
    )
    parser.add_argument(
        "--output", default=str(GENERATIONS_PATH)
    )
    parser.add_argument("--config", default=str(TRANSLATION_CONFIG))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    translate_responses(parse_args())
