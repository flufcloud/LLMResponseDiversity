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
from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import LANGUAGES
from experiments.hivemind_reproduction.en_te_yo_30_query.resolve_paths import ExperimentPaths


TARGET_LANGUAGES = {code: name for code, name in LANGUAGES.items() if code != "en"}
TRANSLATION_CONFIG = Path(__file__).with_name("translation_config.json")
CHECKPOINT_COLUMNS = [
    "prompt_id",
    "source_row_index",
    "query_en",
    "language_code",
    "language",
    "query",
    "translation_provider",
    "translation_model",
    "translation_revision",
]


def load_translation_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    checkpoint = pd.read_csv(path)
    if "translation_model" not in checkpoint:
        checkpoint["translation_model"] = None
    if "translation_revision" not in checkpoint:
        checkpoint["translation_revision"] = None
    checkpoint = checkpoint[CHECKPOINT_COLUMNS]
    return checkpoint.loc[
        checkpoint["translation_provider"] == TRANSLATION_PROVIDER
    ].copy()


def load_translation_settings(path: Path) -> LocalTranslationSettings:
    return load_local_translation_settings(json.loads(path.read_text()))


def build_translation_rows(
    source: pd.DataFrame,
    translations: list[str],
    settings: LocalTranslationSettings,
    language_code: str,
) -> pd.DataFrame:
    rows = source[["prompt_id", "source_row_index", "query_en"]].copy()
    rows["language_code"] = language_code
    rows["language"] = TARGET_LANGUAGES[language_code]
    rows["query"] = translations
    rows["translation_provider"] = TRANSLATION_PROVIDER
    rows["translation_model"] = settings.model_id
    rows["translation_revision"] = settings.revision
    return rows[CHECKPOINT_COLUMNS]


def write_multilingual_prompts(
    source: pd.DataFrame,
    translations: pd.DataFrame,
    paths: ExperimentPaths,
) -> None:
    base_columns = ["prompt_id", "source_row_index", "query_en"]
    english = source[base_columns].copy()
    english["language_code"] = "en"
    english["language"] = LANGUAGES["en"]
    english["query"] = english["query_en"]
    translated = source[base_columns].merge(
        translations[["prompt_id", "query_en", "language_code", "language", "query"]],
        on=["prompt_id", "query_en"],
        how="inner",
    )
    multilingual = pd.concat(
        [english, translated[base_columns + ["language_code", "language", "query"]]],
        ignore_index=True,
    ).sort_values(["prompt_id", "language_code"])
    multilingual.to_csv(paths.multilingual_prompts, index=False)


def translate_prompts(args: argparse.Namespace, translator: Any | None = None) -> None:
    settings = load_translation_settings(Path(args.translation_config))
    paths = ExperimentPaths(Path(args.output_dir))
    source = pd.read_csv(paths.source_prompts)
    checkpoint = load_translation_checkpoint(paths.translation_checkpoint)
    local_translator = translator
    for language_code in TARGET_LANGUAGES:
        checkpointed_keys = set(
            checkpoint.loc[
                checkpoint["language_code"] == language_code,
                ["prompt_id", "query_en"],
            ]
            .itertuples(index=False, name=None)
        )
        missing = source[
            ~source[["prompt_id", "query_en"]]
            .apply(tuple, axis=1)
            .isin(checkpointed_keys)
        ]
        with tqdm(
            total=len(source),
            initial=len(source) - len(missing),
            desc=f"Translate en to {language_code}",
            unit="query",
        ) as progress:
            for batch in get_translation_batches(
                missing,
                "query_en",
                settings.max_batch_characters,
                settings.max_batch_items,
            ):
                local_translator = get_local_translator(settings, local_translator)
                translated = translate_texts(
                    local_translator,
                    batch["query_en"].tolist(),
                    language_code,
                )
                additions = build_translation_rows(
                    batch,
                    translated,
                    settings,
                    language_code,
                )
                checkpoint = pd.concat(
                    [checkpoint, additions],
                    ignore_index=True,
                )
                checkpoint.to_csv(paths.translation_checkpoint, index=False)
                progress.update(len(additions))
    write_multilingual_prompts(source, checkpoint, paths)
    print(f"Wrote {len(source) * len(LANGUAGES)} prompts to {paths.multilingual_prompts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate the selected prompts")
    parser.add_argument("--output-dir", default=".cache/experiments/en_te_yo_30_query")
    parser.add_argument("--translation-config", default=str(TRANSLATION_CONFIG))
    return parser.parse_args()


if __name__ == "__main__":
    translate_prompts(parse_args())
