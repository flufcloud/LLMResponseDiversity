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


EXPERIMENT_ID = "intralingual_multilingual_embedding_validation"
EXPERIMENT_DIR = Path("experiments/embedding_validation/intralingual")
RESULTS_DIR = Path("results/embedding_validation/intralingual")
DATA_DIR = RESULTS_DIR / "data"
CHECKPOINT_COLUMNS = [
    "mrpc_sentence_id",
    "sentence_kind",
    "target_lang_identifier",
    "translation",
    "translation_provider",
    "translation_model",
    "translation_revision",
]
OUTPUT_COLUMNS = [
    "index",
    "mrpc_sentence_id",
    "target_lang_identifier",
    "translated_base",
    "translated_paraphrase",
]


@dataclass(frozen=True)
class TranslationSettings:
    source_dataset: str
    source_subset: str
    source_split: str
    required_label: int
    sample_count: int
    random_seed: int
    source_language: str
    languages: tuple[str, ...]
    local: LocalTranslationSettings


def load_translation_settings(path: Path) -> TranslationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TranslationSettings(
        raw["source_dataset"],
        raw["source_subset"],
        raw["source_split"],
        raw["required_label"],
        raw["sample_count"],
        raw["random_seed"],
        raw["source_language"],
        tuple(language["code"] for language in raw["languages"]),
        load_local_translation_settings(raw),
    )


def sample_mrpc_pairs(
    rows: Iterable[dict[str, Any]],
    settings: TranslationSettings,
) -> pd.DataFrame:
    mrpc = pd.DataFrame(rows)[["idx", "sentence1", "sentence2", "label"]]
    pairs = mrpc[mrpc["label"] == settings.required_label].sample(
        n=settings.sample_count,
        random_state=settings.random_seed,
    )
    return pairs.rename(
        columns={
            "idx": "mrpc_sentence_id",
            "sentence1": "base",
            "sentence2": "paraphrase",
        }
    )[["mrpc_sentence_id", "base", "paraphrase"]].reset_index(drop=True)


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


def get_missing_translation_inputs(
    pairs: pd.DataFrame,
    language: str,
    checkpoint: pd.DataFrame,
) -> pd.DataFrame:
    completed = {
        (row.mrpc_sentence_id, row.sentence_kind)
        for row in checkpoint.loc[
            checkpoint["target_lang_identifier"] == language
        ].itertuples(index=False)
    }
    records = []
    for row in pairs.itertuples(index=False):
        for sentence_kind in ("base", "paraphrase"):
            if (row.mrpc_sentence_id, sentence_kind) not in completed:
                records.append(
                    {
                        "mrpc_sentence_id": row.mrpc_sentence_id,
                        "sentence_kind": sentence_kind,
                        "text": getattr(row, sentence_kind),
                    }
                )
    return pd.DataFrame(records, columns=["mrpc_sentence_id", "sentence_kind", "text"])


def write_translated_mrpc(
    pairs: pd.DataFrame,
    checkpoint: pd.DataFrame,
    settings: TranslationSettings,
    path: Path,
) -> None:
    translations = {
        (row.mrpc_sentence_id, row.sentence_kind, row.target_lang_identifier): row.translation
        for row in checkpoint.itertuples(index=False)
    }
    records = []
    for pair in pairs.itertuples(index=False):
        for language in settings.languages:
            if language == settings.source_language:
                base = pair.base
                paraphrase = pair.paraphrase
            else:
                base = translations[(pair.mrpc_sentence_id, "base", language)]
                paraphrase = translations[(pair.mrpc_sentence_id, "paraphrase", language)]
            records.append(
                {
                    "index": len(records),
                    "mrpc_sentence_id": pair.mrpc_sentence_id,
                    "target_lang_identifier": language,
                    "translated_base": base,
                    "translated_paraphrase": paraphrase,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def translate_mrpc(
    args: argparse.Namespace,
    translator: Any | None = None,
    source_rows: Iterable[dict[str, Any]] | None = None,
) -> None:
    settings = load_translation_settings(Path(args.config))
    if source_rows is None:
        source_rows = load_dataset(
            settings.source_dataset,
            settings.source_subset,
            split=settings.source_split,
            cache_dir=args.cache_dir,
        )
    pairs = sample_mrpc_pairs(source_rows, settings)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_translation_checkpoint(checkpoint_path)
    local_translator = translator
    for language in settings.languages:
        if language == settings.source_language:
            continue
        missing = get_missing_translation_inputs(pairs, language, checkpoint)
        completed = len(pairs) * 2 - len(missing)
        with tqdm(
            total=len(pairs) * 2,
            initial=completed,
            desc=f"Translate en to {language}",
            unit="text",
        ) as progress:
            for batch in get_translation_batches(
                missing,
                "text",
                settings.local.max_batch_characters,
                settings.local.max_batch_items,
            ):
                local_translator = get_local_translator(
                    settings.local, local_translator
                )
                translations = translate_texts(
                    local_translator,
                    batch["text"].tolist(),
                    language,
                    settings.source_language,
                )
                additions = batch.drop(columns="text").copy()
                additions["target_lang_identifier"] = language
                additions["translation"] = translations
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
    write_translated_mrpc(pairs, checkpoint, settings, Path(args.output))
    print(f"Wrote {len(pairs) * len(settings.languages)} rows to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate positive MRPC sentence pairs")
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "translation_config.json"))
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(".cache/experiments") / EXPERIMENT_ID / "translation_checkpoint.csv"),
    )
    parser.add_argument("--output", default=str(DATA_DIR / "mrpc_translated.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    translate_mrpc(parse_args())
