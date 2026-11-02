from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.compute_identifiers import compute_stable_hash, create_stable_id


@dataclass(frozen=True)
class RegressionSettings:
    experiment_id: str
    source_csv: Path
    word_column: str
    target_columns: tuple[str, ...]
    exclude_parenthetical_senses: bool
    train_fraction: float
    dev_fraction: float
    test_fraction: float
    split_seed: int
    stratify_bins: int
    ridge_alphas: tuple[float, ...]


def load_regression_settings(path: Path) -> RegressionSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RegressionSettings(
        experiment_id=raw["experiment_id"],
        source_csv=Path(raw["source_csv"]),
        word_column=raw["word_column"],
        target_columns=tuple(raw["target_columns"]),
        exclude_parenthetical_senses=raw["exclude_parenthetical_senses"],
        train_fraction=raw["train_fraction"],
        dev_fraction=raw["dev_fraction"],
        test_fraction=raw["test_fraction"],
        split_seed=raw["split_seed"],
        stratify_bins=raw["stratify_bins"],
        ridge_alphas=tuple(raw["ridge_alphas"]),
    )


def normalize_word(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    return normalized.replace("’", "'")


def assign_splits(rows: pd.DataFrame, settings: RegressionSettings) -> pd.Series:
    composite_score = rows[list(settings.target_columns)].rank(pct=True).mean(axis=1)
    score_bins = pd.qcut(
        composite_score,
        q=settings.stratify_bins,
        labels=False,
        duplicates="drop",
    )
    train_indices, held_out_indices = train_test_split(
        rows.index,
        test_size=1 - settings.train_fraction,
        random_state=settings.split_seed,
        stratify=score_bins,
    )
    test_share = settings.test_fraction / (
        settings.dev_fraction + settings.test_fraction
    )
    dev_indices, test_indices = train_test_split(
        held_out_indices,
        test_size=test_share,
        random_state=settings.split_seed,
        stratify=score_bins.loc[held_out_indices],
    )
    splits = pd.Series("train", index=rows.index, dtype="string")
    splits.loc[dev_indices] = "dev"
    splits.loc[test_indices] = "test"
    return splits


def prepare_training_rows(settings: RegressionSettings) -> pd.DataFrame:
    source = pd.read_csv(settings.source_csv)
    if settings.exclude_parenthetical_senses:
        source = source[
            ~source[settings.word_column].str.contains(r"\([^)]*\)", regex=True)
        ].copy()
    source[settings.word_column] = source[settings.word_column].map(normalize_word)
    source["word_id"] = source[settings.word_column].map(
        lambda word: create_stable_id("word", word)
    )
    source["split"] = assign_splits(source, settings)
    columns = ["word_id", settings.word_column, *settings.target_columns, "split"]
    return source[columns].rename(columns={settings.word_column: "word"})


def prepare_embedding_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    inputs = rows.rename(columns={"word_id": "response_id", "word": "response"})
    inputs["response_sha256"] = inputs["response"].map(compute_stable_hash)
    return inputs[["response_id", "response", "response_sha256"]]
