from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def get_translation_batches(
    rows: pd.DataFrame,
    text_column: str,
    max_characters: int,
    max_items: int,
) -> Iterator[pd.DataFrame]:
    batches = get_text_batches(
        rows[[text_column]].itertuples(), max_characters, max_items
    )
    for batch in batches:
        yield rows.loc[[index for index, _ in batch]]


def get_text_batches(
    rows: Iterable[tuple[Any, str]],
    max_characters: int,
    max_items: int,
) -> Iterator[list[tuple[Any, str]]]:
    if max_characters < 1 or max_items < 1:
        raise ValueError("Translation batch limits must be positive")
    batch: list[tuple[Any, str]] = []
    characters = 0
    for index, text in rows:
        length = len(text)
        exceeds_limit = characters + length > max_characters
        if batch and (exceeds_limit or len(batch) == max_items):
            yield batch
            batch = []
            characters = 0
        batch.append((index, text))
        characters += length
    if batch:
        yield batch


def split_text(text: str, max_characters: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_characters, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
            if boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks
