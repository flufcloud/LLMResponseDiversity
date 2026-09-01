import pandas as pd

from lib.translate_locally import (
    LocalTranslationSettings,
    get_local_translator,
    get_translation_batches,
    split_text,
    translate_texts,
)


def test_gets_translation_batches_within_limits() -> None:
    rows = pd.DataFrame({"text": ["1234", "5678", "9"]})

    batches = list(get_translation_batches(rows, "text", 8, 2))

    assert [batch["text"].tolist() for batch in batches] == [
        ["1234", "5678"],
        ["9"],
    ]


def test_splits_long_text_at_word_boundaries() -> None:
    assert split_text("one two three", 8) == ["one two", "three"]


def test_translates_texts_with_local_translator() -> None:
    class Translator:
        def translate(self, texts, source_language, target_language):
            return [f"translated: {text}" for text in texts]

    translator = Translator()
    translated = translate_texts(translator, ["source"], "yo")

    assert translated == ["translated: source"]


def test_reuses_injected_translator_without_loading_model() -> None:
    translator = object()
    settings = LocalTranslationSettings(
        model_id="model",
        revision="revision",
        device="cpu",
        batch_size=1,
        max_batch_characters=100,
        max_batch_items=1,
        max_chunk_characters=100,
        max_input_tokens=100,
        max_output_tokens=100,
    )

    assert get_local_translator(settings, translator) is translator
