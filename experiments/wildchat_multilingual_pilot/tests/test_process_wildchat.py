import json
from argparse import Namespace

import polars as pl

from experiments.wildchat_multilingual_pilot.process_wildchat import (
    CONFIG_PATH,
    TranslationSettings,
    build_translation_rows,
    filter_language,
    finalize_outputs,
    flatten_conversations,
    load_translation_settings,
    open_translation_store,
    process_dataset,
    translate_missing_texts,
)
from lib.translate_locally import MarianSettings, split_text


CUDA_CONFIG_PATH = CONFIG_PATH.with_name("translation_config_cuda.json")


def test_pilot_configuration() -> None:
    settings = load_translation_settings(CONFIG_PATH)

    assert settings.source_row_limit == 100_000
    assert settings.dataset_batch_size == 10_000
    assert settings.conversation_turn_limit == 1
    assert settings.language_ids == {
        "Spanish": "es",
        "Chinese": "zh",
        "Telugu": "te",
        "Yoruba": "yo",
    }
    assert settings.marian.model_id == "Helsinki-NLP/opus-mt-mul-en"
    assert settings.marian.device == "mps"
    assert settings.marian.dtype == "float32"


def test_cuda_configuration() -> None:
    settings = load_translation_settings(CUDA_CONFIG_PATH)

    assert settings.marian.device == "cuda"
    assert settings.marian.dtype == "float16"
    assert settings.marian.model_batch_size == 16


def get_settings(max_chunk_characters: int = 100) -> TranslationSettings:
    return TranslationSettings(
        source_dataset="dataset",
        source_split="train",
        source_row_limit=100,
        dataset_batch_size=100,
        conversation_turn_limit=1,
        target_language="en",
        language_ids={"Spanish": "es"},
        marian=MarianSettings(
            model_id="Helsinki-NLP/opus-mt-mul-en",
            revision="revision",
            device="cpu",
            dtype="float32",
            model_batch_size=2,
            max_batch_characters=100,
            max_batch_items=10,
            max_chunk_characters=max_chunk_characters,
            max_input_tokens=50,
            max_output_tokens=50,
        ),
    )


def test_filters_flattens_and_builds_translation_rows(monkeypatch):
    monkeypatch.setattr(
        "experiments.wildchat_multilingual_pilot.process_wildchat.matches_language",
        lambda text, language_id: text != "wrong language",
    )
    frame = pl.DataFrame(
        {
            "source_row": [10, 11, 12],
            "language": ["Spanish", "Spanish", "Portuguese"],
            "conversation": [
                [
                    {"role": "user", "content": "hola"},
                    {"role": "assistant", "content": "respuesta"},
                ],
                [{"role": "user", "content": "wrong language"}],
                [{"role": "user", "content": "ola"}],
            ],
        }
    )

    filtered = filter_language(frame, "Spanish", "es")
    turns = flatten_conversations(filtered, 1)
    translated = build_translation_rows(
        turns, {"hola": "hello", "respuesta": "answer"}
    )

    assert translated.to_dicts() == [
        {
            "source_row": 10,
            "turn_idx": 0,
            "untranslated": "hola",
            "role": "user",
            "translated": "hello",
        }
    ]


def test_reuses_cached_translations_and_deduplicates_text(tmp_path):
    connection = open_translation_store(tmp_path / "translations.sqlite")
    calls = []

    def request_batch(texts):
        calls.append(list(texts))
        return [f"en:{text}" for text in texts]

    try:
        first = translate_missing_texts(
            ["hola", "respuesta", "hola"],
            "es",
            connection,
            request_batch,
            get_settings(),
        )
        second = translate_missing_texts(
            ["respuesta", "hola"],
            "es",
            connection,
            request_batch,
            get_settings(),
        )
    finally:
        connection.close()

    assert first == {"hola": "en:hola", "respuesta": "en:respuesta"}
    assert second == {"respuesta": "en:respuesta", "hola": "en:hola"}
    assert calls == [["hola", "respuesta"]]


def test_finalizes_legacy_conversation_shard(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    shard_dir = checkpoint_dir / "checkpoints" / "local-marian"
    shard_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "source_row": [10],
            "conversation": [
                json.dumps(
                    [
                        {
                            "turn_idx": 0,
                            "untranslated": "hola",
                            "role": "user",
                            "translated": "hello",
                        }
                    ]
                )
            ],
        }
    ).write_parquet(shard_dir / "Spanish-00000.parquet")

    finalize_outputs(checkpoint_dir, tmp_path / "output", ["Spanish"])

    assert pl.read_csv(tmp_path / "output" / "Spanish.csv").to_dicts() == [
        {
            "turn_idx": 0,
            "untranslated": "hola",
            "role": "user",
            "translated": "hello",
        }
    ]


def test_splits_oversized_text_at_word_boundaries(tmp_path):
    connection = open_translation_store(tmp_path / "translations.sqlite")
    calls = []

    def request_batch(texts):
        calls.append(list(texts))
        return [f"en:{text}" for text in texts]

    try:
        translated = translate_missing_texts(
            ["hola respuesta larga"],
            "es",
            connection,
            request_batch,
            get_settings(max_chunk_characters=10),
        )
    finally:
        connection.close()

    assert split_text("hola respuesta larga", 10) == [
        "hola",
        "respuesta",
        "larga",
    ]
    assert translated == {"hola respuesta larga": "en:hola\nen:respuesta\nen:larga"}
    assert calls == [["hola", "respuesta", "larga"]]


def test_processes_initial_queries_and_resumes(tmp_path, monkeypatch):
    config = {
        "source_dataset": "dataset",
        "source_split": "train",
        "source_row_limit": 100,
        "dataset_batch_size": 100,
        "conversation_turn_limit": 1,
        "target_language": "en",
        "languages": [{"code": "es", "name": "Spanish"}],
        "translation_provider": "local-marian",
        "translation_model": "Helsinki-NLP/opus-mt-mul-en",
        "translation_revision": "848eae0c1676cfce9bb791c200e8228e5a6396ff",
        "device": "cpu",
        "dtype": "float32",
        "model_batch_size": 2,
        "max_batch_characters": 100,
        "max_batch_items": 10,
        "max_chunk_characters": 100,
        "max_input_tokens": 50,
        "max_output_tokens": 50,
    }
    config_path = tmp_path / "translation_config.json"
    config_path.write_text(json.dumps(config))
    args = Namespace(
        config=config_path,
        output_dir=tmp_path / "output",
        checkpoint_dir=tmp_path / "checkpoint",
        max_dataset_batches=None,
    )

    class Stream:
        def iter(self, batch_size):
            assert batch_size == 100
            yield {
                "language": ["Spanish", "English"],
                "conversation": [
                    [
                        {"role": "user", "content": "hola"},
                        {"role": "assistant", "content": "respuesta"},
                    ],
                    [{"role": "user", "content": "hello"}],
                ],
            }
            yield {
                "language": ["Spanish"],
                "conversation": [[{"role": "user", "content": "adios"}]],
            }

    class Translator:
        def __init__(self):
            self.requests = []

        def translate(self, texts, source_language, target_language):
            self.requests.append((texts, source_language, target_language))
            return [f"en:{text}" for text in texts]

    monkeypatch.setattr(
        "experiments.wildchat_multilingual_pilot.process_wildchat.matches_language",
        lambda text, language_id: True,
    )
    translator = Translator()

    process_dataset(args, translator=translator, stream=Stream())
    process_dataset(args, translator=translator, stream=Stream())

    output = pl.read_csv(args.output_dir / "Spanish.csv")
    assert output.to_dicts() == [
        {
            "turn_idx": 0,
            "untranslated": "hola",
            "role": "user",
            "translated": "en:hola",
        }
    ]
    assert translator.requests == [(["hola"], "es", "en")]
