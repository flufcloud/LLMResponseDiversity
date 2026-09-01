import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from experiments.intralingual_multilingual_embedding_validation.translate_mrpc import (
    load_translation_settings,
    sample_mrpc_pairs,
    translate_mrpc,
)


def write_translation_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_dataset": "test/dataset",
                "source_subset": "mrpc",
                "source_split": "train",
                "required_label": 1,
                "sample_count": 2,
                "random_seed": 0,
                "source_language": "en",
                "languages": [
                    {"code": "en", "name": "English"},
                    {"code": "fr", "name": "French"},
                    {"code": "yo", "name": "Yoruba"},
                ],
                "translation_provider": "local-nllb-200-distilled-600m",
                "translation_model": "facebook/nllb-200-distilled-600M",
                "translation_revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
                "device": "cpu",
                "batch_size": 2,
                "max_batch_characters": 1000,
                "max_batch_items": 100,
                "max_chunk_characters": 100,
                "max_input_tokens": 100,
                "max_output_tokens": 100,
            }
        )
    )


def get_source_rows() -> list[dict]:
    return [
        {"idx": 10, "sentence1": "base 10", "sentence2": "para 10", "label": 1},
        {"idx": 11, "sentence1": "base 11", "sentence2": "para 11", "label": 0},
        {"idx": 12, "sentence1": "base 12", "sentence2": "para 12", "label": 1},
        {"idx": 13, "sentence1": "base 13", "sentence2": "para 13", "label": 1},
    ]


def test_samples_only_positive_pairs_reproducibly(tmp_path: Path) -> None:
    config = tmp_path / "translation_config.json"
    write_translation_config(config)
    settings = load_translation_settings(config)

    first = sample_mrpc_pairs(get_source_rows(), settings)
    second = sample_mrpc_pairs(get_source_rows(), settings)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2
    assert 11 not in set(first["mrpc_sentence_id"])


def test_translates_pairs_and_resumes(tmp_path: Path) -> None:
    class Translator:
        def __init__(self) -> None:
            self.requests = []

        def translate(self, texts, source_language, target_language):
            self.requests.append((texts, source_language, target_language))
            return [f"{target_language}: {text}" for text in texts]

    config = tmp_path / "translation_config.json"
    write_translation_config(config)
    args = SimpleNamespace(
        config=str(config),
        cache_dir=str(tmp_path / "huggingface"),
        checkpoint=str(tmp_path / "translation_checkpoint.csv"),
        output=str(tmp_path / "mrpc_translated.csv"),
    )
    translator = Translator()

    translate_mrpc(args, translator, get_source_rows())

    translated = pd.read_csv(args.output)
    assert len(translator.requests) == 2
    assert translated.columns.tolist() == [
        "index",
        "mrpc_sentence_id",
        "target_lang_identifier",
        "translated_base",
        "translated_paraphrase",
    ]
    assert len(translated) == 6
    assert set(translated["target_lang_identifier"]) == {"en", "fr", "yo"}
    assert translated["index"].tolist() == list(range(6))

    resumed_translator = Translator()
    translate_mrpc(args, resumed_translator, get_source_rows())
    assert not resumed_translator.requests
