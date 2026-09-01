import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from experiments.cross_lingual_embedding_validation.translate_queries import (
    prepare_query_rows,
    translate_queries,
)


def write_translation_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "source_dataset": "test/dataset",
                "source_split": "train",
                "query_count": 2,
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
                "max_batch_characters": 100,
                "max_batch_items": 2,
                "max_chunk_characters": 100,
                "max_input_tokens": 100,
                "max_output_tokens": 100,
            }
        )
    )


def test_prepares_all_source_queries_in_order() -> None:
    rows = [{"query": f"query {index}"} for index in range(1, 101)]

    queries = prepare_query_rows(rows, 100)

    assert queries.columns.tolist() == ["query_id", "query"]
    assert queries.iloc[0].to_dict() == {"query_id": "q001", "query": "query 1"}
    assert queries.iloc[-1].to_dict() == {"query_id": "q100", "query": "query 100"}


def test_translates_and_resumes_from_checkpoint(tmp_path: Path) -> None:
    class Translator:
        def __init__(self) -> None:
            self.requests = []

        def translate(self, texts, source_language, target_language):
            self.requests.append((texts, source_language, target_language))
            return [f"{target_language}: {text}" for text in texts]

    config = tmp_path / "translation_config.json"
    write_translation_config(config)
    source_rows = [{"query": "first"}, {"query": "second"}]
    args = SimpleNamespace(
        config=str(config),
        cache_dir=str(tmp_path / "huggingface"),
        checkpoint=str(tmp_path / "translation_checkpoint.csv"),
        output=str(tmp_path / "translated_queries.csv"),
    )
    translator = Translator()

    translate_queries(args, translator, source_rows)

    output = pd.read_csv(args.output)
    assert len(translator.requests) == 2
    assert output.columns.tolist() == ["query_id", "lang", "query"]
    assert len(output) == 6
    assert set(output["lang"]) == {"en", "fr", "yo"}
    assert {request[2] for request in translator.requests} == {"fr", "yo"}

    resumed_translator = Translator()
    translate_queries(args, resumed_translator, source_rows)
    assert not resumed_translator.requests
