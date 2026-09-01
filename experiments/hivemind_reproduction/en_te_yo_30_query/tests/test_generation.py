import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    GENERATION_COLUMNS,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.generate_responses import (
    GenerationSettings,
    finalize_responses,
    prepare_generation_requests,
    select_eval_prompts,
    submit_generations,
    validate_generation_requests,
)
from lib.generate_with_gemini import parse_batch_results
from experiments.hivemind_reproduction.en_te_yo_30_query.resolve_paths import ExperimentPaths
from experiments.hivemind_reproduction.en_te_yo_30_query.translate_queries import (
    translate_prompts,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.translate_responses import (
    translate_responses,
)


def make_eval_rows(count: int = 20) -> list[dict[str, str]]:
    return [{"query": f"Open-ended evaluation prompt {index}."} for index in range(count)]


def write_generation_config(
    paths: ExperimentPaths,
    models: list[str],
    samples: int,
    approved: bool = False,
) -> GenerationSettings:
    raw = {
        "models": models,
        "responses_per_prompt": samples,
        "system_instructions": {
            code: f"test instruction for {code}" for code in ("en", "te", "yo")
        },
        "generation_config": {"max_output_tokens": 100},
        "user_approved_for_api": approved,
    }
    paths.generation_config.write_text(json.dumps(raw))
    return GenerationSettings(
        tuple(models),
        samples,
        raw["system_instructions"],
        raw["generation_config"],
        approved,
    )


def test_selects_seeded_eval_prompts() -> None:
    selected = select_eval_prompts(make_eval_rows(), count=8, seed=7)
    repeated = select_eval_prompts(make_eval_rows(), count=8, seed=7)

    assert len(selected) == 8
    assert selected["prompt_id"].is_unique
    assert selected["query_en"].tolist() == repeated["query_en"].tolist()
    assert selected["source_row_index"].is_unique
    assert selected["source_row_index"].ge(10).all()


def write_translation_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
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


def test_gemini_access_requires_explicit_confirmation(tmp_path: Path) -> None:
    args = SimpleNamespace(
        confirm_api=False,
        output_dir=str(tmp_path),
        models=["gemini-test"],
    )

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_generations(args)


def test_gemini_submission_requires_user_config_approval(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    write_generation_config(paths, ["gemini-test"], 1)

    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_generations(
            SimpleNamespace(
                confirm_api=True,
                output_dir=str(tmp_path),
                config=str(paths.generation_config),
            )
        )


def test_prepares_expected_request_counts(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    settings = write_generation_config(paths, ["gemini-test"], 3)
    source = select_eval_prompts(make_eval_rows(), count=6, seed=7)
    prompts = pd.concat(
        [
            source.assign(language_code=code, language=language, query=source["query_en"])
            for code, language in {"en": "English", "te": "Telugu", "yo": "Yoruba"}.items()
        ],
        ignore_index=True,
    )
    generations = prepare_generation_requests(prompts, "gemini-test", paths, settings)

    assert generations == 54
    assert len(paths.generation_manifest("gemini-test").read_text().splitlines()) == 55


def test_rejects_requests_that_do_not_match_current_config(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    settings = write_generation_config(paths, ["gemini-test"], 1, approved=True)
    prompts = pd.DataFrame(
        [
            {
                "prompt_id": "prompt-001",
                "query": "query",
                "query_en": "query",
                "language_code": "en",
                "language": "English",
            }
        ]
    )
    prepare_generation_requests(prompts, "gemini-test", paths, settings)
    changed = GenerationSettings(
        settings.models,
        settings.responses_per_prompt,
        settings.system_instructions,
        {"temperature": 0.5},
        True,
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_generation_requests(paths, "gemini-test", changed)


def test_translates_locally_and_resumes_from_checkpoint(tmp_path: Path) -> None:
    class Translator:
        def __init__(self) -> None:
            self.requests = []

        def translate(self, texts, source_language, target_language):
            self.requests.append((texts, source_language, target_language))
            return [f"{target_language}: {text}" for text in texts]

    paths = ExperimentPaths(tmp_path)
    translation_config = tmp_path / "translation_config.json"
    write_translation_config(translation_config)
    source = select_eval_prompts(make_eval_rows(), count=6, seed=7)
    source.to_csv(paths.source_prompts, index=False)
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        translation_config=str(translation_config),
    )
    translator = Translator()

    translate_prompts(args, translator)

    prompts = pd.read_csv(paths.multilingual_prompts)
    checkpoint = pd.read_csv(paths.translation_checkpoint)
    assert len(translator.requests) == 2
    assert len(checkpoint) == 12
    assert len(prompts) == 18
    assert set(prompts["language_code"]) == {"en", "te", "yo"}

    resumed_translator = Translator()
    translate_prompts(args, resumed_translator)
    assert not resumed_translator.requests


def test_translates_responses_to_english_and_resumes_from_checkpoint(
    tmp_path: Path,
) -> None:
    class Translator:
        def __init__(self) -> None:
            self.requests = []

        def translate(self, texts, source_language, target_language):
            self.requests.append((texts, source_language, target_language))
            return [f"English: {text}" for text in texts]

    generations = []
    for code, language in {"en": "English", "te": "Telugu", "yo": "Yoruba"}.items():
        for sample_idx in (1, 2):
            translated = code != "en"
            generations.append(
                {
                    "generation_id": f"g-{code}-{sample_idx}",
                    "query_id": "q001",
                    "query_tag": "test-query",
                    "sample_idx": sample_idx,
                    "base_query": "query-en",
                    "base_language_code": "en",
                    "base_language": "English",
                    "translated_query": f"query-{code}" if translated else None,
                    "translated_language_code": code if translated else None,
                    "translated_language": language if translated else None,
                    "query_translation_provider": (
                        "test-translator" if translated else None
                    ),
                    "query_translation_model": "test-model" if translated else None,
                    "model_response": f"response-{code}-{sample_idx}",
                    "response_translation": None,
                    "response_translation_language_code": None,
                    "response_translation_provider": None,
                    "provider": "test-provider",
                    "model_version": "model-a",
                    "generation_config": "config",
                    "system_instruction": None,
                }
            )
    response_path = tmp_path / "generations.csv"
    checkpoint_path = tmp_path / "translations.csv"
    output_path = tmp_path / "generations_translated.csv"
    translation_config = tmp_path / "translation_config.json"
    write_translation_config(translation_config)
    pd.DataFrame(generations)[GENERATION_COLUMNS].to_csv(response_path, index=False)
    args = SimpleNamespace(
        input=str(response_path),
        checkpoint=str(checkpoint_path),
        output=str(output_path),
        config=str(translation_config),
        dry_run=False,
    )
    translator = Translator()

    translate_responses(args, translator)

    translated = pd.read_csv(output_path)
    assert len(translator.requests) == 2
    assert {request[1] for request in translator.requests} == {"te", "yo"}
    assert all(request[2] == "en" for request in translator.requests)
    base_rows = translated["translated_language_code"].isna()
    assert translated.loc[base_rows, "response_translation"].isna().all()
    assert translated.loc[~base_rows, "response_translation"].notna().all()
    resumed_translator = Translator()
    translate_responses(args, resumed_translator)
    assert not resumed_translator.requests


def test_parses_batch_successes_and_failures(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    records = [
        {
            "key": "request-1",
            "response": {
                "candidates": [{"content": {"parts": [{"text": "answer"}]}}]
            },
        },
        {"key": "request-2", "status": {"code": 13, "message": "failed"}},
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    successes, failures = parse_batch_results(results)

    assert successes == {"request-1": "answer"}
    assert failures[0]["request_key"] == "request-2"


def test_finalization_preserves_archival_response_schema(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path)
    prompts = pd.DataFrame(
        [
            {
                "prompt_id": "prompt-001",
                "query": f"query-{code}",
                "query_en": "query-en",
                "language_code": code,
                "language": language,
            }
            for code, language in {"en": "English", "te": "Telugu", "yo": "Yoruba"}.items()
        ]
    )
    models = ["gemini-flash", "gemini-pro"]
    settings = write_generation_config(paths, models, 1)
    paths.experiment_manifest.write_text(json.dumps({"num_prompts": 1}))
    for model in models:
        prepare_generation_requests(prompts, model, paths, settings)
        manifest = pd.read_csv(paths.generation_manifest(model))
        output = [
            {
                "key": row.request_key,
                "response": {
                    "candidates": [{"content": {"parts": [{"text": f"answer-{row.request_key}"}]}}]
                },
            }
            for row in manifest.itertuples(index=False)
        ]
        paths.generation_results(model).parent.mkdir(parents=True, exist_ok=True)
        paths.generation_results(model).write_text(
            "\n".join(json.dumps(record) for record in output) + "\n"
        )

    finalize_responses(
        SimpleNamespace(output_dir=str(tmp_path), config=str(paths.generation_config))
    )
    responses = pd.read_csv(paths.responses)

    assert list(responses.columns) == [
        "Unnamed: 0",
        "query",
        "response",
        "model",
        "generation_config",
        "sample_idx",
    ]
    assert len(responses) == 6
