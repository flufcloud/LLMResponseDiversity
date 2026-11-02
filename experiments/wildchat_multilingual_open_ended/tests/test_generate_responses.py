import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.wildchat_multilingual_open_ended.generate_responses import (
    finalize_results,
    load_settings,
    prepare_requests,
    submit_requests,
    validate_input_requests,
    validate_requests,
)


OUTPUT_COLUMNS = [
    "original_query",
    "revised_query",
    "meaningful_information",
    "greeting_model_inquiry",
    "response_type",
    "language",
]


def write_settings(tmp_path: Path, approved: bool = False) -> SimpleNamespace:
    input_dir = tmp_path / "pilot"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "untranslated": ["¿Qué hora es?", "你好", "你好"],
            "translated": ["What time is it?", "Hello", "Hello"],
        }
    ).to_csv(input_dir / "Spanish.csv", index=False)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Classify {USER_QUERY}", encoding="utf-8")
    experiment_config = tmp_path / "experiment_config.json"
    experiment_config.write_text(
        json.dumps(
            {
                "experiment_id": "test-experiment",
                "inputs": [
                    {"language": "Spanish", "path": str(input_dir / "Spanish.csv")}
                ],
                "input_column": "untranslated",
                "output_root": str(tmp_path / "results"),
                "raw_output_dir": str(tmp_path / "results/data/classifier_outputs"),
                "output_filename_suffix": "_classifier_output",
                "output_columns": OUTPUT_COLUMNS,
            }
        ),
        encoding="utf-8",
    )
    generation_config = tmp_path / "generation_config.json"
    generation_config.write_text(
        json.dumps(
            {
                "models": ["gemini-test"],
                "responses_per_query": 1,
                "system_instruction_template_path": str(prompt_path),
                "user_query_placeholder": "{USER_QUERY}",
                "generation_config": {"temperature": 0},
                "user_approved_for_api": approved,
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        experiment_config=str(experiment_config),
        generation_config=str(generation_config),
        runtime_dir=str(tmp_path / "runtime"),
    )


def test_prepares_untranslated_queries(tmp_path: Path) -> None:
    args = write_settings(tmp_path)

    prepare_requests(args)

    requests_path = tmp_path / "runtime/generation_requests/spanish.jsonl"
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    assert len(requests) == 2
    assert requests[0]["request"]["contents"][0]["parts"][0]["text"] == "¿Qué hora es?"
    instruction = requests[0]["request"]["system_instruction"]["parts"][0]["text"]
    assert instruction == 'Classify "¿Qué hora es?"'
    assert "What time is it?" not in requests_path.read_text()
    validate_requests(args)


def test_deduplicates_queries_across_inputs(tmp_path: Path) -> None:
    args = write_settings(tmp_path)
    chinese_path = tmp_path / "pilot/Chinese.csv"
    pd.DataFrame(
        {
            "untranslated": ["你好", "另一个"],
            "translated": ["Hello", "Another"],
        }
    ).to_csv(chinese_path, index=False)
    config_path = Path(args.experiment_config)
    config = json.loads(config_path.read_text())
    config["inputs"].append({"language": "Chinese", "path": str(chinese_path)})
    config_path.write_text(json.dumps(config))

    prepare_requests(args)
    experiment, _ = load_settings(args)

    spanish_requests = experiment.paths.generation_requests("Spanish").read_text().splitlines()
    chinese_requests = experiment.paths.generation_requests("Chinese").read_text().splitlines()
    chinese_manifest = pd.read_csv(experiment.paths.generation_manifest("Chinese"))
    assert len(spanish_requests) == 2
    assert len(chinese_requests) == 1
    assert chinese_manifest["result_source"].tolist() == ["Spanish", "Chinese"]
    validate_requests(args)


def test_validation_detects_prompt_changes(tmp_path: Path) -> None:
    args = write_settings(tmp_path)
    prepare_requests(args)
    experiment, generation = load_settings(args)
    generation.system_instruction_template_path.write_text("Changed {USER_QUERY}")

    with pytest.raises(ValueError, match="approved configuration"):
        validate_input_requests(experiment.inputs[0], experiment, generation)


def test_submission_requires_confirmation_and_approval(tmp_path: Path) -> None:
    args = write_settings(tmp_path)
    args.confirm_api = False

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_requests(args)

    args.confirm_api = True
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_requests(args)


def test_finalizes_json_and_csv_outputs(tmp_path: Path) -> None:
    args = write_settings(tmp_path)
    prepare_requests(args)
    experiment, _ = load_settings(args)
    manifest = pd.read_csv(experiment.paths.generation_manifest("Spanish"))
    results = []
    for row in manifest.itertuples(index=False):
        output = {
            "original_query": row.original_query,
            "revised_query": None,
            "meaningful_information": {
                "reasoning": "The query is meaningful.",
                "classification": True,
            },
            "greeting_model_inquiry": {
                "reasoning": "The query is not a greeting.",
                "classification": False,
            },
            "response_type": {
                "reasoning": "It has one answer.",
                "classification": "Single",
            },
            "language": "Spanish",
        }
        results.append(
            {
                "key": row.request_key,
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "```json\n"
                                        + json.dumps(output, ensure_ascii=False)
                                        + "\n```"
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        )
    result_path = experiment.paths.generation_results("Spanish")
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
        encoding="utf-8",
    )

    finalize_results(args)

    output_path = tmp_path / "results/data/Spanish_classifier_output.csv"
    output = pd.read_csv(output_path)
    assert output.columns.tolist() == OUTPUT_COLUMNS
    assert output["original_query"].tolist() == ["¿Qué hora es?", "你好"]
    assert json.loads(output.loc[0, "response_type"])["classification"] == "Single"
    raw_dir = tmp_path / "results/data/classifier_outputs"
    raw_files = sorted(raw_dir.glob("*.json"))
    assert len(raw_files) == 2


def test_finalization_skips_invalid_classifier_output(tmp_path: Path) -> None:
    args = write_settings(tmp_path)
    prepare_requests(args)
    experiment, _ = load_settings(args)
    manifest = pd.read_csv(experiment.paths.generation_manifest("Spanish"))
    result_path = experiment.paths.generation_results("Spanish")
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": row.request_key,
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": "not json"}]}}]
                    },
                }
            )
            for row in manifest.itertuples(index=False)
        )
        + "\n"
    )

    finalize_results(args)

    failures = pd.read_csv(tmp_path / "results/data/failed_requests.csv")
    output = pd.read_csv(tmp_path / "results/data/Spanish_classifier_output.csv")
    assert len(failures) == 2
    assert output.empty


def test_finalization_accepts_original_query_mismatch(tmp_path: Path) -> None:
    args = write_settings(tmp_path)
    prepare_requests(args)
    experiment, _ = load_settings(args)
    manifest = pd.read_csv(experiment.paths.generation_manifest("Spanish"))
    results = []
    for row in manifest.itertuples(index=False):
        output = {
            "original_query": row.original_query.strip() + " changed",
            "revised_query": None,
            "meaningful_information": None,
            "greeting_model_inquiry": None,
            "response_type": None,
            "language": "Spanish",
        }
        results.append(
            {
                "key": row.request_key,
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": json.dumps(output)}]}}
                    ]
                },
            }
        )
    result_path = experiment.paths.generation_results("Spanish")
    result_path.parent.mkdir(parents=True)
    result_path.write_text("\n".join(json.dumps(row) for row in results) + "\n")

    finalize_results(args)

    output = pd.read_csv(tmp_path / "results/data/Spanish_classifier_output.csv")
    assert output["original_query"].str.endswith(" changed").all()
