import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.multimodal_thinking_experiment.generate_responses import (
    finalize_results,
    load_settings,
    prepare_requests,
    submit_requests,
    validate_requests,
)


def write_topics(tmp_path: Path) -> Path:
    path = tmp_path / "poetry_topics.json"
    path.write_text(
        json.dumps(
            {
                "SUBJECTS": {
                    "Love": ["Familial Love", "Doubt"],
                    "The Mind": ["Doubt"],
                },
                "OCCASIONS": ["Birth"],
                "HOLIDAYS": [],
                "EMOTIONS": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_config(tmp_path: Path, approved: bool = False) -> Path:
    path = tmp_path / "generation_config.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "multimodal_thinking_experiment",
                "method_id": "test_method",
                "output_filename": "test_generations.csv",
                "user_approved_for_api": approved,
                "topics_path": str(write_topics(tmp_path)),
                "responses_per_topic": 1,
                "prompt_template": "Write a poem on the following topic: {TOPIC}",
                "text": {
                    "model": "gemini-3.1-flash-lite",
                    "system_instruction": None,
                    "conditions": [
                        {
                            "condition_id": "minimal",
                            "generation_config": {
                                "max_output_tokens": 16384,
                                "thinking_config": {
                                    "include_thoughts": True,
                                    "thinking_level": "minimal",
                                },
                            },
                        },
                        {
                            "condition_id": "high",
                            "generation_config": {
                                "max_output_tokens": 16384,
                                "thinking_config": {
                                    "include_thoughts": True,
                                    "thinking_level": "high",
                                },
                            },
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def make_args(tmp_path: Path, approved: bool = False) -> Namespace:
    return Namespace(
        config=str(write_config(tmp_path, approved)),
        runtime_root=str(tmp_path / "runtime"),
        results_root=str(tmp_path / "results"),
        confirm_api=False,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


@pytest.mark.parametrize(
    ("config_name", "method_id", "output_filename", "prompt_template"),
    [
        (
            "generation_config.json",
            "poetry_basic",
            "poetry_basic_generations.csv",
            "Write a poem on the following topic: {TOPIC}",
        ),
        (
            "poetry_vs5_generation_config.json",
            "poetry_vs5",
            "poetry_vs5_generations.csv",
            "Write a poem about {TOPIC}. Generate 5 responses and their "
            "corresponding probabilities. Respond in JSON format.",
        ),
        (
            "poetry_ssot_generation_config.json",
            "poetry_ssot",
            "poetry_ssot_generations.csv",
            "Generate a random string, then manipulate it to generate a diverse "
            "response. Write a poem about {TOPIC}.",
        ),
    ],
)
def test_full_configs_prepare_4060_requests(
    tmp_path: Path,
    config_name: str,
    method_id: str,
    output_filename: str,
    prompt_template: str,
) -> None:
    config = Path("experiments/multimodal_thinking_experiment") / config_name
    settings = load_settings(config)
    assert settings.method_id == method_id
    assert settings.output_filename == output_filename
    assert settings.prompt_template == prompt_template
    assert settings.responses_per_topic == 10
    assert settings.model == "gemini-3.1-flash-lite"
    assert [condition.condition_id for condition in settings.conditions] == [
        "minimal",
        "high",
    ]

    runtime_root = tmp_path / method_id
    args = Namespace(config=str(config), runtime_root=str(runtime_root))
    prepare_requests(args)
    requests = [
        json.loads(line)
        for line in (runtime_root / "generation_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifest = read_csv(runtime_root / "generation_manifest.csv")

    assert len(requests) == 4_060
    assert len(manifest) == 4_060
    assert len({row["topic_id"] for row in manifest}) == 203
    assert {row["topic"] for row in manifest} >= {"Love", "Familial Love"}
    generation_config = requests[0]["request"]["generationConfig"]
    assert generation_config == {
        "maxOutputTokens": 16384,
        "thinkingConfig": {
            "includeThoughts": True,
            "thinkingLevel": "MINIMAL",
        },
    }
    assert not {"temperature", "topP", "topK"} & set(generation_config)
    assert requests[0]["request"]["contents"][0]["parts"][0]["text"] == (
        prompt_template.format(TOPIC=manifest[0]["topic"])
    )
    validate_requests(runtime_root, settings)


def test_prepare_deduplicates_topic_text_and_retains_paths(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)
    manifest = read_csv(Path(args.runtime_root) / "generation_manifest.csv")
    doubt_rows = [row for row in manifest if row["topic"] == "Doubt"]

    assert len(manifest) == 10
    assert len(doubt_rows) == 2
    assert json.loads(doubt_rows[0]["hierarchy_paths"]) == [
        ["SUBJECTS", "Love", "Doubt"],
        ["SUBJECTS", "The Mind", "Doubt"],
    ]


def test_submission_requires_confirmation_and_user_approval(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_requests(args)

    args.confirm_api = True
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_requests(args)


def test_finalize_writes_responses_usage_and_failures(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)
    runtime_root = Path(args.runtime_root)
    manifest = read_csv(runtime_root / "generation_manifest.csv")
    failed_key = manifest[0]["request_key"]
    results = []
    for row in manifest:
        finish_reason = "MAX_TOKENS" if row["request_key"] == failed_key else "STOP"
        results.append(
            {
                "key": row["request_key"],
                "response": {
                    "candidates": [
                        {
                            "finishReason": finish_reason,
                            "content": {
                                "parts": [
                                    {"thought": True, "text": "Plan the poem."},
                                    {"text": f"Poem about {row['topic']}"},
                                ]
                            },
                        }
                    ],
                    "modelVersion": "gemini-3.1-flash-lite-001",
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 20,
                        "thoughtsTokenCount": 30,
                        "totalTokenCount": 62,
                    },
                },
            }
        )
    (runtime_root / "generation_results.jsonl").write_text(
        "\n".join(json.dumps(result) for result in results) + "\n",
        encoding="utf-8",
    )

    finalize_results(args)

    output = read_csv(Path(args.results_root) / "data/test_generations.csv")
    assert len(output) == 9
    assert output[0]["thinking"] == "Plan the poem."
    assert output[0]["model_version"] == "gemini-3.1-flash-lite-001"
    assert output[0]["thought_token_count"] == "30"
    failures = read_csv(
        Path(args.results_root) / "data/test_method_failed_requests.csv"
    )
    assert failures == [
        {
            "request_key": failed_key,
            "method_id": "test_method",
            "topic_id": manifest[0]["topic_id"],
            "topic": manifest[0]["topic"],
            "thinking_level": manifest[0]["thinking_level"],
            "sample_idx": "1",
            "error": "Response finished with MAX_TOKENS",
        }
    ]
    metadata = json.loads(
        (Path(args.results_root) / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {
        "experiment_id": "multimodal_thinking_experiment",
        "embedding_model": None,
        "max_token_length": 16384,
    }
