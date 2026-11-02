"""Tests for templated prompt generation and materialization."""

import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.multimodal_thinking_experiment.generate_responses import (
    load_topic_paths,
)
from experiments.templated_prompts_compositionality.generate_responses import (
    finalize_results,
    get_adjectives,
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
                "SUBJECTS": {"Love": ["Familial Love"]},
                "OCCASIONS": ["Birth"],
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
                "experiment_id": "templated_prompts_compositionality",
                "user_approved_for_api": approved,
                "topics_path": str(write_topics(tmp_path)),
                "responses_per_pair": 2,
                "prompt_template": "Write a(n) {ADJECTIVE} poem about {TOPIC}.",
                "adjective_groups": {
                    "happy": ["joyful", "delightful"],
                    "sad": ["somber"],
                },
                "text": {
                    "model": "gemini-3.1-flash-lite",
                    "system_instruction": None,
                    "generation_config": {
                        "max_output_tokens": 2048,
                        "thinking_config": {
                            "include_thoughts": False,
                            "thinking_level": "minimal",
                        },
                    },
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


def test_full_config_has_expected_design() -> None:
    settings = load_settings(
        Path("experiments/templated_prompts_compositionality/generation_config.json")
    )
    topics = load_topic_paths(settings.topics_path)
    adjectives = get_adjectives(settings)

    assert settings.experiment_id == "templated_prompts_compositionality"
    assert settings.prompt_template == "Write a(n) {ADJECTIVE} poem about {TOPIC}."
    assert settings.model == "gemini-3.1-flash-lite"
    assert settings.responses_per_pair == 10
    assert len(topics) == 203
    assert len(adjectives) == 15
    assert len(topics) * len(adjectives) * settings.responses_per_pair == 30_450
    assert isinstance(settings.user_approved_for_api, bool)


def test_prepare_and_validate_requests(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)
    runtime_root = Path(args.runtime_root)
    manifest = read_csv(runtime_root / "generation_manifest.csv")
    requests = [
        json.loads(line)
        for line in (runtime_root / "generation_requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(manifest) == 18
    assert len(requests) == 18
    assert manifest[0]["adjective_group"] == "happy"
    assert manifest[0]["adjective"] == "joyful"
    assert manifest[0]["prompt"] == "Write a(n) joyful poem about Love."
    assert requests[0]["request"]["generationConfig"] == {
        "maxOutputTokens": 2048,
        "thinkingConfig": {
            "includeThoughts": False,
            "thinkingLevel": "MINIMAL",
        },
    }
    validate_requests(runtime_root, load_settings(Path(args.config)))


def test_submission_requires_confirmation_and_approval(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_requests(args)

    args.confirm_api = True
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_requests(args)


def test_finalize_writes_successes_failures_and_metadata(tmp_path: Path) -> None:
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
                                "parts": [{"text": f"Poem about {row['topic']}"}]
                            },
                        }
                    ],
                    "modelVersion": "gemini-3.1-flash-lite-001",
                    "usageMetadata": {
                        "promptTokenCount": 9,
                        "candidatesTokenCount": 20,
                        "totalTokenCount": 29,
                    },
                },
            }
        )
    (runtime_root / "generation_results.jsonl").write_text(
        "\n".join(json.dumps(result) for result in results) + "\n",
        encoding="utf-8",
    )

    finalize_results(args)

    output = read_csv(Path(args.results_root) / "data/generations.csv")
    failures = read_csv(Path(args.results_root) / "data/failed_requests.csv")
    metadata = json.loads(
        (Path(args.results_root) / "metadata.json").read_text(encoding="utf-8")
    )
    assert len(output) == 17
    assert output[0]["adjective_group"] == "happy"
    assert output[0]["adjective"] == "joyful"
    assert failures[0]["request_key"] == failed_key
    assert metadata == {
        "experiment_id": "templated_prompts_compositionality",
        "embedding_model": None,
        "max_token_length": 2048,
    }
