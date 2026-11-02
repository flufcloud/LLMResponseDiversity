import base64
import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.acfr_collapse.generate_responses import (
    finalize_results,
    load_settings,
    prepare_requests,
    submit_requests,
    validate_requests,
)


def write_source(tmp_path: Path) -> Path:
    source_path = tmp_path / "words.csv"
    source_path.write_text("Word\nability\nforest\njoy\n", encoding="utf-8")
    return source_path


def write_config(tmp_path: Path, approved: bool = False) -> Path:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_id": "acfr_collapse",
                "user_approved_for_api": approved,
                "source": {
                    "path": str(write_source(tmp_path)),
                    "word_column": "Word",
                    "word_sample_count": 2,
                    "sampling": "random_without_replacement",
                    "random_seed": None,
                },
                "responses_per_prompt": 2,
                "text": {
                    "model": "text-model",
                    "system_instruction": None,
                    "generation_config": {},
                },
                "image": {
                    "model": "image-model",
                    "system_instruction": None,
                    "generation_config": {"response_modalities": ["IMAGE"]},
                },
                "prompts": {
                    "text": [
                        {
                            "template_id": "poem",
                            "template": "Write about {word}",
                        }
                    ],
                    "image": [
                        {
                            "template_id": "painting",
                            "template": "Paint {word}",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def make_args(tmp_path: Path, approved: bool = False) -> Namespace:
    return Namespace(
        config=str(write_config(tmp_path, approved)),
        runtime_root=str(tmp_path / "runtime"),
        results_root=str(tmp_path / "results"),
        modality="text",
        confirm_api=False,
    )


def test_approved_config_has_expected_scope_and_prompts(tmp_path: Path) -> None:
    settings = load_settings(Path("experiments/acfr_collapse/generation_config.json"))
    assert settings.source["word_sample_count"] == 150
    assert settings.source["random_seed"] is None
    assert settings.responses_per_prompt == 5
    assert settings.text["model"] == "gemini-3.1-flash-lite"
    assert settings.image["model"] == "gemini-3.1-flash-lite-image"
    assert settings.text["generation_config"] == {
        "max_output_tokens": 2048,
        "thinking_config": {
            "include_thoughts": True,
            "thinking_level": "medium",
        },
    }
    assert settings.image["generation_config"] == {
        "max_output_tokens": 2048,
        "response_modalities": ["IMAGE"],
        "thinking_config": {"include_thoughts": True},
    }

    args = Namespace(
        config="experiments/acfr_collapse/generation_config.json",
        runtime_root=str(tmp_path / "runtime"),
    )
    prepare_requests(args)
    manifest = list(
        csv.DictReader(
            (tmp_path / "runtime/generation_manifest.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(manifest) == 4_500
    assert len({row["word"] for row in manifest}) == 150
    assert {row["sample_idx"] for row in manifest} == {"1", "2", "3", "4", "5"}


def test_prepare_and_validate_requests(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)
    runtime_root = Path(args.runtime_root)

    text_requests = [
        json.loads(line)
        for line in (runtime_root / "text_requests.jsonl").read_text().splitlines()
    ]
    image_requests = [
        json.loads(line)
        for line in (runtime_root / "image_requests.jsonl").read_text().splitlines()
    ]
    assert len(text_requests) == 4
    assert len(image_requests) == 4
    assert image_requests[0]["request"]["generationConfig"] == {
        "responseModalities": ["IMAGE"]
    }
    validate_requests(runtime_root, load_settings(Path(args.config)))


def test_submission_requires_confirmation_and_user_approval(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_requests(args)

    args.confirm_api = True
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_requests(args)


def test_finalize_writes_one_csv_and_hierarchical_images(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)
    runtime_root = Path(args.runtime_root)
    manifest = list(
        csv.DictReader(
            (runtime_root / "generation_manifest.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    text_rows = [row for row in manifest if row["modality"] == "text"]
    image_rows = [row for row in manifest if row["modality"] == "image"]

    (runtime_root / "text_results.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "key": row["request_key"],
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"thought": True, "text": "text thought"},
                                        {"text": "text response"},
                                    ]
                                }
                            }
                        ]
                    },
                }
            )
            for row in text_rows
        )
        + "\n",
        encoding="utf-8",
    )
    encoded_image = base64.b64encode(b"png-data").decode()
    failed_image_key = image_rows[0]["request_key"]
    (runtime_root / "image_results.jsonl").write_text(
        "\n".join(
            json.dumps(
                ({
                    "key": row["request_key"],
                    "response": {
                        "candidates": [
                            {"content": {}, "finishReason": "NO_IMAGE"}
                        ]
                    },
                } if row["request_key"] == failed_image_key else {
                    "key": row["request_key"],
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"thought": True, "text": "image thought"},
                                        {
                                            "inlineData": {
                                                "mimeType": "image/png",
                                                "data": encoded_image,
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                })
            )
            for row in image_rows
        )
        + "\n",
        encoding="utf-8",
    )

    finalize_results(args)

    output_path = Path(args.results_root) / "data/generations.csv"
    with output_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 7
    assert list(rows[0]) == [
        "word",
        "prompt",
        "modality",
        "sample_idx",
        "response",
        "thinking",
    ]
    text_row = next(row for row in rows if row["modality"] == "text")
    assert text_row["thinking"] == "text thought"
    image_row = next(row for row in rows if row["modality"] == "image")
    assert image_row["thinking"] == "image thought"
    assert Path(image_row["response"]).read_bytes() == b"png-data"
    assert image_row["response"].endswith(
        f"/{image_row['word']}/painting/sample_{int(image_row['sample_idx']):02d}.png"
    )
    with (Path(args.results_root) / "data/failed_requests.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        failures = list(csv.DictReader(source))
    assert failures == [
        {
            "modality": "image",
            "request_key": failed_image_key,
            "error": "Response contained 0 final images",
        }
    ]
