import base64
import csv
import json
from argparse import Namespace

from experiments.text_image_concept_collapse.generate_image_responses import (
    build_image_request,
    finalize_images,
    prepare_requests,
)


def test_prepare_requests_uses_configured_prompts_and_sample_indices(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "responses_per_prompt": 2,
                "image": {
                    "model": "image-model",
                    "system_instruction": None,
                    "generation_config": {
                        "response_modalities": ["IMAGE"],
                        "thinking_config": {"include_thoughts": True},
                    },
                },
                "prompts": [
                    {"query_id": "nature", "image": "Create a nature image."}
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_root = tmp_path / "runtime"

    prepare_requests(
        Namespace(config=str(config_path), runtime_root=str(runtime_root))
    )

    requests = [
        json.loads(line)
        for line in (runtime_root / "image_response_batch.jsonl").read_text().splitlines()
    ]
    assert [item["key"] for item in requests] == [
        "image--nature--001",
        "image--nature--002",
    ]
    assert requests[0]["request"] == build_image_request(
        "Create a nature image.", json.loads(config_path.read_text())["image"]
    )
    assert requests[0]["request"]["generationConfig"] == {
        "responseModalities": ["IMAGE"],
        "thinkingConfig": {"includeThoughts": True},
    }


def test_finalize_images_writes_image_and_csv(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    with (runtime_root / "image_request_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "request_key",
                "query_id",
                "query",
                "sample_idx",
                "model",
                "generation_config",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "request_key": "image--nature--001",
                "query_id": "nature",
                "query": "Create an image of a scene in nature.",
                "sample_idx": 1,
                "model": "gemini-3-pro-image",
                "generation_config": "{}",
            }
        )
    result = {
        "key": "image--nature--001",
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "Plan the composition."},
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(b"png-data").decode(),
                                }
                            },
                        ]
                    }
                }
            ]
        },
    }
    (runtime_root / "image_batch_results.jsonl").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )
    results_root = tmp_path / "results"

    finalize_images(
        Namespace(runtime_root=str(runtime_root), results_root=str(results_root))
    )

    image_path = results_root / "data" / "images" / "nature" / "001.png"
    assert image_path.read_bytes() == b"png-data"
    with (results_root / "data" / "image_experiment.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["sample_idx"] == "1"
    assert rows[0]["image_path"] == str(image_path)
