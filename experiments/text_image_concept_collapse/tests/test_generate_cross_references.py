import base64
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from experiments.text_image_concept_collapse.generate_cross_references import (
    build_caption_request,
    build_response_image_request,
    build_retry_response_image_request,
    finalize_captions,
    finalize_response_image_retries,
    finalize_response_images,
    prepare_caption_requests,
    prepare_response_image_requests,
    prepare_response_image_retry_requests,
)


CAPTION_SETTINGS = {
    "model": "gemini-3.8-flash",
    "system_instruction": None,
    "prompt": "Describe only what is visible.",
    "generation_config": {
        "response_modalities": ["TEXT"],
        "thinking_config": {
            "include_thoughts": True,
            "thinking_level": "medium",
        },
    },
}
IMAGE_SETTINGS = {
    "model": "gemini-3-pro-image",
    "system_instruction": None,
    "prompt_mode": "response_verbatim",
    "generation_config": {
        "response_modalities": ["IMAGE"],
        "thinking_config": {"include_thoughts": True},
    },
}
RETRY_SETTINGS = {
    **IMAGE_SETTINGS,
    "prompt_mode": "explicit_instruction_retry",
    "instruction": "Create an image depicting the following description:",
}


def test_prepare_caption_request_embeds_source_image(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"image")
    source_path = tmp_path / "image_experiment.csv"
    pd.DataFrame(
        [
            {
                "query_id": "nature",
                "sample_idx": 1,
                "image_path": str(image_path),
                "mime_type": "image/jpeg",
            }
        ]
    ).to_csv(source_path, index=False)

    requests, manifest = prepare_caption_requests(CAPTION_SETTINGS, source_path)

    assert requests[0]["key"] == "caption--nature--001"
    assert requests[0]["request"] == build_caption_request(
        image_path, "image/jpeg", CAPTION_SETTINGS
    )
    assert requests[0]["request"]["contents"][0]["parts"][0]["inlineData"][
        "data"
    ] == base64.b64encode(b"image").decode("ascii")
    assert manifest[0]["image_sample_idx"] == 1


def test_prepare_response_image_request_uses_response_verbatim(tmp_path: Path) -> None:
    source_path = tmp_path / "text_experiment.csv"
    pd.DataFrame(
        [
            {
                "query": "Write a paragraph on a scene in nature.",
                "sample_idx": 1,
                "response": "Exact response text.",
                "model": "gemini-3.8-flash",
            }
        ]
    ).to_csv(source_path, index=False)

    requests, manifest = prepare_response_image_requests(IMAGE_SETTINGS, source_path)

    assert requests[0]["key"] == "response-image--nature--001"
    assert requests[0]["request"] == build_response_image_request(
        "Exact response text.", IMAGE_SETTINGS
    )
    assert manifest[0]["source_response"] == "Exact response text."


def test_prepare_response_image_retry_uses_only_failed_requests(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "request_manifest.csv"
    pd.DataFrame(
        [
            {
                "request_key": "response-image--nature--001",
                "query_id": "nature",
                "response_sample_idx": 1,
                "source_response": "First response.",
                "source_text_model": "text-model",
            },
            {
                "request_key": "response-image--nature--002",
                "query_id": "nature",
                "response_sample_idx": 2,
                "source_response": "Second response.",
                "source_text_model": "text-model",
            },
        ]
    ).to_csv(manifest_path, index=False)
    failure_path = tmp_path / "failed_requests.csv"
    pd.DataFrame(
        [{"request_key": "response-image--nature--002", "error": "No image"}]
    ).to_csv(failure_path, index=False)

    requests, manifest = prepare_response_image_retry_requests(
        RETRY_SETTINGS, manifest_path, failure_path
    )

    assert [request["key"] for request in requests] == [
        "retry--response-image--nature--002"
    ]
    assert requests[0]["request"] == build_retry_response_image_request(
        "Second response.", RETRY_SETTINGS
    )
    assert manifest[0]["original_request_key"] == "response-image--nature--002"


def test_finalize_captions_stores_text_and_thoughts(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [{"request_key": "caption--nature--001", "query_id": "nature"}]
    ).set_index("request_key", drop=False)
    items = {
        "caption--nature--001": {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Inspect the image.", "thought": True},
                                {"text": "A forest scene."},
                            ]
                        }
                    }
                ]
            }
        }
    }

    finalize_captions(manifest, items, tmp_path)

    output = pd.read_csv(tmp_path / "image_captions.csv")
    assert output.loc[0, "caption"] == "A forest scene."
    assert json.loads(output.loc[0, "thoughts"]) == ["Inspect the image."]


def test_finalize_response_images_stores_generated_file(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "request_key": "response-image--nature--001",
                "query_id": "nature",
                "response_sample_idx": 1,
            }
        ]
    ).set_index("request_key", drop=False)
    items = {
        "response-image--nature--001": {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": base64.b64encode(b"generated").decode(),
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
    }

    finalize_response_images(manifest, items, tmp_path)

    output = pd.read_csv(tmp_path / "generated_images.csv")
    image_path = Path(output.loc[0, "image_path"])
    assert image_path.read_bytes() == b"generated"
    assert output.loc[0, "response_sample_idx"] == 1


def test_finalize_retries_merges_original_and_retry_images(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    original_runtime = runtime_root / "text-to-image-embed"
    original_runtime.mkdir(parents=True)
    original_manifest = pd.DataFrame(
        [
            {
                "request_key": "response-image--nature--001",
                "query_id": "nature",
                "response_sample_idx": 1,
            },
            {
                "request_key": "response-image--nature--002",
                "query_id": "nature",
                "response_sample_idx": 2,
            },
        ]
    )
    original_manifest.to_csv(original_runtime / "request_manifest.csv", index=False)
    original_items = [
        create_image_result("response-image--nature--001", b"original"),
        {
            "key": "response-image--nature--002",
            "response": {
                "candidates": [{"content": {}, "finishReason": "NO_IMAGE"}]
            },
        },
    ]
    (original_runtime / "results.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in original_items), encoding="utf-8"
    )
    retry_manifest = pd.DataFrame(
        [
            {
                "request_key": "retry--response-image--nature--002",
                "original_request_key": "response-image--nature--002",
                "query_id": "nature",
                "response_sample_idx": 2,
            }
        ]
    ).set_index("request_key", drop=False)
    retry_items = {
        "retry--response-image--nature--002": create_image_result(
            "retry--response-image--nature--002", b"retry"
        )
    }
    data_dir = tmp_path / "data"

    finalize_response_image_retries(
        Namespace(runtime_root=str(runtime_root)),
        retry_manifest,
        retry_items,
        data_dir,
    )

    output = pd.read_csv(data_dir / "generated_images.csv")
    assert output.generation_attempt.tolist() == ["original", "retry"]
    assert Path(output.loc[0, "image_path"]).read_bytes() == b"original"
    assert Path(output.loc[1, "image_path"]).read_bytes() == b"retry"


def create_image_result(key: str, image: bytes) -> dict:
    return {
        "key": key,
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": base64.b64encode(image).decode(),
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    }
