from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from google.genai import types
from tqdm import tqdm

from lib.compute_identifiers import compute_file_sha256, create_stable_id
from lib.generate_with_gemini import (
    download_batch,
    extract_response_image,
    extract_response_text,
    submit_batch,
    update_batch_job,
)


EXPERIMENT_ID = "text_image_concept_collapse"
EXPERIMENT_DIR = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "cross_reference_generation_config.json"
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID
TARGET_SETTINGS = {
    "captions": {
        "config_key": "captioning",
        "runtime_dir": "image-caption-embed",
        "display_name": "text-image-collapse-captions",
    },
    "response-images": {
        "config_key": "response_to_image",
        "runtime_dir": "text-to-image-embed",
        "display_name": "text-image-collapse-response-images",
    },
    "response-images-retry": {
        "config_key": "response_to_image_retry",
        "runtime_dir": "text-to-image-embed-retry",
        "display_name": "text-image-collapse-response-images-retry",
    },
}
MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    return types.GenerateContentConfig(**config).model_dump(
        by_alias=True,
        exclude_none=True,
    )


def build_caption_request(
    image_path: Path,
    mime_type: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    request = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": mime_type, "data": encoded}},
                    {"text": settings["prompt"]},
                ],
            }
        ],
        "generationConfig": serialize_generation_config(
            settings["generation_config"]
        ),
    }
    if settings["system_instruction"]:
        request["systemInstruction"] = {
            "parts": [{"text": settings["system_instruction"]}]
        }
    return request


def build_response_image_request(
    response: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "contents": [{"role": "user", "parts": [{"text": response}]}],
        "generationConfig": serialize_generation_config(
            settings["generation_config"]
        ),
    }
    if settings["system_instruction"]:
        request["systemInstruction"] = {
            "parts": [{"text": settings["system_instruction"]}]
        }
    return request


def build_retry_response_image_request(
    response: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    prompt = f"{settings['instruction']}\n\n{response}"
    return build_response_image_request(prompt, settings)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def get_target_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    target = TARGET_SETTINGS[args.target]
    runtime_dir = Path(args.runtime_root) / target["runtime_dir"]
    return (
        runtime_dir / "requests.jsonl",
        runtime_dir / "request_manifest.csv",
        runtime_dir / "job.json",
        runtime_dir / "results.jsonl",
    )


def prepare_caption_requests(
    settings: dict[str, Any],
    image_data_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = []
    manifest = []
    images = pd.read_csv(image_data_path)
    config_label = json.dumps(settings["generation_config"], sort_keys=True)
    for row in tqdm(
        images.itertuples(index=False),
        total=len(images),
        desc="Preparing caption requests",
    ):
        image_path = Path(row.image_path)
        request_key = f"caption--{row.query_id}--{int(row.sample_idx):03d}"
        requests.append(
            {
                "key": request_key,
                "request": build_caption_request(
                    image_path, row.mime_type, settings
                ),
            }
        )
        manifest.append(
            {
                "request_key": request_key,
                "query_id": row.query_id,
                "image_sample_idx": int(row.sample_idx),
                "source_image_path": str(image_path),
                "source_mime_type": row.mime_type,
                "source_image_sha256": compute_file_sha256(image_path),
                "caption_prompt": settings["prompt"],
                "model": settings["model"],
                "generation_config": config_label,
                "system_instruction": settings["system_instruction"],
            }
        )
    return requests, manifest


def prepare_response_image_requests(
    settings: dict[str, Any],
    text_data_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = []
    manifest = []
    texts = pd.read_csv(text_data_path)
    config_label = json.dumps(settings["generation_config"], sort_keys=True)
    query_ids = load_query_ids()
    for row in tqdm(
        texts.itertuples(index=False),
        total=len(texts),
        desc="Preparing response-image requests",
    ):
        query_id = query_ids[row.query.rstrip(".")]
        request_key = f"response-image--{query_id}--{int(row.sample_idx):03d}"
        requests.append(
            {
                "key": request_key,
                "request": build_response_image_request(row.response, settings),
            }
        )
        manifest.append(
            {
                "request_key": request_key,
                "query_id": query_id,
                "response_sample_idx": int(row.sample_idx),
                "source_response": row.response,
                "source_text_model": row.model,
                "prompt_mode": settings["prompt_mode"],
                "model": settings["model"],
                "generation_config": config_label,
                "system_instruction": settings["system_instruction"],
            }
        )
    return requests, manifest


def prepare_response_image_retry_requests(
    settings: dict[str, Any],
    original_manifest_path: Path,
    failure_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original = pd.read_csv(original_manifest_path, keep_default_na=False).set_index(
        "request_key"
    )
    failed_keys = pd.read_csv(failure_path, keep_default_na=False)[
        "request_key"
    ]
    config_label = json.dumps(settings["generation_config"], sort_keys=True)
    requests = []
    manifest = []
    for original_key in tqdm(failed_keys, desc="Preparing response-image retries"):
        row = original.loc[original_key]
        request_key = f"retry--{original_key}"
        requests.append(
            {
                "key": request_key,
                "request": build_retry_response_image_request(
                    row["source_response"], settings
                ),
            }
        )
        manifest.append(
            {
                "request_key": request_key,
                "original_request_key": original_key,
                "query_id": row["query_id"],
                "response_sample_idx": int(row["response_sample_idx"]),
                "source_response": row["source_response"],
                "source_text_model": row["source_text_model"],
                "prompt_mode": settings["prompt_mode"],
                "retry_instruction": settings["instruction"],
                "model": settings["model"],
                "generation_config": config_label,
                "system_instruction": settings["system_instruction"],
            }
        )
    return requests, manifest


def load_query_ids() -> dict[str, str]:
    generation_path = EXPERIMENT_DIR / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    return {
        prompt["text"].rstrip("."): prompt["query_id"]
        for prompt in generation["prompts"]
    }


def prepare_requests(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    target = TARGET_SETTINGS[args.target]
    settings = config[target["config_key"]]
    if args.target == "captions":
        requests, manifest = prepare_caption_requests(
            settings,
            Path(args.results_root) / "data/image_experiment.csv",
        )
    elif args.target == "response-images":
        requests, manifest = prepare_response_image_requests(
            settings,
            Path(args.results_root) / "data/text_experiment.csv",
        )
    else:
        requests, manifest = prepare_response_image_retry_requests(
            settings,
            Path(args.runtime_root)
            / TARGET_SETTINGS["response-images"]["runtime_dir"]
            / "request_manifest.csv",
            Path(args.results_root)
            / "data/text-to-image-embed/failed_requests.csv",
        )
    request_path, manifest_path, _, _ = get_target_paths(args)
    count = write_jsonl(request_path, requests)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"Wrote {count} requests to {request_path}")


def validate_requests(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    target = TARGET_SETTINGS[args.target]
    settings = config[target["config_key"]]
    request_path, manifest_path, _, _ = get_target_paths(args)
    manifest = pd.read_csv(manifest_path, keep_default_na=False).set_index(
        "request_key"
    )
    seen = set()
    with request_path.open(encoding="utf-8") as source:
        for line in tqdm(source, total=len(manifest), desc="Validating requests"):
            item = json.loads(line)
            key = item["key"]
            row = manifest.loc[key]
            if args.target == "captions":
                image_path = Path(row["source_image_path"])
                if compute_file_sha256(image_path) != row["source_image_sha256"]:
                    raise ValueError(f"Source image changed for {key}")
                expected = build_caption_request(
                    image_path,
                    row["source_mime_type"],
                    settings,
                )
            elif args.target == "response-images":
                expected = build_response_image_request(row["source_response"], settings)
            else:
                expected = build_retry_response_image_request(
                    row["source_response"], settings
                )
            if item["request"] != expected:
                raise ValueError(f"Prepared request {key} differs from approved settings")
            seen.add(key)
    if seen != set(manifest.index):
        raise ValueError("Prepared requests differ from the request manifest")


def require_api_access(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not args.confirm_api:
        raise PermissionError("API access requires the explicit --confirm-api flag")
    if not config["user_approved_for_api"]:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {args.config}"
        )


def submit_requests(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    require_api_access(args, config)
    validate_requests(args)
    target = TARGET_SETTINGS[args.target]
    request_path, _, job_path, _ = get_target_paths(args)
    submit_batch(
        request_path,
        config[target["config_key"]]["model"],
        job_path,
        target["display_name"],
    )


def show_status(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    require_api_access(args, config)
    _, _, job_path, _ = get_target_paths(args)
    _, record = update_batch_job(job_path)
    print(f"{record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    require_api_access(args, config)
    _, _, job_path, result_path = get_target_paths(args)
    download_batch(job_path, result_path)


def extract_thoughts(item: dict[str, Any]) -> list[str]:
    response = item.get("response") or item
    candidates = response.get("candidates") or []
    if not candidates:
        return []
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return [
        str(part["text"])
        for part in parts
        if part.get("thought") and part.get("text")
    ]


def read_result_items(path: Path) -> dict[str, dict[str, Any]]:
    items = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item.get("key") or (item.get("metadata") or {}).get("key") or "")
            if not key or key in items:
                raise ValueError("Result contains a missing or duplicate request key")
            items[key] = item
    return items


def finalize_captions(
    manifest: pd.DataFrame,
    items: dict[str, dict[str, Any]],
    data_dir: Path,
) -> None:
    rows = []
    failures = []
    for key, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Finalizing captions"
    ):
        text, error = extract_response_text(items.get(key, {}))
        if error:
            failures.append({"request_key": key, "error": error})
            continue
        rows.append(
            {
                "caption_id": create_stable_id("caption", EXPERIMENT_ID, key),
                **row.to_dict(),
                "caption": text,
                "thoughts": json.dumps(extract_thoughts(items[key]), ensure_ascii=False),
                "provider": "google",
            }
        )
    write_materialized_rows(rows, failures, data_dir, "image_captions.csv")


def finalize_response_images(
    manifest: pd.DataFrame,
    items: dict[str, dict[str, Any]],
    data_dir: Path,
) -> None:
    rows = []
    failures = []
    for key, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Finalizing response images"
    ):
        materialize_response_image(
            key,
            row,
            items.get(key, {}),
            data_dir,
            "original",
            rows,
            failures,
        )
    write_materialized_rows(rows, failures, data_dir, "generated_images.csv")


def finalize_response_image_retries(
    args: argparse.Namespace,
    retry_manifest: pd.DataFrame,
    retry_items: dict[str, dict[str, Any]],
    data_dir: Path,
) -> None:
    original_runtime = (
        Path(args.runtime_root) / TARGET_SETTINGS["response-images"]["runtime_dir"]
    )
    original_manifest = pd.read_csv(
        original_runtime / "request_manifest.csv", keep_default_na=False
    ).set_index("request_key", drop=False)
    original_items = read_result_items(original_runtime / "results.jsonl")
    retries = retry_manifest.set_index("original_request_key", drop=False)
    rows = []
    failures = []
    for key, row in tqdm(
        original_manifest.iterrows(),
        total=len(original_manifest),
        desc="Merging response images",
    ):
        original_image, original_mime, _, original_error = extract_response_image(
            original_items.get(key, {})
        )
        if (
            not original_error
            and original_image
            and original_mime in MIME_EXTENSIONS
        ):
            materialize_response_image(
                key, row, original_items[key], data_dir, "original", rows, failures
            )
            continue
        if key not in retries.index:
            failures.append({"request_key": key, "error": "Missing retry request"})
            continue
        retry = retries.loc[key]
        retry_key = retry["request_key"]
        materialize_response_image(
            retry_key,
            retry,
            retry_items.get(retry_key, {}),
            data_dir,
            "retry",
            rows,
            failures,
        )
    write_materialized_rows(
        rows,
        failures,
        data_dir,
        "generated_images.csv",
        "retry_failed_requests.csv",
    )


def materialize_response_image(
    key: str,
    row: pd.Series,
    item: dict[str, Any],
    data_dir: Path,
    attempt: str,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    image, mime_type, thoughts, error = extract_response_image(item)
    if error or not image or mime_type not in MIME_EXTENSIONS:
        failures.append(
            {
                "request_key": key,
                "error": error or f"Unsupported image MIME type: {mime_type}",
            }
        )
        return
    image_path = (
        data_dir
        / "generated_images"
        / row["query_id"]
        / f"{int(row['response_sample_idx']):03d}{MIME_EXTENSIONS[mime_type]}"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image)
    rows.append(
        {
            "generation_id": create_stable_id("image", EXPERIMENT_ID, key),
            **row.to_dict(),
            "generation_attempt": attempt,
            "image_path": str(image_path),
            "mime_type": mime_type,
            "thoughts": json.dumps(thoughts, ensure_ascii=False),
            "provider": "google",
        }
    )


def write_materialized_rows(
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    data_dir: Path,
    filename: str,
    failure_filename: str = "failed_requests.csv",
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        pd.DataFrame(failures).to_csv(data_dir / failure_filename, index=False)
        raise RuntimeError(f"Generation had {len(failures)} failures in {data_dir}")
    pd.DataFrame(rows).to_csv(data_dir / filename, index=False)
    print(f"Wrote {len(rows)} rows to {data_dir / filename}")


def finalize_results(args: argparse.Namespace) -> None:
    _, manifest_path, _, result_path = get_target_paths(args)
    manifest = pd.read_csv(manifest_path, keep_default_na=False).set_index(
        "request_key", drop=False
    )
    items = read_result_items(result_path)
    unknown = set(items) - set(manifest.index)
    if unknown:
        raise ValueError(f"Results contain {len(unknown)} unknown request keys")
    result_dir = (
        TARGET_SETTINGS["response-images"]["runtime_dir"]
        if args.target == "response-images-retry"
        else TARGET_SETTINGS[args.target]["runtime_dir"]
    )
    data_dir = Path(args.results_root) / "data" / result_dir
    if args.target == "captions":
        finalize_captions(manifest, items, data_dir)
    elif args.target == "response-images-retry":
        finalize_response_image_retries(args, manifest, items, data_dir)
    else:
        finalize_response_images(manifest, items, data_dir)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-reference analysis data")
    targets = parser.add_subparsers(dest="target", required=True)
    for target_name in TARGET_SETTINGS:
        target_parser = targets.add_parser(target_name)
        commands = target_parser.add_subparsers(dest="command", required=True)
        for command_name, function in (
            ("prepare", prepare_requests),
            ("validate", validate_requests),
            ("submit", submit_requests),
            ("status", show_status),
            ("download", download_results),
            ("finalize", finalize_results),
        ):
            command = commands.add_parser(command_name)
            add_common_arguments(command)
            if command_name in {"submit", "status", "download"}:
                command.add_argument("--confirm-api", action="store_true")
            command.set_defaults(function=function)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
