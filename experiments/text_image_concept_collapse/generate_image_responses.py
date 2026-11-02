from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from google.genai import types

from lib.generate_with_gemini import (
    download_batch,
    extract_response_image,
    submit_batch,
    update_batch_job,
)


EXPERIMENT_ID = "text_image_concept_collapse"
DEFAULT_CONFIG = Path(__file__).with_name("generation_config.json")
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID
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


def build_image_request(prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": serialize_generation_config(config["generation_config"]),
    }
    if config["system_instruction"]:
        request["systemInstruction"] = {
            "parts": [{"text": config["system_instruction"]}]
        }
    return request


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_requests(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    image_config = config["image"]
    requests = []
    manifest = []
    for prompt in config["prompts"]:
        for sample_idx in range(1, config["responses_per_prompt"] + 1):
            key = f"image--{prompt['query_id']}--{sample_idx:03d}"
            requests.append(
                {
                    "key": key,
                    "request": build_image_request(prompt["image"], image_config),
                }
            )
            manifest.append(
                {
                    "request_key": key,
                    "query_id": prompt["query_id"],
                    "query": prompt["image"],
                    "sample_idx": sample_idx,
                    "model": image_config["model"],
                    "generation_config": json.dumps(
                        image_config["generation_config"], sort_keys=True
                    ),
                }
            )

    runtime_root = Path(args.runtime_root)
    count = write_jsonl(runtime_root / "image_response_batch.jsonl", requests)
    runtime_root.mkdir(parents=True, exist_ok=True)
    with (runtime_root / "image_request_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote {count} image requests under {runtime_root}")


def validate_requests(runtime_root: Path, config: dict[str, Any]) -> None:
    manifest_path = runtime_root / "image_request_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as source:
        manifest = {row["request_key"]: row for row in csv.DictReader(source)}

    request_path = runtime_root / "image_response_batch.jsonl"
    with request_path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            row = manifest[item["key"]]
            expected = build_image_request(row["query"], config["image"])
            if item["request"] != expected:
                raise ValueError(
                    f"Prepared request {item['key']} does not match {DEFAULT_CONFIG}"
                )


def require_api_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_api:
        raise PermissionError("API access requires the explicit --confirm-api flag")


def submit_requests(args: argparse.Namespace) -> None:
    require_api_confirmation(args)
    config_path = Path(args.config)
    config = load_config(config_path)
    if not config["user_approved_for_api"]:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {config_path}"
        )
    runtime_root = Path(args.runtime_root)
    validate_requests(runtime_root, config)
    submit_batch(
        runtime_root / "image_response_batch.jsonl",
        config["image"]["model"],
        runtime_root / "image_job.json",
        "text-image-concept-collapse-images",
    )


def check_status(args: argparse.Namespace) -> None:
    require_api_confirmation(args)
    _, record = update_batch_job(Path(args.runtime_root) / "image_job.json")
    print(f"{record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace) -> None:
    require_api_confirmation(args)
    runtime_root = Path(args.runtime_root)
    download_batch(
        runtime_root / "image_job.json",
        runtime_root / "image_batch_results.jsonl",
    )


def iter_result_items(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def finalize_images(args: argparse.Namespace) -> None:
    runtime_root = Path(args.runtime_root)
    results_root = Path(args.results_root)
    with (runtime_root / "image_request_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        manifest = {row["request_key"]: row for row in csv.DictReader(source)}

    rows = []
    failures = []
    for item in iter_result_items(runtime_root / "image_batch_results.jsonl"):
        key = str(item.get("key") or (item.get("metadata") or {}).get("key") or "")
        image, mime_type, thoughts, error = extract_response_image(item)
        if key not in manifest:
            failures.append({"request_key": key, "error": "Unknown request key"})
            continue
        if error or not image or mime_type not in MIME_EXTENSIONS:
            failures.append(
                {
                    "request_key": key,
                    "error": error or f"Unsupported image MIME type: {mime_type}",
                }
            )
            continue

        row = manifest[key]
        image_path = (
            results_root
            / "data"
            / "images"
            / row["query_id"]
            / f"{int(row['sample_idx']):03d}{MIME_EXTENSIONS[mime_type]}"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image)
        rows.append(
            {
                **row,
                "image_path": str(image_path),
                "mime_type": mime_type,
                "thoughts": json.dumps(thoughts, ensure_ascii=False),
            }
        )

    if failures:
        failure_path = results_root / "data" / "failed_image_requests.csv"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=failures[0].keys())
            writer.writeheader()
            writer.writerows(failures)
        raise RuntimeError(f"Image generation had failures; see {failure_path}")

    rows.sort(key=lambda row: (row["query_id"], int(row["sample_idx"])))
    csv_path = results_root / "data" / "image_experiment.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} image records to {csv_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))


def add_api_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur API charges",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cross-modal experiment images")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("prepare", prepare_requests),
        ("submit", submit_requests),
        ("status", check_status),
        ("download", download_results),
        ("finalize", finalize_images),
    ):
        command = commands.add_parser(name)
        add_common_arguments(command)
        if name in {"submit", "status", "download"}:
            add_api_confirmation(command)
        command.set_defaults(function=function)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
