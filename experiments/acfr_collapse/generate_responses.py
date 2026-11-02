from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from google.genai import types

from lib.generate_with_gemini import (
    download_batch,
    extract_response_image,
    extract_result_key,
    parse_batch_results_with_thoughts,
    submit_batch,
    update_batch_job,
)


EXPERIMENT_ID = "acfr_collapse"
DEFAULT_CONFIG = Path(__file__).with_name("generation_config.json")
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID
MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class GenerationSettings:
    experiment_id: str
    source: dict[str, Any]
    responses_per_prompt: int
    text: dict[str, Any]
    image: dict[str, Any]
    prompts: dict[str, tuple[dict[str, str], ...]]
    user_approved_for_api: bool


def load_settings(path: Path) -> GenerationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GenerationSettings(
        experiment_id=raw["experiment_id"],
        source=raw["source"],
        responses_per_prompt=raw["responses_per_prompt"],
        text=raw["text"],
        image=raw["image"],
        prompts={
            modality: tuple(prompts)
            for modality, prompts in raw["prompts"].items()
        },
        user_approved_for_api=raw["user_approved_for_api"],
    )


def serialize_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    return types.GenerateContentConfig(**config).model_dump(
        by_alias=True,
        exclude_none=True,
    )


def build_request(prompt: str, modality_config: dict[str, Any]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": serialize_generation_config(
            modality_config["generation_config"]
        ),
    }
    if modality_config["system_instruction"]:
        request["systemInstruction"] = {
            "parts": [{"text": modality_config["system_instruction"]}]
        }
    return request


def read_words(source: dict[str, Any]) -> list[str]:
    with Path(source["path"]).open(newline="", encoding="utf-8") as input_file:
        return [row[source["word_column"]] for row in csv.DictReader(input_file)]


def sample_words(settings: GenerationSettings) -> list[str]:
    words = read_words(settings.source)
    return random.SystemRandom().sample(
        words,
        settings.source["word_sample_count"],
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_requests(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    selected_words = sample_words(settings)
    requests = {"text": [], "image": []}
    manifest = []

    for word_idx, word in enumerate(selected_words, start=1):
        for modality in ("text", "image"):
            modality_config = getattr(settings, modality)
            config_label = json.dumps(
                modality_config["generation_config"], sort_keys=True
            )
            for prompt_template in settings.prompts[modality]:
                prompt = prompt_template["template"].format(word=word)
                template_id = prompt_template["template_id"]
                for sample_idx in range(1, settings.responses_per_prompt + 1):
                    request_key = (
                        f"{modality}--w{word_idx:03d}--{template_id}--"
                        f"{sample_idx:02d}"
                    )
                    requests[modality].append(
                        {
                            "key": request_key,
                            "request": build_request(prompt, modality_config),
                        }
                    )
                    manifest.append(
                        {
                            "request_key": request_key,
                            "word": word,
                            "prompt_template": template_id,
                            "prompt": prompt,
                            "modality": modality,
                            "sample_idx": sample_idx,
                            "model": modality_config["model"],
                            "generation_config": config_label,
                            "system_instruction": (
                                modality_config["system_instruction"] or ""
                            ),
                        }
                    )

    runtime_root = Path(args.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    for modality, records in requests.items():
        count = write_jsonl(runtime_root / f"{modality}_requests.jsonl", records)
        print(f"Wrote {count} {modality} requests under {runtime_root}")
    with (runtime_root / "generation_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return {row["request_key"]: row for row in csv.DictReader(source)}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def validate_requests(runtime_root: Path, settings: GenerationSettings) -> None:
    manifest = read_manifest(runtime_root / "generation_manifest.csv")
    template_lookup = {
        (modality, prompt["template_id"]): prompt["template"]
        for modality, prompts in settings.prompts.items()
        for prompt in prompts
    }
    for modality in ("text", "image"):
        modality_config = getattr(settings, modality)
        prepared = {
            item["key"]: item["request"]
            for item in read_jsonl(runtime_root / f"{modality}_requests.jsonl")
        }
        expected_keys = {
            request_key
            for request_key, row in manifest.items()
            if row["modality"] == modality
        }
        if set(prepared) != expected_keys:
            raise ValueError(f"Prepared {modality} requests differ from the manifest")

        for request_key in expected_keys:
            row = manifest[request_key]
            expected_prompt = template_lookup[
                (modality, row["prompt_template"])
            ].format(word=row["word"])
            expected_request = build_request(expected_prompt, modality_config)
            if row["prompt"] != expected_prompt or prepared[request_key] != expected_request:
                raise ValueError(
                    f"Prepared request {request_key} does not match the approved "
                    "configuration"
                )


def validate_prepared_requests(args: argparse.Namespace) -> None:
    validate_requests(
        Path(args.runtime_root),
        load_settings(Path(args.config)),
    )


def require_api_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_api:
        raise PermissionError("API access requires the explicit --confirm-api flag")


def submit_requests(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    config_path = Path(args.config)
    settings = load_settings(config_path)
    if not settings.user_approved_for_api:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {config_path}"
        )

    runtime_root = Path(args.runtime_root)
    validate_requests(runtime_root, settings)
    modality_config = getattr(settings, args.modality)
    submit_batch(
        runtime_root / f"{args.modality}_requests.jsonl",
        modality_config["model"],
        runtime_root / f"{args.modality}_job.json",
        f"{EXPERIMENT_ID}-{args.modality}",
        client,
    )


def show_status(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    _, record = update_batch_job(
        Path(args.runtime_root) / f"{args.modality}_job.json",
        client,
    )
    print(f"{record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    runtime_root = Path(args.runtime_root)
    download_batch(
        runtime_root / f"{args.modality}_job.json",
        runtime_root / f"{args.modality}_results.jsonl",
        client,
    )


def parse_image_results(
    path: Path,
) -> tuple[dict[str, tuple[bytes, str, list[str]]], list[dict[str, str]]]:
    successes = {}
    failures = []
    for line_number, item in enumerate(read_jsonl(path), start=1):
        request_key = extract_result_key(item)
        image, mime_type, thoughts, error = extract_response_image(item)
        if not request_key:
            failures.append(
                {
                    "request_key": "",
                    "error": f"Line {line_number}: missing key",
                }
            )
        elif error or not image or mime_type not in MIME_EXTENSIONS:
            failures.append(
                {
                    "request_key": request_key,
                    "error": error or f"Unsupported image MIME type: {mime_type}",
                }
            )
        elif request_key in successes:
            failures.append(
                {"request_key": request_key, "error": "Duplicate result key"}
            )
        else:
            successes[request_key] = (image, mime_type, thoughts)
    return successes, failures


def add_result_coverage_failures(
    manifest: dict[str, dict[str, str]],
    modality: str,
    successes: dict[str, Any],
    failures: list[dict[str, str]],
) -> None:
    expected_keys = {
        key for key, row in manifest.items() if row["modality"] == modality
    }
    failures.extend(
        {"request_key": key, "error": "Unknown result key"}
        for key in sorted(set(successes) - expected_keys)
    )
    failed_keys = {failure["request_key"] for failure in failures}
    failures.extend(
        {"request_key": key, "error": "Missing result"}
        for key in sorted(expected_keys - set(successes) - failed_keys)
    )


def get_image_reference(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def finalize_results(args: argparse.Namespace) -> None:
    runtime_root = Path(args.runtime_root)
    results_root = Path(args.results_root)
    manifest = read_manifest(runtime_root / "generation_manifest.csv")
    text_responses, text_failures = parse_batch_results_with_thoughts(
        runtime_root / "text_results.jsonl"
    )
    image_responses, image_failures = parse_image_results(
        runtime_root / "image_results.jsonl"
    )
    add_result_coverage_failures(
        manifest, "text", text_responses, text_failures
    )
    add_result_coverage_failures(
        manifest, "image", image_responses, image_failures
    )

    failures = [
        {"modality": "text", **failure} for failure in text_failures
    ] + [{"modality": "image", **failure} for failure in image_failures]
    data_root = results_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    failed_keys = {failure["request_key"] for failure in failures}
    if failures:
        failure_path = data_root / "failed_requests.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=failures[0].keys())
            writer.writeheader()
            writer.writerows(failures)
        print(f"Skipping {len(failures)} failed requests recorded in {failure_path}")

    rows = []
    for request_key, row in manifest.items():
        if request_key in failed_keys:
            continue
        if row["modality"] == "text":
            response, thoughts = text_responses[request_key]
        else:
            image, mime_type, thoughts = image_responses[request_key]
            image_path = (
                data_root
                / "images"
                / row["word"]
                / row["prompt_template"]
                / f"sample_{int(row['sample_idx']):02d}{MIME_EXTENSIONS[mime_type]}"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image)
            response = get_image_reference(image_path)
        rows.append(
            {
                "word": row["word"],
                "prompt": row["prompt"],
                "modality": row["modality"],
                "sample_idx": int(row["sample_idx"]),
                "response": response,
                "thinking": "\n".join(thoughts),
            }
        )

    rows.sort(
        key=lambda row: (
            row["word"],
            row["modality"],
            row["prompt"],
            row["sample_idx"],
        )
    )
    output_path = data_root / "generations.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} responses to {output_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--modality", choices=("text", "image"), required=True)
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur API charges",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ACF ratings collapse experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("prepare", prepare_requests),
        ("validate", validate_prepared_requests),
        ("submit", submit_requests),
        ("status", show_status),
        ("download", download_results),
        ("finalize", finalize_results),
    ):
        command = commands.add_parser(name)
        add_common_arguments(command)
        if name in {"submit", "status", "download"}:
            add_api_arguments(command)
        command.set_defaults(function=function)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
