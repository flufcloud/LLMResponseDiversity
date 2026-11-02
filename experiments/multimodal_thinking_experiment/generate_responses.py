from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lib.compute_identifiers import create_stable_id
from lib.generate_with_gemini import (
    build_batch_generation_request,
    download_batch,
    parse_batch_text_generation_results,
    submit_batch,
    update_batch_job,
)


EXPERIMENT_ID = "multimodal_thinking_experiment"
DEFAULT_CONFIG = Path(__file__).with_name("generation_config.json")
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID


@dataclass(frozen=True)
class GenerationCondition:
    condition_id: str
    generation_config: dict[str, Any]


@dataclass(frozen=True)
class GenerationSettings:
    experiment_id: str
    method_id: str
    output_filename: str
    topics_path: Path
    responses_per_topic: int
    prompt_template: str
    model: str
    system_instruction: str | None
    conditions: tuple[GenerationCondition, ...]
    user_approved_for_api: bool


def load_settings(path: Path) -> GenerationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = raw["text"]
    return GenerationSettings(
        experiment_id=raw["experiment_id"],
        method_id=raw["method_id"],
        output_filename=raw["output_filename"],
        topics_path=Path(raw["topics_path"]),
        responses_per_topic=raw["responses_per_topic"],
        prompt_template=raw["prompt_template"],
        model=text["model"],
        system_instruction=text["system_instruction"],
        conditions=tuple(
            GenerationCondition(
                condition_id=condition["condition_id"],
                generation_config=condition["generation_config"],
            )
            for condition in text["conditions"]
        ),
        user_approved_for_api=raw["user_approved_for_api"],
    )


def get_runtime_root(
    args: argparse.Namespace,
    settings: GenerationSettings,
) -> Path:
    if args.runtime_root:
        return Path(args.runtime_root)
    return DEFAULT_RUNTIME_ROOT / settings.method_id


def add_topic_path(
    topic_paths: dict[str, list[list[str]]],
    topic: str,
    path: list[str],
) -> None:
    topic_paths.setdefault(topic, []).append(path)


def load_topic_paths(path: Path) -> dict[str, list[list[str]]]:
    hierarchy = json.loads(path.read_text(encoding="utf-8"))
    topic_paths: dict[str, list[list[str]]] = {}

    for group, topics in hierarchy["SUBJECTS"].items():
        add_topic_path(topic_paths, group, ["SUBJECTS", group])
        for topic in topics:
            add_topic_path(topic_paths, topic, ["SUBJECTS", group, topic])

    for section, topics in hierarchy.items():
        if section == "SUBJECTS":
            continue
        for topic in topics:
            add_topic_path(topic_paths, topic, [section, topic])
    return topic_paths


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def prepare_requests(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    topic_paths = load_topic_paths(settings.topics_path)
    requests = []
    manifest = []

    for topic, paths in topic_paths.items():
        topic_id = create_stable_id("topic", settings.experiment_id, topic)
        prompt = settings.prompt_template.format(TOPIC=topic)
        for condition in settings.conditions:
            config_label = json.dumps(
                condition.generation_config,
                sort_keys=True,
            )
            request = build_batch_generation_request(
                prompt,
                condition.generation_config,
                settings.system_instruction,
            )
            for sample_idx in range(1, settings.responses_per_topic + 1):
                request_key = (
                    f"{settings.method_id}--{condition.condition_id}--"
                    f"{topic_id}--{sample_idx:02d}"
                )
                requests.append({"key": request_key, "request": request})
                manifest.append(
                    {
                        "request_key": request_key,
                        "method_id": settings.method_id,
                        "topic_id": topic_id,
                        "topic": topic,
                        "hierarchy_paths": json.dumps(paths, ensure_ascii=False),
                        "prompt": prompt,
                        "thinking_level": condition.condition_id,
                        "sample_idx": sample_idx,
                        "model": settings.model,
                        "generation_config": config_label,
                        "system_instruction": settings.system_instruction or "",
                    }
                )

    runtime_root = get_runtime_root(args, settings)
    count = write_jsonl(runtime_root / "generation_requests.jsonl", requests)
    write_manifest(runtime_root / "generation_manifest.csv", manifest)
    print(
        f"Wrote {count} requests for {len(topic_paths)} distinct topics "
        f"under {runtime_root}"
    )


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return {row["request_key"]: row for row in csv.DictReader(source)}


def read_requests(path: Path) -> dict[str, dict[str, Any]]:
    prepared = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                prepared[item["key"]] = item["request"]
    return prepared


def validate_requests(runtime_root: Path, settings: GenerationSettings) -> None:
    manifest = read_manifest(runtime_root / "generation_manifest.csv")
    prepared = read_requests(runtime_root / "generation_requests.jsonl")
    topic_paths = load_topic_paths(settings.topics_path)
    conditions = {
        condition.condition_id: condition for condition in settings.conditions
    }

    if set(prepared) != set(manifest):
        raise ValueError("Prepared requests differ from the request manifest")

    for request_key, row in manifest.items():
        condition = conditions[row["thinking_level"]]
        prompt = settings.prompt_template.format(TOPIC=row["topic"])
        expected = build_batch_generation_request(
            prompt,
            condition.generation_config,
            settings.system_instruction,
        )
        if (
            row["method_id"] != settings.method_id
            or row["prompt"] != prompt
            or json.loads(row["hierarchy_paths"]) != topic_paths[row["topic"]]
            or row["generation_config"]
            != json.dumps(condition.generation_config, sort_keys=True)
            or prepared[request_key] != expected
        ):
            raise ValueError(
                f"Prepared request {request_key} does not match the approved "
                "configuration"
            )


def validate_prepared_requests(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    runtime_root = get_runtime_root(args, settings)
    validate_requests(
        runtime_root,
        settings,
    )
    print(f"Validated requests under {runtime_root}")


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

    runtime_root = get_runtime_root(args, settings)
    validate_requests(runtime_root, settings)
    submit_batch(
        runtime_root / "generation_requests.jsonl",
        settings.model,
        runtime_root / "generation_job.json",
        f"{settings.experiment_id}-{settings.method_id}",
        client,
    )


def show_status(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    settings = load_settings(Path(args.config))
    _, record = update_batch_job(
        get_runtime_root(args, settings) / "generation_job.json",
        client,
    )
    print(f"{record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    settings = load_settings(Path(args.config))
    runtime_root = get_runtime_root(args, settings)
    download_batch(
        runtime_root / "generation_job.json",
        runtime_root / "generation_results.jsonl",
        client,
    )


def build_failure_rows(
    manifest: dict[str, dict[str, str]],
    failures: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for failure in failures:
        source = manifest.get(failure["request_key"], {})
        rows.append(
            {
                "request_key": failure["request_key"],
                "method_id": source.get("method_id", ""),
                "topic_id": source.get("topic_id", ""),
                "topic": source.get("topic", ""),
                "thinking_level": source.get("thinking_level", ""),
                "sample_idx": source.get("sample_idx", ""),
                "error": failure["error"],
            }
        )
    return rows


def get_max_token_length(settings: GenerationSettings) -> int:
    values = {
        condition.generation_config["max_output_tokens"]
        for condition in settings.conditions
    }
    if len(values) != 1:
        raise ValueError("All thinking conditions must use one max_output_tokens value")
    return values.pop()


def finalize_results(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    runtime_root = get_runtime_root(args, settings)
    results_root = Path(args.results_root)
    manifest = read_manifest(runtime_root / "generation_manifest.csv")
    responses, failures = parse_batch_text_generation_results(
        runtime_root / "generation_results.jsonl"
    )
    failures.extend(
        {"request_key": key, "error": "Unknown result key"}
        for key in sorted(set(responses) - set(manifest))
    )
    failed_keys = {failure["request_key"] for failure in failures}
    failures.extend(
        {"request_key": key, "error": "Missing result"}
        for key in sorted(set(manifest) - set(responses) - failed_keys)
    )

    data_root = results_root / "data"
    for path in (data_root, results_root / "tables", results_root / "plots"):
        path.mkdir(parents=True, exist_ok=True)

    if failures:
        write_manifest(
            data_root / f"{settings.method_id}_failed_requests.csv",
            build_failure_rows(manifest, failures),
        )
        print(f"Recorded {len(failures)} failed requests")

    rows = []
    for request_key, row in manifest.items():
        if request_key not in responses:
            continue
        result = responses[request_key]
        rows.append(
            {
                "generation_id": create_stable_id(
                    "gen", settings.experiment_id, settings.method_id, request_key
                ),
                "method_id": settings.method_id,
                "topic_id": row["topic_id"],
                "topic": row["topic"],
                "hierarchy_paths": row["hierarchy_paths"],
                "prompt": row["prompt"],
                "thinking_level": row["thinking_level"],
                "sample_idx": int(row["sample_idx"]),
                "response": result.text,
                "thinking": "\n".join(result.thoughts),
                "provider": "google",
                "model": row["model"],
                "model_version": result.model_version,
                "generation_config": row["generation_config"],
                "system_instruction": row["system_instruction"],
                "finish_reason": result.finish_reason,
                "prompt_token_count": result.prompt_token_count,
                "response_token_count": result.response_token_count,
                "thought_token_count": result.thought_token_count,
                "total_token_count": result.total_token_count,
            }
        )

    if rows:
        write_manifest(data_root / settings.output_filename, rows)
    metadata = {
        "experiment_id": settings.experiment_id,
        "embedding_model": None,
        "max_token_length": get_max_token_length(settings),
    }
    (results_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} responses to {data_root / settings.output_filename}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-root")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))


def add_api_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur API charges",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multimodal thinking text experiment"
    )
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
            add_api_confirmation(command)
        command.set_defaults(function=function)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
