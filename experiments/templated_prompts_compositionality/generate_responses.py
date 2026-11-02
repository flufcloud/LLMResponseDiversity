from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from experiments.multimodal_thinking_experiment.generate_responses import (
    load_topic_paths,
    read_manifest,
    read_requests,
    write_jsonl,
    write_manifest,
)
from lib.compute_identifiers import create_stable_id
from lib.generate_with_gemini import (
    build_batch_generation_request,
    download_batch,
    parse_batch_text_generation_results,
    submit_batch,
    update_batch_job,
)


EXPERIMENT_ID = "templated_prompts_compositionality"
DEFAULT_CONFIG = Path(__file__).with_name("generation_config.json")
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID


@dataclass(frozen=True)
class GenerationSettings:
    experiment_id: str
    topics_path: Path
    responses_per_pair: int
    prompt_template: str
    adjective_groups: dict[str, tuple[str, ...]]
    model: str
    system_instruction: str | None
    generation_config: dict[str, Any]
    user_approved_for_api: bool


def load_settings(path: Path) -> GenerationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = raw["text"]
    return GenerationSettings(
        experiment_id=raw["experiment_id"],
        topics_path=Path(raw["topics_path"]),
        responses_per_pair=raw["responses_per_pair"],
        prompt_template=raw["prompt_template"],
        adjective_groups={
            group: tuple(adjectives)
            for group, adjectives in raw["adjective_groups"].items()
        },
        model=text["model"],
        system_instruction=text["system_instruction"],
        generation_config=text["generation_config"],
        user_approved_for_api=raw["user_approved_for_api"],
    )


def get_adjectives(settings: GenerationSettings) -> list[tuple[str, str]]:
    return [
        (group, adjective)
        for group, adjectives in settings.adjective_groups.items()
        for adjective in adjectives
    ]


def prepare_requests(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    topic_paths = load_topic_paths(settings.topics_path)
    adjectives = get_adjectives(settings)
    requests = []
    manifest = []
    config_label = json.dumps(settings.generation_config, sort_keys=True)
    total = len(adjectives) * len(topic_paths) * settings.responses_per_pair
    progress = tqdm(total=total, desc="Preparing poem requests", unit="request")

    for adjective_group, adjective in adjectives:
        adjective_id = create_stable_id(
            "adjective",
            settings.experiment_id,
            adjective_group,
            adjective,
        )
        for topic, paths in topic_paths.items():
            topic_id = create_stable_id("topic", settings.experiment_id, topic)
            prompt = settings.prompt_template.format(
                ADJECTIVE=adjective,
                TOPIC=topic,
            )
            request = build_batch_generation_request(
                prompt,
                settings.generation_config,
                settings.system_instruction,
            )
            for sample_idx in range(1, settings.responses_per_pair + 1):
                request_key = (
                    f"generation--{adjective_id}--{topic_id}--{sample_idx:02d}"
                )
                requests.append({"key": request_key, "request": request})
                manifest.append(
                    {
                        "request_key": request_key,
                        "adjective_group": adjective_group,
                        "adjective": adjective,
                        "adjective_id": adjective_id,
                        "topic": topic,
                        "topic_id": topic_id,
                        "hierarchy_paths": json.dumps(paths, ensure_ascii=False),
                        "prompt": prompt,
                        "sample_idx": sample_idx,
                        "model": settings.model,
                        "generation_config": config_label,
                        "system_instruction": settings.system_instruction or "",
                    }
                )
                progress.update()
    progress.close()

    runtime_root = Path(args.runtime_root)
    count = write_jsonl(runtime_root / "generation_requests.jsonl", requests)
    write_manifest(runtime_root / "generation_manifest.csv", manifest)
    print(
        f"Wrote {count} requests for {len(adjectives)} adjectives and "
        f"{len(topic_paths)} topics under {runtime_root}"
    )


def validate_requests(runtime_root: Path, settings: GenerationSettings) -> None:
    manifest = read_manifest(runtime_root / "generation_manifest.csv")
    prepared = read_requests(runtime_root / "generation_requests.jsonl")
    topic_paths = load_topic_paths(settings.topics_path)
    adjective_groups = {
        adjective: group
        for group, adjectives in settings.adjective_groups.items()
        for adjective in adjectives
    }
    config_label = json.dumps(settings.generation_config, sort_keys=True)
    if set(prepared) != set(manifest):
        raise ValueError("Prepared requests differ from the request manifest")

    progress = tqdm(manifest.items(), desc="Validating requests", unit="request")
    for request_key, row in progress:
        prompt = settings.prompt_template.format(
            ADJECTIVE=row["adjective"],
            TOPIC=row["topic"],
        )
        expected = build_batch_generation_request(
            prompt,
            settings.generation_config,
            settings.system_instruction,
        )
        expected_adjective_id = create_stable_id(
            "adjective",
            settings.experiment_id,
            row["adjective_group"],
            row["adjective"],
        )
        expected_topic_id = create_stable_id(
            "topic",
            settings.experiment_id,
            row["topic"],
        )
        if (
            adjective_groups[row["adjective"]] != row["adjective_group"]
            or row["adjective_id"] != expected_adjective_id
            or row["topic_id"] != expected_topic_id
            or json.loads(row["hierarchy_paths"]) != topic_paths[row["topic"]]
            or row["prompt"] != prompt
            or row["model"] != settings.model
            or row["generation_config"] != config_label
            or row["system_instruction"] != (settings.system_instruction or "")
            or prepared[request_key] != expected
        ):
            raise ValueError(
                f"Prepared request {request_key} does not match the approved "
                "configuration"
            )


def validate_prepared_requests(args: argparse.Namespace) -> None:
    validate_requests(
        Path(args.runtime_root),
        load_settings(Path(args.config)),
    )
    print(f"Validated requests under {args.runtime_root}")


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
    submit_batch(
        runtime_root / "generation_requests.jsonl",
        settings.model,
        runtime_root / "generation_job.json",
        settings.experiment_id,
        client,
    )


def show_status(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    _, record = update_batch_job(
        Path(args.runtime_root) / "generation_job.json",
        client,
    )
    print(f"{record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    runtime_root = Path(args.runtime_root)
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
                "adjective_group": source.get("adjective_group", ""),
                "adjective": source.get("adjective", ""),
                "topic": source.get("topic", ""),
                "sample_idx": source.get("sample_idx", ""),
                "error": failure["error"],
            }
        )
    return rows


def finalize_results(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    runtime_root = Path(args.runtime_root)
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
            data_root / "failed_requests.csv",
            build_failure_rows(manifest, failures),
        )
        print(f"Recorded {len(failures)} failed requests")

    rows = []
    progress = tqdm(manifest.items(), desc="Materializing responses", unit="response")
    for request_key, row in progress:
        if request_key not in responses:
            continue
        result = responses[request_key]
        rows.append(
            {
                "generation_id": create_stable_id(
                    "gen",
                    settings.experiment_id,
                    request_key,
                ),
                "adjective_group": row["adjective_group"],
                "adjective": row["adjective"],
                "adjective_id": row["adjective_id"],
                "topic": row["topic"],
                "topic_id": row["topic_id"],
                "hierarchy_paths": row["hierarchy_paths"],
                "prompt": row["prompt"],
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
        write_manifest(data_root / "generations.csv", rows)

    metadata = {
        "experiment_id": settings.experiment_id,
        "embedding_model": None,
        "max_token_length": settings.generation_config["max_output_tokens"],
    }
    (results_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} responses to {data_root / 'generations.csv'}")


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
    parser = argparse.ArgumentParser(
        description="Run the templated-prompts compositionality experiment"
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
