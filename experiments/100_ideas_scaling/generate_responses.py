from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lib.compute_identifiers import create_stable_id
from lib.generate_with_gemini import (
    build_generation_request,
    download_batch,
    parse_batch_results,
    submit_batch,
    update_batch_job,
)


EXPERIMENT_ID = "100_ideas_scaling"
DEFAULT_CONFIG = Path(__file__).with_name("generation_config.json")
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID


@dataclass(frozen=True)
class GenerationSettings:
    experiment_id: str
    model: str
    responses_per_prompt: int
    system_instruction: str | None
    generation_config: dict[str, Any]
    diversity_instruction: str
    prompts: tuple[dict[str, str], ...]
    user_approved_for_api: bool


def load_settings(path: Path) -> GenerationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GenerationSettings(
        experiment_id=raw["experiment_id"],
        model=raw["model"],
        responses_per_prompt=raw["responses_per_prompt"],
        system_instruction=raw["system_instruction"],
        generation_config=raw["generation_config"],
        diversity_instruction=raw["diversity_instruction"],
        prompts=tuple(raw["prompts"]),
        user_approved_for_api=raw["user_approved_for_api"],
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
    requests = []
    manifest = []
    config_label = json.dumps(settings.generation_config, sort_keys=True)
    for prompt in settings.prompts:
        query = f"{prompt['base_prompt']}\n\n{settings.diversity_instruction}"
        for sample_idx in range(1, settings.responses_per_prompt + 1):
            request_key = f"generation--{prompt['prompt_id']}--{sample_idx:03d}"
            requests.append(
                {
                    "key": request_key,
                    "request": build_generation_request(
                        query,
                        settings.generation_config,
                        settings.system_instruction,
                    ),
                }
            )
            manifest.append(
                {
                    "request_key": request_key,
                    "prompt_id": prompt["prompt_id"],
                    "base_prompt": prompt["base_prompt"],
                    "diversity_instruction": settings.diversity_instruction,
                    "query": query,
                    "sample_idx": sample_idx,
                    "model": settings.model,
                    "generation_config": config_label,
                    "system_instruction": settings.system_instruction,
                }
            )

    runtime_root = Path(args.runtime_root)
    count = write_jsonl(runtime_root / "generation_requests.jsonl", requests)
    pd.DataFrame(manifest).to_csv(
        runtime_root / "generation_manifest.csv", index=False
    )
    print(f"Wrote {count} requests under {runtime_root}")


def validate_requests(runtime_root: Path, settings: GenerationSettings) -> None:
    manifest = pd.read_csv(
        runtime_root / "generation_manifest.csv", keep_default_na=False
    ).set_index("request_key")
    prepared = {}
    with (runtime_root / "generation_requests.jsonl").open(
        encoding="utf-8"
    ) as source:
        for line in source:
            item = json.loads(line)
            prepared[item["key"]] = item["request"]

    for request_key, row in manifest.iterrows():
        expected = build_generation_request(
            row["query"], settings.generation_config, settings.system_instruction
        )
        if prepared.get(request_key) != expected:
            raise ValueError(
                f"Prepared request {request_key} does not match the approved "
                "configuration"
            )
    if set(prepared) != set(manifest.index):
        raise ValueError("Prepared requests differ from the request manifest")


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
    submit_batch(
        runtime_root / "generation_requests.jsonl",
        settings.model,
        runtime_root / "generation_job.json",
        EXPERIMENT_ID,
        client,
    )


def show_status(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    _, record = update_batch_job(
        Path(args.runtime_root) / "generation_job.json", client
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


def finalize_results(args: argparse.Namespace) -> None:
    settings = load_settings(Path(args.config))
    runtime_root = Path(args.runtime_root)
    results_root = Path(args.results_root)
    manifest = pd.read_csv(
        runtime_root / "generation_manifest.csv", keep_default_na=False
    ).set_index("request_key", drop=False)
    responses, failures = parse_batch_results(
        runtime_root / "generation_results.jsonl"
    )
    failures.extend(
        {"request_key": key, "error": "Unknown result key"}
        for key in sorted(set(responses) - set(manifest.index))
    )
    failed_keys = {failure["request_key"] for failure in failures}
    failures.extend(
        {"request_key": key, "error": "Missing result"}
        for key in sorted(set(manifest.index) - set(responses) - failed_keys)
    )

    data_root = results_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    if failures:
        pd.DataFrame(failures).to_csv(data_root / "failed_requests.csv", index=False)
        raise RuntimeError(
            f"Generation batch had {len(failures)} failures; "
            f"see {data_root / 'failed_requests.csv'}"
        )

    rows = []
    for request_key, row in manifest.iterrows():
        rows.append(
            {
                "generation_id": create_stable_id(
                    "gen", settings.experiment_id, request_key
                ),
                "prompt_id": row["prompt_id"],
                "base_prompt": row["base_prompt"],
                "diversity_instruction": row["diversity_instruction"],
                "query": row["query"],
                "sample_idx": int(row["sample_idx"]),
                "response": responses[request_key],
                "provider": "google",
                "model": row["model"],
                "generation_config": row["generation_config"],
                "system_instruction": row["system_instruction"] or None,
            }
        )
    output = pd.DataFrame(rows).sort_values(["prompt_id", "sample_idx"])
    output.to_csv(data_root / "generations.csv", index=False)
    print(f"Wrote {len(output)} responses to {data_root / 'generations.csv'}")


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
    parser = argparse.ArgumentParser(description="Run the 100-ideas scaling experiment")
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
