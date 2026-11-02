from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.multimodal_thinking_experiment.generate_responses import (
    read_manifest,
    read_requests,
    write_jsonl,
    write_manifest,
)
from experiments.multimodal_thinking_experiment.prepare_poems import (
    DEFAULT_RESULTS_ROOT,
    EXPERIMENT_ID,
)
from lib.generate_with_gemini import (
    build_batch_generation_request,
    download_batch,
    parse_batch_text_generation_results,
    submit_batch,
    update_batch_job,
)


DEFAULT_CONFIG = Path(__file__).with_name("pii_judge_config.json")
DEFAULT_RUNTIME_ROOT = Path(".cache/experiments") / EXPERIMENT_ID / "pii_judge"


@dataclass(frozen=True)
class JudgeSettings:
    experiment_id: str
    stage_id: str
    prompt_template_path: Path
    poem_placeholder: str
    model: str
    system_instruction: str | None
    generation_config: dict[str, Any]
    user_approved_for_api: bool


def load_judge_settings(path: Path) -> JudgeSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    text = raw["text"]
    return JudgeSettings(
        experiment_id=raw["experiment_id"],
        stage_id=raw["stage_id"],
        prompt_template_path=Path(raw["prompt_template_path"]),
        poem_placeholder=raw["poem_placeholder"],
        model=text["model"],
        system_instruction=text["system_instruction"],
        generation_config=text["generation_config"],
        user_approved_for_api=raw["user_approved_for_api"],
    )


def get_runtime_dir(args: argparse.Namespace) -> Path:
    root = Path(args.runtime_root) if args.runtime_root else DEFAULT_RUNTIME_ROOT
    return root / args.method


def get_data_dir(args: argparse.Namespace) -> Path:
    return Path(args.results_root) / "data" / args.method


def build_judge_prompt(poem: str, settings: JudgeSettings) -> str:
    template = settings.prompt_template_path.read_text(encoding="utf-8")
    return template.replace(settings.poem_placeholder, poem)


def build_judge_request(poem: str, settings: JudgeSettings) -> dict[str, Any]:
    return build_batch_generation_request(
        build_judge_prompt(poem, settings),
        settings.generation_config,
        settings.system_instruction,
    )


def prepare_requests(args: argparse.Namespace) -> None:
    settings = load_judge_settings(Path(args.config))
    responses = pd.read_csv(get_data_dir(args) / "responses.csv", keep_default_na=False)
    requests = []
    manifest = []
    for row in responses.itertuples(index=False):
        request_key = f"pii--{row.poem_id}"
        requests.append(
            {"key": request_key, "request": build_judge_request(row.response, settings)}
        )
        manifest.append(
            {
                "request_key": request_key,
                "poem_id": row.poem_id,
                "generation_id": row.generation_id,
                "method_id": row.method_id,
                "topic_id": row.topic_id,
                "topic": row.topic,
                "thinking_level": row.thinking_level,
                "source_sample_idx": row.source_sample_idx,
                "response_idx": row.response_idx,
            }
        )
    runtime_dir = get_runtime_dir(args)
    count = write_jsonl(runtime_dir / "pii_requests.jsonl", requests)
    write_manifest(runtime_dir / "pii_manifest.csv", manifest)
    print(f"Wrote {count} PII judge requests under {runtime_dir}")


def validate_prepared_requests(
    args: argparse.Namespace,
    settings: JudgeSettings | None = None,
) -> None:
    settings = settings or load_judge_settings(Path(args.config))
    runtime_dir = get_runtime_dir(args)
    manifest = read_manifest(runtime_dir / "pii_manifest.csv")
    prepared = read_requests(runtime_dir / "pii_requests.jsonl")
    responses = pd.read_csv(get_data_dir(args) / "responses.csv", keep_default_na=False)
    poems = dict(zip(responses["poem_id"], responses["response"], strict=True))

    if set(prepared) != set(manifest):
        raise ValueError("Prepared requests differ from the PII manifest")
    for request_key, row in manifest.items():
        expected = build_judge_request(poems[row["poem_id"]], settings)
        if row["method_id"] != args.method or prepared[request_key] != expected:
            raise ValueError(
                f"Prepared request {request_key} does not match the approved configuration"
            )
    print(f"Validated {len(prepared)} PII judge requests under {runtime_dir}")


def require_api_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_api:
        raise PermissionError("API access requires the explicit --confirm-api flag")


def submit_requests(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    settings = load_judge_settings(Path(args.config))
    if not settings.user_approved_for_api:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {args.config}"
        )
    validate_prepared_requests(args, settings)
    runtime_dir = get_runtime_dir(args)
    submit_batch(
        runtime_dir / "pii_requests.jsonl",
        settings.model,
        runtime_dir / "pii_job.json",
        f"{settings.experiment_id}-{settings.stage_id}-{args.method}",
        client,
    )


def show_status(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    _, record = update_batch_job(get_runtime_dir(args) / "pii_job.json", client)
    print(f"{record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    runtime_dir = get_runtime_dir(args)
    download_batch(
        runtime_dir / "pii_job.json",
        runtime_dir / "pii_results.jsonl",
        client,
    )


def parse_judgment(text: str) -> tuple[int, list[str]]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        value = "\n".join(lines)
    result = json.loads(value)
    if not isinstance(result, dict) or set(result) != {"count", "evidence_phrases"}:
        raise ValueError("PII response must contain only count and evidence_phrases")
    count = result["count"]
    evidence = result["evidence_phrases"]
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or not isinstance(evidence, list)
        or not all(isinstance(phrase, str) for phrase in evidence)
    ):
        raise ValueError("PII response has invalid field types")
    if count != len(evidence):
        raise ValueError("PII count must equal the number of evidence phrases")
    return count, evidence


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
                "poem_id": source.get("poem_id", ""),
                "method_id": source.get("method_id", ""),
                "topic": source.get("topic", ""),
                "thinking_level": source.get("thinking_level", ""),
                "error": failure["error"],
            }
        )
    return rows


def finalize_results(args: argparse.Namespace) -> None:
    settings = load_judge_settings(Path(args.config))
    runtime_dir = get_runtime_dir(args)
    manifest = read_manifest(runtime_dir / "pii_manifest.csv")
    poem_rows = pd.read_csv(get_data_dir(args) / "responses.csv", keep_default_na=False)
    poems = dict(zip(poem_rows["poem_id"], poem_rows["response"], strict=True))
    responses, failures = parse_batch_text_generation_results(
        runtime_dir / "pii_results.jsonl"
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

    records = []
    for request_key, source in manifest.items():
        if request_key not in responses:
            continue
        result = responses[request_key]
        try:
            count, evidence = parse_judgment(result.text)
            if any(phrase not in poems[source["poem_id"]] for phrase in evidence):
                raise ValueError("PII evidence phrase was not copied from the poem")
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            failures.append({"request_key": request_key, "error": str(error)})
            continue
        records.append(
            {
                "poem_id": source["poem_id"],
                "generation_id": source["generation_id"],
                "method_id": source["method_id"],
                "topic_id": source["topic_id"],
                "topic": source["topic"],
                "thinking_level": source["thinking_level"],
                "source_sample_idx": int(source["source_sample_idx"]),
                "response_idx": int(source["response_idx"]),
                "count": count,
                "evidence_phrases": json.dumps(evidence, ensure_ascii=False),
                "judge_response": result.text,
                "provider": "google",
                "model": settings.model,
                "model_version": result.model_version,
                "finish_reason": result.finish_reason,
                "prompt_token_count": result.prompt_token_count,
                "response_token_count": result.response_token_count,
                "thought_token_count": result.thought_token_count,
                "total_token_count": result.total_token_count,
            }
        )

    data_dir = get_data_dir(args)
    data_dir.mkdir(parents=True, exist_ok=True)
    if failures:
        write_manifest(
            data_dir / "pii_failed_requests.csv",
            build_failure_rows(manifest, failures),
        )
        print(f"Recorded {len(failures)} failed PII judgments")
    if records:
        write_manifest(data_dir / "pii_judgments.csv", records)
    print(f"Wrote {len(records)} PII judgments under {data_dir}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", required=True)
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
    parser = argparse.ArgumentParser(description="Judge poem PII with Gemini Batch")
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
