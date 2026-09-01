from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset

from lib.translate_locally import (
    TRANSLATION_PROVIDER,
)
from lib.generate_with_gemini import (
    build_generation_request,
    download_batch,
    get_gemini_client,
    parse_batch_results,
    submit_batch,
    update_batch_jobs,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    GENERATION_METADATA_PATH,
    GENERATION_COLUMNS,
    GENERATION_METADATA_COLUMNS,
    GENERATIONS_PATH,
    LANGUAGES,
    load_generation_metadata,
    load_generations,
    write_csv,
)
from lib.compute_identifiers import (
    create_stable_id,
    slugify,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.resolve_paths import ExperimentPaths
from experiments.hivemind_reproduction.en_te_yo_30_query.translate_queries import (
    TARGET_LANGUAGES,
    load_translation_checkpoint,
    translate_prompts,
)


SOURCE_DATASET = "liweijiang/infinite-chats-eval"
DEFAULT_GENERATION_CONFIG = Path(
    "experiments/hivemind_reproduction/en_te_yo_30_query/generation_config.json"
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def select_eval_prompts(
    rows: Iterable[dict[str, Any]],
    count: int,
    seed: int,
    exclude_first: int = 10,
) -> pd.DataFrame:
    candidates = pd.DataFrame(rows).reset_index(names="source_row_index")
    candidates["query"] = candidates["query"].fillna("").astype(str).str.strip()
    candidates = candidates[candidates["source_row_index"] >= exclude_first]
    selected = candidates.sample(n=count, random_state=seed).reset_index(drop=True)
    selected = selected.rename(columns={"query": "query_en"})
    selected.insert(
        0,
        "prompt_id",
        [f"prompt-{index:03d}" for index in range(1, len(selected) + 1)],
    )
    selected["source_dataset"] = SOURCE_DATASET
    selected["selection_seed"] = seed
    return selected


def load_source_dataset(cache_dir: Path) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(SOURCE_DATASET, split="train", cache_dir=str(cache_dir))
    return dataset


@dataclass(frozen=True)
class GenerationSettings:
    models: tuple[str, ...]
    responses_per_prompt: int
    system_instructions: dict[str, str | None]
    generation_config: dict[str, Any]
    user_approved_for_api: bool


def load_generation_settings(path: Path) -> GenerationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GenerationSettings(
        tuple(raw["models"]),
        raw["responses_per_prompt"],
        raw["system_instructions"],
        raw["generation_config"],
        raw["user_approved_for_api"],
    )


def prepare_experiment(args: argparse.Namespace) -> None:
    paths = ExperimentPaths(Path(args.output_dir))
    settings = load_generation_settings(Path(args.config))
    source = select_eval_prompts(
        load_source_dataset(Path(args.cache_dir)),
        args.num_prompts,
        args.seed,
        args.exclude_first,
    )
    source.to_csv(paths.source_prompts, index=False)
    write_json(
        paths.experiment_manifest,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": SOURCE_DATASET,
            "selection_seed": args.seed,
            "excluded_source_rows": list(range(args.exclude_first)),
            "num_prompts": len(source),
            "languages": LANGUAGES,
            "translation_provider": TRANSLATION_PROVIDER,
            "generation_models": list(settings.models),
            "responses_per_prompt": settings.responses_per_prompt,
            "generation_config": settings.generation_config,
            "planned_translation_count": len(source) * len(TARGET_LANGUAGES),
            "planned_generation_request_count": len(source)
            * len(LANGUAGES)
            * len(settings.models)
            * settings.responses_per_prompt,
        },
    )
    print(f"Wrote {len(source)} Infinity-Chat eval prompts to {paths.source_prompts}")


def update_job_statuses(paths: ExperimentPaths) -> None:
    job_paths = sorted((paths.root / "jobs").glob("*.json"))
    if not job_paths:
        raise FileNotFoundError(f"No job records found under {paths.root / 'jobs'}")
    for job_path, record in update_batch_jobs(job_paths):
        print(f"{job_path.name}: {record['state']} ({record['batch_name']})")


def prepare_generation_requests(
    prompts: pd.DataFrame,
    model: str,
    paths: ExperimentPaths,
    settings: GenerationSettings,
) -> int:
    manifest = []
    requests = []
    config_label = json.dumps(settings.generation_config, sort_keys=True)
    for row in prompts.itertuples(index=False):
        instruction = settings.system_instructions[row.language_code]
        for sample_idx in range(1, settings.responses_per_prompt + 1):
            key = (
                f"generation--{slugify(model)}--{row.prompt_id}--"
                f"{row.language_code}--{sample_idx:03d}"
            )
            requests.append(
                {
                    "key": key,
                    "request": build_generation_request(
                        row.query, settings.generation_config, instruction
                    ),
                }
            )
            manifest.append(
                {
                    "request_key": key,
                    "prompt_id": row.prompt_id,
                    "query": row.query,
                    "query_en": row.query_en,
                    "language_code": row.language_code,
                    "language": row.language,
                    "model": model,
                    "generation_config": config_label,
                    "system_instruction": instruction,
                    "sample_idx": sample_idx,
                }
            )
    paths.generation_manifest(model).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(paths.generation_manifest(model), index=False)
    return write_jsonl(paths.generation_requests(model), requests)


def prepare_generations(args: argparse.Namespace) -> None:
    paths = ExperimentPaths(Path(args.output_dir))
    settings = load_generation_settings(Path(args.config))
    prompts = pd.read_csv(paths.multilingual_prompts)
    for model in settings.models:
        count = prepare_generation_requests(prompts, model, paths, settings)
        print(f"Wrote {count} requests for {model} to {paths.generation_requests(model)}")


def validate_generation_requests(
    paths: ExperimentPaths,
    model: str,
    settings: GenerationSettings,
    config_path: Path | None = None,
) -> None:
    config_label = config_path or "the generation configuration"
    manifest = pd.read_csv(paths.generation_manifest(model)).set_index("request_key")
    seen = set()
    with paths.generation_requests(model).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            item = json.loads(line)
            key = item.get("key")
            if key not in manifest.index:
                raise ValueError(f"Line {line_number}: unknown request key {key!r}")
            request = item.get("request", {})
            row = manifest.loc[key]
            instruction = settings.system_instructions[row.language_code]
            expected = build_generation_request(
                row.query, settings.generation_config, instruction
            )
            if request != expected:
                raise ValueError(
                    f"Prepared request {key} does not match {config_label}; "
                    "run prepare-generations again after the user approves the configuration"
                )
            seen.add(key)
    missing = set(manifest.index) - seen
    if missing:
        raise ValueError(f"Prepared request file is missing {len(missing)} manifest requests")


def submit_generations(args: argparse.Namespace) -> None:
    require_api_confirmation(args)
    paths = ExperimentPaths(Path(args.output_dir))
    config_path = Path(args.config)
    settings = load_generation_settings(config_path)
    if not settings.user_approved_for_api:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {config_path}"
        )
    for model in settings.models:
        validate_generation_requests(paths, model, settings, config_path)
        submit_batch(
            paths.generation_requests(model),
            model,
            paths.generation_job(model),
            f"hivemind-en-te-yo-30-query-{slugify(model)}",
        )


def download_generations(args: argparse.Namespace) -> None:
    require_api_confirmation(args)
    paths = ExperimentPaths(Path(args.output_dir))
    settings = load_generation_settings(Path(args.config))
    client = get_gemini_client()
    for model in settings.models:
        download_batch(paths.generation_job(model), paths.generation_results(model), client)


def finalize_responses(args: argparse.Namespace) -> None:
    paths = ExperimentPaths(Path(args.output_dir))
    settings = load_generation_settings(Path(args.config))
    response_rows = []
    failures = []
    for model in settings.models:
        manifest = pd.read_csv(paths.generation_manifest(model)).set_index(
            "request_key", drop=False
        )
        results, model_failures = parse_batch_results(paths.generation_results(model))
        failures.extend({**failure, "model": model} for failure in model_failures)
        for request_key, text in results.items():
            if request_key not in manifest.index:
                failures.append(
                    {
                        "request_key": request_key,
                        "error": "Unknown result key",
                        "model": model,
                    }
                )
                continue
            record = manifest.loc[request_key].to_dict()
            record["response"] = text
            response_rows.append(record)
        failed_keys = {failure["request_key"] for failure in model_failures}
        missing = set(manifest.index) - set(results) - failed_keys
        failures.extend(
            {"request_key": request_key, "error": "Missing result", "model": model}
            for request_key in sorted(missing)
        )
    if failures:
        pd.DataFrame(failures).to_csv(paths.failures, index=False)
        raise RuntimeError(
            f"Generation batches have {len(failures)} failures; see {paths.failures}"
        )
    metadata = pd.DataFrame(response_rows).sort_values(
        ["prompt_id", "language_code", "model", "sample_idx"]
    ).reset_index(drop=True)
    responses = metadata[["query", "response", "model", "generation_config", "sample_idx"]]
    responses.to_csv(paths.responses)
    metadata.insert(0, "response_row_index", metadata.index)
    metadata.drop(columns="response").to_csv(paths.response_metadata, index=False)
    if getattr(args, "generations", None):
        append_canonical_generations(metadata, paths, args)
    print(f"Wrote {len(responses)} responses to {paths.responses}")
    print(f"Wrote response provenance to {paths.response_metadata}")


def append_canonical_generations(
    responses: pd.DataFrame,
    paths: ExperimentPaths,
    args: argparse.Namespace,
) -> None:
    generation_path = Path(args.generations)
    metadata_path = Path(args.generations_metadata)
    existing = load_generations(generation_path) if generation_path.exists() else pd.DataFrame(
        columns=GENERATION_COLUMNS
    )
    existing_metadata = (
        load_generation_metadata(metadata_path)
        if metadata_path.exists()
        else pd.DataFrame(columns=GENERATION_METADATA_COLUMNS)
    )
    query_lookup = {
        row.base_query: (row.query_id, row.query_tag)
        for row in existing[["query_id", "query_tag", "base_query"]]
        .drop_duplicates("base_query")
        .itertuples(index=False)
    }
    next_query = 1 + max(
        [int(value[0][1:]) for value in query_lookup.values()], default=0
    )
    manifest = json.loads(paths.experiment_manifest.read_text(encoding="utf-8"))
    created_at = manifest["created_at"].replace("-", "").replace(":", "")
    experiment_id = getattr(args, "experiment_id", None) or (
        f"{paths.root.name}-{created_at[:15]}Z"
    )
    prompt_metadata = pd.read_csv(paths.source_prompts).set_index("prompt_id")
    translation_checkpoint = load_translation_checkpoint(paths.translation_checkpoint)
    translation_lookup = translation_checkpoint.set_index(["prompt_id", "language_code"])
    job_lookup = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (paths.root / "jobs").glob("*.json")
    }

    generation_rows = []
    metadata_rows = []
    for row in responses.itertuples(index=False):
        if row.query_en not in query_lookup:
            query_lookup[row.query_en] = (f"q{next_query:03d}", slugify(row.query_en)[:80])
            next_query += 1
        query_id, query_tag = query_lookup[row.query_en]
        identifier = create_stable_id("gen", experiment_id, row.request_key)
        translated = row.language_code != "en"
        translation = (
            translation_lookup.loc[(row.prompt_id, row.language_code)]
            if translated
            else None
        )
        generation_rows.append(
            {
                "generation_id": identifier,
                "query_id": query_id,
                "query_tag": query_tag,
                "sample_idx": int(row.sample_idx),
                "base_query": row.query_en,
                "base_language_code": "en",
                "base_language": "English",
                "translated_query": row.query if translated else None,
                "translated_language_code": row.language_code if translated else None,
                "translated_language": row.language if translated else None,
                "query_translation_provider": (
                    translation["translation_provider"] if translated else None
                ),
                "query_translation_model": (
                    translation["translation_model"] if translated else None
                ),
                "model_response": row.response,
                "response_translation": None,
                "response_translation_language_code": None,
                "response_translation_provider": None,
                "provider": "google",
                "model_version": row.model,
                "generation_config": row.generation_config,
                "system_instruction": row.system_instruction,
            }
        )
        source = prompt_metadata.loc[row.prompt_id]
        job = job_lookup[slugify(row.model)]
        metadata_rows.append(
            {
                "generation_id": identifier,
                "experiment_id": experiment_id,
                "request_key": row.request_key,
                "batch_job_id": job["batch_name"],
                "request_status": "succeeded",
                "source_dataset": source["source_dataset"],
                "source_row_index": int(source["source_row_index"]),
                "selection_seed": int(source["selection_seed"]),
                "source_artifact": str(paths.generation_results(row.model)),
                "submitted_at": job["submitted_at"],
                "completed_at": None,
            }
        )

    combined = pd.concat([existing, pd.DataFrame(generation_rows)], ignore_index=True)
    combined_metadata = pd.concat(
        [existing_metadata, pd.DataFrame(metadata_rows)], ignore_index=True
    )
    combined = combined.drop_duplicates("generation_id", keep="last")
    combined_metadata = combined_metadata.drop_duplicates("generation_id", keep="last")
    write_csv(combined, generation_path, GENERATION_COLUMNS)
    write_csv(combined_metadata, metadata_path, GENERATION_METADATA_COLUMNS)
    print(f"Upserted {len(generation_rows)} rows into {generation_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=".cache/experiments/en_te_yo_30_query")
    parser.add_argument("--config", default=str(DEFAULT_GENERATION_CONFIG))


def add_api_confirmation_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur API charges",
    )


def require_api_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_api:
        raise PermissionError("API access requires the explicit --confirm-api flag")


def run_status_command(args: argparse.Namespace) -> None:
    require_api_confirmation(args)
    update_job_statuses(ExperimentPaths(Path(args.output_dir)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multilingual Gemini Batch API experiment"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="sample Infinity-Chat eval source prompts")
    add_common_arguments(prepare)
    prepare.add_argument("--cache-dir", default=".cache/huggingface")
    prepare.add_argument("--num-prompts", type=int, default=30)
    prepare.add_argument("--exclude-first", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(function=prepare_experiment)

    translate = commands.add_parser("translate-prompts")
    add_common_arguments(translate)
    translate.add_argument(
        "--translation-config",
        default=str(Path(__file__).with_name("translation_config.json")),
    )
    translate.set_defaults(function=translate_prompts)

    status = commands.add_parser("status")
    add_common_arguments(status)
    add_api_confirmation_argument(status)
    status.set_defaults(function=run_status_command)

    prepare_generation = commands.add_parser("prepare-generations")
    add_common_arguments(prepare_generation)
    prepare_generation.set_defaults(function=prepare_generations)

    submit_generation = commands.add_parser("submit-generations")
    add_common_arguments(submit_generation)
    add_api_confirmation_argument(submit_generation)
    submit_generation.set_defaults(function=submit_generations)

    download_generation = commands.add_parser("download-generations")
    add_common_arguments(download_generation)
    add_api_confirmation_argument(download_generation)
    download_generation.set_defaults(function=download_generations)

    finalize = commands.add_parser("finalize")
    add_common_arguments(finalize)
    finalize.add_argument("--generations", default=str(GENERATIONS_PATH))
    finalize.add_argument(
        "--generations-metadata", default=str(GENERATION_METADATA_PATH)
    )
    finalize.add_argument("--experiment-id")
    finalize.set_defaults(function=finalize_responses)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
