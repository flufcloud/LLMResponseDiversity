from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from experiments.wildchat_multilingual_open_ended.resolve_paths import ExperimentPaths
from lib.generate_with_gemini import (
    build_generation_request,
    download_batch,
    get_gemini_client,
    parse_batch_results,
    submit_batch,
    update_batch_jobs,
)


EXPERIMENT_CONFIG = Path(
    "experiments/wildchat_multilingual_open_ended/experiment_config.json"
)
GENERATION_CONFIG = Path(
    "experiments/wildchat_multilingual_open_ended/generation_config.json"
)
RUNTIME_ROOT = Path(".cache/experiments/wildchat_multilingual_open_ended")


@dataclass(frozen=True)
class InputSettings:
    language: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class ExperimentSettings:
    experiment_id: str
    inputs: tuple[InputSettings, ...]
    input_column: str
    output_filename_suffix: str
    output_columns: tuple[str, ...]
    paths: ExperimentPaths


@dataclass(frozen=True)
class GenerationSettings:
    model: str
    responses_per_query: int
    system_instruction_template_path: Path
    user_query_placeholder: str
    generation_config: dict[str, Any]
    user_approved_for_api: bool


def load_experiment_settings(path: Path, runtime_root: Path) -> ExperimentSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result_root = Path(raw["output_root"])
    paths = ExperimentPaths(runtime_root, result_root, Path(raw["raw_output_dir"]))
    inputs = tuple(
        InputSettings(row["language"], Path(row["path"])) for row in raw["inputs"]
    )
    return ExperimentSettings(
        experiment_id=raw["experiment_id"],
        inputs=inputs,
        input_column=raw["input_column"],
        output_filename_suffix=raw["output_filename_suffix"],
        output_columns=tuple(raw["output_columns"]),
        paths=paths,
    )


def load_generation_settings(path: Path) -> GenerationSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if len(raw["models"]) != 1 or raw["responses_per_query"] != 1:
        raise ValueError("This experiment requires one model and one response per query")
    return GenerationSettings(
        model=raw["models"][0],
        responses_per_query=raw["responses_per_query"],
        system_instruction_template_path=Path(raw["system_instruction_template_path"]),
        user_query_placeholder=raw["user_query_placeholder"],
        generation_config=raw["generation_config"],
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


def build_classifier_request(
    query: str,
    prompt_template: str,
    settings: GenerationSettings,
) -> dict[str, Any]:
    quoted_query = json.dumps(query, ensure_ascii=False)
    instruction = prompt_template.replace(settings.user_query_placeholder, quoted_query)
    return build_generation_request(query, settings.generation_config, instruction)


def prepare_input_requests(
    source: InputSettings,
    experiment: ExperimentSettings,
    generation: GenerationSettings,
    owners: dict[str, tuple[str, str]],
) -> int:
    rows = pd.read_csv(source.path, keep_default_na=False)
    rows = rows.reset_index(names="source_row").drop_duplicates(
        experiment.input_column, keep="first"
    )
    prompt_template = generation.system_instruction_template_path.read_text(
        encoding="utf-8"
    )
    manifest = []
    requests = []
    for row in rows.itertuples(index=False):
        query = str(getattr(row, experiment.input_column))
        for sample_index in range(1, generation.responses_per_query + 1):
            owner = owners.get(query)
            if owner is None:
                request_key = (
                    f"classifier--{source.name.lower()}--"
                    f"{row.source_row:06d}--{sample_index:03d}"
                )
                owner = (source.name, request_key)
                owners[query] = owner
                requests.append(
                    {
                        "key": request_key,
                        "request": build_classifier_request(
                            query, prompt_template, generation
                        ),
                    }
                )
            result_source, request_key = owner
            manifest.append(
                {
                    "request_key": request_key,
                    "result_source": result_source,
                    "source_file": str(source.path),
                    "source_row": row.source_row,
                    "source_language": source.language,
                    "original_query": query,
                    "sample_index": sample_index,
                }
            )
    manifest_path = experiment.paths.generation_manifest(source.name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    return write_jsonl(experiment.paths.generation_requests(source.name), requests)


def prepare_requests(args: argparse.Namespace) -> None:
    experiment, generation = load_settings(args)
    owners = {}
    total = 0
    for source in experiment.inputs:
        count = prepare_input_requests(source, experiment, generation, owners)
        total += count
        print(f"Wrote {count} requests for {source.language}")
    print(f"Wrote {total} distinct requests across all inputs")


def validate_input_requests(
    source: InputSettings,
    experiment: ExperimentSettings,
    generation: GenerationSettings,
) -> None:
    source_rows = pd.read_csv(source.path, keep_default_na=False)
    manifest = pd.read_csv(
        experiment.paths.generation_manifest(source.name), keep_default_na=False
    )
    prompt_template = generation.system_instruction_template_path.read_text(
        encoding="utf-8"
    )
    requests_path = experiment.paths.generation_requests(source.name)
    prepared = {}
    with requests_path.open(encoding="utf-8") as lines:
        for line in lines:
            item = json.loads(line)
            prepared[item["key"]] = item["request"]

    for row in manifest.itertuples(index=False):
        query = str(source_rows.iloc[row.source_row][experiment.input_column])
        if query != row.original_query:
            raise ValueError(f"Source query changed for {row.request_key}")
        if row.result_source != source.name:
            continue
        expected = build_classifier_request(query, prompt_template, generation)
        if prepared.get(row.request_key) != expected:
            raise ValueError(
                f"Prepared request {row.request_key} does not match the approved configuration"
            )
    owned_keys = set(
        manifest.loc[manifest["result_source"] == source.name, "request_key"]
    )
    if set(prepared) != owned_keys:
        raise ValueError(f"Prepared request count differs from {source.name} manifest")


def validate_requests(args: argparse.Namespace) -> None:
    experiment, generation = load_settings(args)
    for source in experiment.inputs:
        validate_input_requests(source, experiment, generation)
        print(f"Validated {source.language} requests")


def require_api_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_api:
        raise PermissionError("API access requires the explicit --confirm-api flag")


def submit_requests(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    experiment, generation = load_settings(args)
    if not generation.user_approved_for_api:
        raise PermissionError(
            f"The user must set user_approved_for_api to true in {args.generation_config}"
        )
    for source in experiment.inputs:
        validate_input_requests(source, experiment, generation)
    client = get_gemini_client(client)
    for source in experiment.inputs:
        submit_batch(
            experiment.paths.generation_requests(source.name),
            generation.model,
            experiment.paths.generation_job(source.name),
            f"{experiment.experiment_id}-{source.name.lower()}",
            client,
        )


def show_status(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    experiment, _ = load_settings(args)
    job_paths = [
        experiment.paths.generation_job(source.name) for source in experiment.inputs
    ]
    for job_path, record in update_batch_jobs(job_paths, client):
        print(f"{job_path.name}: {record['state']} ({record['batch_name']})")


def download_results(args: argparse.Namespace, client: Any | None = None) -> None:
    require_api_confirmation(args)
    experiment, _ = load_settings(args)
    client = get_gemini_client(client)
    for source in experiment.inputs:
        download_batch(
            experiment.paths.generation_job(source.name),
            experiment.paths.generation_results(source.name),
            client,
        )


def parse_classifier_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        value = "\n".join(lines)
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Classifier response is not a JSON object")
    return result


def serialize_output_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def finalize_results(args: argparse.Namespace) -> None:
    experiment, _ = load_settings(args)
    manifests = {
        source.name: pd.read_csv(
            experiment.paths.generation_manifest(source.name), keep_default_na=False
        )
        for source in experiment.inputs
    }
    query_by_key = {
        row.request_key: row.original_query
        for manifest in manifests.values()
        for row in manifest.itertuples(index=False)
    }
    responses = {}
    failures = []
    for source in experiment.inputs:
        source_responses, source_failures = parse_batch_results(
            experiment.paths.generation_results(source.name)
        )
        duplicate_keys = set(responses) & set(source_responses)
        failures.extend(
            {"request_key": key, "error": "Duplicate result across batches"}
            for key in sorted(duplicate_keys)
        )
        responses.update(source_responses)
        failures.extend(source_failures)
    failures.extend(
        {"request_key": key, "error": "Unknown result key"}
        for key in sorted(set(responses) - set(query_by_key))
    )
    failed_keys = {failure["request_key"] for failure in failures}
    failures.extend(
        {"request_key": key, "error": "Missing result"}
        for key in sorted(set(query_by_key) - set(responses) - failed_keys)
    )

    parsed = {}
    failed_keys = {failure["request_key"] for failure in failures}
    for request_key in query_by_key:
        if request_key in failed_keys:
            continue
        try:
            result = parse_classifier_json(responses[request_key])
            missing = set(experiment.output_columns) - set(result)
            if missing:
                raise ValueError(f"Missing output fields: {sorted(missing)}")
            parsed[request_key] = result
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            failures.append({"request_key": request_key, "error": str(error)})

    if failures:
        experiment.paths.failures.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(experiment.paths.failures, index=False)
        print(f"Skipped {len(failures)} unusable responses; see {experiment.paths.failures}")

    experiment.paths.raw_output_dir.mkdir(parents=True, exist_ok=True)
    for request_key, result in parsed.items():
        output_path = experiment.paths.raw_output_dir / f"{request_key}.json"
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for source in experiment.inputs:
        records = [
            {
                column: serialize_output_value(parsed[row.request_key][column])
                for column in experiment.output_columns
            }
            for row in manifests[source.name].itertuples(index=False)
            if row.request_key in parsed
        ]
        output_path = (
            experiment.paths.result_root
            / "data"
            / f"{source.name}{experiment.output_filename_suffix}.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records, columns=experiment.output_columns).to_csv(
            output_path, index=False
        )
        print(f"Wrote {len(records)} classifier rows to {output_path}")


def load_settings(
    args: argparse.Namespace,
) -> tuple[ExperimentSettings, GenerationSettings]:
    experiment = load_experiment_settings(
        Path(args.experiment_config), Path(args.runtime_dir)
    )
    generation = load_generation_settings(Path(args.generation_config))
    return experiment, generation


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-config", default=str(EXPERIMENT_CONFIG))
    parser.add_argument("--generation-config", default=str(GENERATION_CONFIG))
    parser.add_argument("--runtime-dir", default=str(RUNTIME_ROOT))


def add_api_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-api",
        action="store_true",
        help="confirm that this command may contact Gemini and incur API charges",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify multilingual WildChat queries with the Gemini Batch API"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in {
        "prepare": prepare_requests,
        "validate": validate_requests,
        "finalize": finalize_results,
    }.items():
        command = commands.add_parser(name)
        add_common_arguments(command)
        command.set_defaults(function=function)
    for name, function in {
        "submit": submit_requests,
        "status": show_status,
        "download": download_results,
    }.items():
        command = commands.add_parser(name)
        add_common_arguments(command)
        add_api_confirmation(command)
        command.set_defaults(function=function)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
