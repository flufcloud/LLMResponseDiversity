from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from google import genai
from google.genai import types

from lib.compute_identifiers import compute_file_sha256


def get_gemini_client(client: Any | None = None) -> Any:
    return client or genai.Client()


def build_generation_request(
    text: str,
    generation_config: dict[str, Any],
    system_instruction: str | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generation_config": generation_config,
    }
    if system_instruction:
        request["system_instruction"] = {"parts": [{"text": system_instruction}]}
    return request


def write_job_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def get_batch_state(batch_job: Any) -> str:
    state = batch_job.state
    return state.name if hasattr(state, "name") else str(state)


def submit_batch(
    input_path: Path,
    model: str,
    job_path: Path,
    display_name: str,
    client: Any | None = None,
) -> None:
    if job_path.exists():
        existing = json.loads(job_path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"Job record already exists at {job_path} ({existing.get('batch_name')}); "
            "refusing duplicate submission"
        )
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    client = get_gemini_client(client)
    uploaded = client.files.upload(
        file=input_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    batch_job = client.batches.create(
        model=model,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    state = get_batch_state(batch_job)
    write_job_record(
        job_path,
        {
            "batch_name": batch_job.name,
            "display_name": display_name,
            "model": model,
            "input_path": str(input_path),
            "input_sha256": compute_file_sha256(input_path),
            "uploaded_file_name": uploaded.name,
            "state": state,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Submitted {batch_job.name} for {model}; state={state}")


def update_batch_job(
    job_path: Path,
    client: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    record = json.loads(job_path.read_text(encoding="utf-8"))
    client = get_gemini_client(client)
    batch_job = client.batches.get(name=record["batch_name"])
    record["state"] = get_batch_state(batch_job)
    record["checked_at"] = datetime.now(timezone.utc).isoformat()
    if batch_job.error:
        record["error"] = str(batch_job.error)
    write_job_record(job_path, record)
    return batch_job, record


def update_batch_jobs(
    job_paths: Iterable[Path],
    client: Any | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    client = get_gemini_client(client)
    updated = []
    for job_path in job_paths:
        _, record = update_batch_job(job_path, client)
        updated.append((job_path, record))
    return updated


def download_batch(
    job_path: Path,
    output_path: Path,
    client: Any | None = None,
) -> None:
    client = get_gemini_client(client)
    batch_job, record = update_batch_job(job_path, client)
    if record["state"] != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"{record['batch_name']} is {record['state']}; results are not ready"
        )
    if not batch_job.dest or not batch_job.dest.file_name:
        raise RuntimeError(f"{record['batch_name']} has no result file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(client.files.download(file=batch_job.dest.file_name))
    record["result_file_name"] = batch_job.dest.file_name
    record["result_path"] = str(output_path)
    record["result_sha256"] = compute_file_sha256(output_path)
    record["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    write_job_record(job_path, record)
    print(f"Downloaded {record['batch_name']} to {output_path}")


def extract_result_key(item: dict[str, Any]) -> str | None:
    if item.get("key"):
        return str(item["key"])
    metadata = item.get("metadata") or {}
    return str(metadata["key"]) if metadata.get("key") else None


def extract_response_text(item: dict[str, Any]) -> tuple[str | None, str | None]:
    error = item.get("error") or item.get("status")
    if error:
        return None, json.dumps(error, ensure_ascii=False)
    response = item.get("response") or item
    candidates = response.get("candidates") or []
    if not candidates:
        return None, "Response contained no candidates"
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [
        str(part["text"])
        for part in parts
        if part.get("text") and not part.get("thought")
    ]
    if not texts:
        return None, "Response contained no text"
    return "\n".join(texts).strip(), None


def parse_batch_results(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    successes: dict[str, str] = {}
    failures = []
    with path.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = extract_result_key(item)
            text, error = extract_response_text(item)
            if not key:
                failures.append(
                    {"request_key": "", "error": f"Line {line_number}: missing key"}
                )
            elif error:
                failures.append({"request_key": key, "error": error})
            elif key in successes:
                failures.append({"request_key": key, "error": "Duplicate result key"})
            else:
                successes[key] = text or ""
    return successes, failures
