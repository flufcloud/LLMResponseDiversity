from __future__ import annotations

import base64
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode

from google import genai
from google.genai import types
from tqdm import tqdm

from lib.compute_identifiers import compute_file_sha256


@dataclass(frozen=True)
class TextGenerationResult:
    text: str
    thoughts: tuple[str, ...]
    finish_reason: str
    model_version: str
    prompt_token_count: int | None
    response_token_count: int | None
    thought_token_count: int | None
    total_token_count: int | None


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


def build_batch_generation_request(
    text: str,
    generation_config: dict[str, Any],
    system_instruction: str | None,
) -> dict[str, Any]:
    serialized_config = types.GenerateContentConfig(**generation_config).model_dump(
        by_alias=True,
        exclude_none=True,
    )
    request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": serialized_config,
    }
    if system_instruction:
        request["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }
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
    progress_factory: Callable[..., Any] = tqdm,
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
    download_file_with_progress(
        client,
        batch_job.dest.file_name,
        output_path,
        progress_factory,
    )
    record["result_file_name"] = batch_job.dest.file_name
    record["result_path"] = str(output_path)
    record["result_sha256"] = compute_file_sha256(output_path)
    record["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    write_job_record(job_path, record)
    print(f"Downloaded {record['batch_name']} to {output_path}")


def download_file_with_progress(
    client: Any,
    file_name: str,
    output_path: Path,
    progress_factory: Callable[..., Any] = tqdm,
    chunk_size: int = 1024 * 1024,
) -> None:
    name = file_name.rsplit("/", 1)[-1]
    path = f"files/{name}:download?{urlencode({'alt': 'media'})}"
    api_client = getattr(client.files, "_api_client", None)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".part",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            if api_client is None:
                data = client.files.download(file=file_name)
                chunks = (
                    data[offset : offset + chunk_size]
                    for offset in range(0, len(data), chunk_size)
                )
                write_download_chunks(
                    output, chunks, len(data), output_path.name, progress_factory
                )
            else:
                request = api_client._build_request(
                    "get",
                    path=path,
                    request_dict={},
                    http_options=None,
                )
                with api_client._httpx_client.stream(
                    method=request.method,
                    url=request.url,
                    headers=request.headers,
                    timeout=request.timeout,
                ) as response:
                    response.raise_for_status()
                    total = response.headers.get("content-length")
                    write_download_chunks(
                        output,
                        response.iter_bytes(chunk_size=chunk_size),
                        int(total) if total else None,
                        output_path.name,
                        progress_factory,
                    )
        temporary_path.replace(output_path)
    except BaseException:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise


def write_download_chunks(
    output: Any,
    chunks: Iterable[bytes],
    total: int | None,
    filename: str,
    progress_factory: Callable[..., Any],
) -> None:
    with progress_factory(
        total=total,
        desc=f"Downloading {filename}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for chunk in chunks:
            output.write(chunk)
            progress.update(len(chunk))


def extract_result_key(item: dict[str, Any]) -> str | None:
    if item.get("key"):
        return str(item["key"])
    metadata = item.get("metadata") or {}
    return str(metadata["key"]) if metadata.get("key") else None


def extract_response_text_and_thoughts(
    item: dict[str, Any],
) -> tuple[str | None, list[str], str | None]:
    error = item.get("error") or item.get("status")
    if error:
        return None, [], json.dumps(error, ensure_ascii=False)
    response = item.get("response") or item
    candidates = response.get("candidates") or []
    if not candidates:
        return None, [], "Response contained no candidates"
    parts = (candidates[0].get("content") or {}).get("parts") or []
    thoughts = [
        str(part["text"])
        for part in parts
        if part.get("thought") and part.get("text")
    ]
    texts = [
        str(part["text"])
        for part in parts
        if part.get("text") and not part.get("thought")
    ]
    if not texts:
        return None, thoughts, "Response contained no text"
    return "\n".join(texts).strip(), thoughts, None


def extract_response_text(item: dict[str, Any]) -> tuple[str | None, str | None]:
    text, _, error = extract_response_text_and_thoughts(item)
    return text, error


def get_optional_int(values: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if values.get(key) is not None:
            return int(values[key])
    return None


def extract_text_generation_result(
    item: dict[str, Any],
) -> tuple[TextGenerationResult | None, str | None]:
    error = item.get("error") or item.get("status")
    if error:
        return None, json.dumps(error, ensure_ascii=False)

    response = item.get("response") or item
    candidates = response.get("candidates") or []
    if not candidates:
        return None, "Response contained no candidates"

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
    if finish_reason != "STOP":
        return None, f"Response finished with {finish_reason or 'no finish reason'}"

    parts = (candidate.get("content") or {}).get("parts") or []
    thoughts = tuple(
        str(part["text"])
        for part in parts
        if part.get("thought") and part.get("text")
    )
    texts = [
        str(part["text"])
        for part in parts
        if part.get("text") and not part.get("thought")
    ]
    if not texts:
        return None, "Response contained no text"

    model_version = response.get("modelVersion") or response.get("model_version")
    if not model_version:
        return None, "Response contained no model version"

    usage = response.get("usageMetadata") or response.get("usage_metadata")
    if not usage:
        return None, "Response contained no usage metadata"

    return TextGenerationResult(
        text="\n".join(texts).strip(),
        thoughts=thoughts,
        finish_reason=str(finish_reason),
        model_version=str(model_version),
        prompt_token_count=get_optional_int(
            usage, "promptTokenCount", "prompt_token_count"
        ),
        response_token_count=get_optional_int(
            usage, "candidatesTokenCount", "candidates_token_count"
        ),
        thought_token_count=get_optional_int(
            usage, "thoughtsTokenCount", "thoughts_token_count"
        ),
        total_token_count=get_optional_int(
            usage, "totalTokenCount", "total_token_count"
        ),
    ), None


def extract_response_image(
    item: dict[str, Any],
) -> tuple[bytes | None, str | None, list[str], str | None]:
    error = item.get("error") or item.get("status")
    if error:
        return None, None, [], json.dumps(error, ensure_ascii=False)
    response = item.get("response") or item
    candidates = response.get("candidates") or []
    if not candidates:
        return None, None, [], "Response contained no candidates"

    parts = (candidates[0].get("content") or {}).get("parts") or []
    thoughts = [
        str(part["text"])
        for part in parts
        if part.get("thought") and part.get("text")
    ]
    images = [
        part.get("inlineData") or part.get("inline_data")
        for part in parts
        if not part.get("thought")
        and (part.get("inlineData") or part.get("inline_data"))
    ]
    if len(images) != 1:
        return None, None, thoughts, f"Response contained {len(images)} final images"

    image = images[0]
    mime_type = image.get("mimeType") or image.get("mime_type")
    data = image.get("data")
    if not mime_type or not data:
        return None, None, thoughts, "Final image lacked MIME type or data"
    try:
        return base64.b64decode(data, validate=True), str(mime_type), thoughts, None
    except (ValueError, TypeError) as exc:
        return None, None, thoughts, f"Final image data was not valid base64: {exc}"


def parse_batch_results(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    successes_with_thoughts, failures = parse_batch_results_with_thoughts(path)
    successes = {
        key: response for key, (response, _) in successes_with_thoughts.items()
    }
    return successes, failures


def parse_batch_results_with_thoughts(
    path: Path,
) -> tuple[dict[str, tuple[str, list[str]]], list[dict[str, str]]]:
    successes: dict[str, tuple[str, list[str]]] = {}
    failures = []
    with path.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = extract_result_key(item)
            text, thoughts, error = extract_response_text_and_thoughts(item)
            if not key:
                failures.append(
                    {"request_key": "", "error": f"Line {line_number}: missing key"}
                )
            elif error:
                failures.append({"request_key": key, "error": error})
            elif key in successes:
                failures.append({"request_key": key, "error": "Duplicate result key"})
            else:
                successes[key] = (text or "", thoughts)
    return successes, failures


def parse_batch_text_generation_results(
    path: Path,
) -> tuple[dict[str, TextGenerationResult], list[dict[str, str]]]:
    successes = {}
    failures = []
    with path.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = extract_result_key(item)
            result, error = extract_text_generation_result(item)
            if not key:
                failures.append(
                    {"request_key": "", "error": f"Line {line_number}: missing key"}
                )
            elif error:
                failures.append({"request_key": key, "error": error})
            elif key in successes:
                failures.append({"request_key": key, "error": "Duplicate result key"})
            elif result:
                successes[key] = result
    return successes, failures
