import json
from pathlib import Path
from types import SimpleNamespace

from lib.generate_with_gemini import (
    build_generation_request,
    download_batch,
    submit_batch,
)


class FakeFiles:
    def __init__(self) -> None:
        self.uploads = []
        self.downloads = []

    def upload(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(name="files/input-jsonl")

    def download(self, **kwargs):
        self.downloads.append(kwargs)
        return b'{"key":"request-1"}\n'


class FakeBatches:
    def __init__(self) -> None:
        self.creates = []
        self.gets = []

    def create(self, **kwargs):
        self.creates.append(kwargs)
        return SimpleNamespace(
            name="batches/test",
            state=SimpleNamespace(name="JOB_STATE_PENDING"),
        )

    def get(self, **kwargs):
        self.gets.append(kwargs)
        return SimpleNamespace(
            name="batches/test",
            state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
            error=None,
            dest=SimpleNamespace(file_name="files/results-jsonl"),
        )


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches()


def test_builds_only_configured_generation_request_fields() -> None:
    config = {"temperature": 0.2, "max_output_tokens": 100}

    without_system = build_generation_request("query", config, None)
    with_system = build_generation_request("query", config, "instruction")

    assert without_system == {
        "contents": [{"role": "user", "parts": [{"text": "query"}]}],
        "generation_config": config,
    }
    assert with_system["system_instruction"] == {
        "parts": [{"text": "instruction"}]
    }


def test_submits_and_downloads_batch_with_injected_client(tmp_path: Path) -> None:
    input_path = tmp_path / "requests.jsonl"
    job_path = tmp_path / "jobs" / "test.json"
    output_path = tmp_path / "results" / "test.jsonl"
    input_path.write_text('{"key":"request-1"}\n')
    client = FakeClient()

    submit_batch(input_path, "gemini-test", job_path, "test-batch", client)
    submitted = json.loads(job_path.read_text())

    assert submitted["batch_name"] == "batches/test"
    assert submitted["model"] == "gemini-test"
    assert submitted["state"] == "JOB_STATE_PENDING"
    assert client.batches.creates[0] == {
        "model": "gemini-test",
        "src": "files/input-jsonl",
        "config": {"display_name": "test-batch"},
    }

    download_batch(job_path, output_path, client)
    downloaded = json.loads(job_path.read_text())

    assert output_path.read_bytes() == b'{"key":"request-1"}\n'
    assert downloaded["state"] == "JOB_STATE_SUCCEEDED"
    assert downloaded["result_file_name"] == "files/results-jsonl"
    assert client.files.downloads == [{"file": "files/results-jsonl"}]
