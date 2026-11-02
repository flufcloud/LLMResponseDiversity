from pathlib import Path
from types import SimpleNamespace

from lib.generate_with_gemini import download_batch


class FakeProgress:
    def __init__(self, updates: list[int], **settings):
        self.settings = settings
        self.updates = updates

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def update(self, size: int) -> None:
        self.updates.append(size)


class FakeResponse:
    headers = {"content-length": "6"}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int):
        assert chunk_size == 1024 * 1024
        yield b"abc"
        yield b"def"


class FakeHttpClient:
    def stream(self, **request):
        assert request["method"] == "get"
        assert request["url"] == "https://example.test/files/result:download?alt=media"
        return FakeResponse()


class FakeApiClient:
    _httpx_client = FakeHttpClient()

    def _build_request(self, method, path, request_dict, http_options):
        assert path == "files/result:download?alt=media"
        assert request_dict == {}
        assert http_options is None
        return SimpleNamespace(
            method=method,
            url=f"https://example.test/{path}",
            headers={"authorization": "test"},
            timeout=60,
        )


class FakeFiles:
    _api_client = FakeApiClient()


class FakeBatches:
    def get(self, name):
        assert name == "batches/test"
        return SimpleNamespace(
            state=SimpleNamespace(name="JOB_STATE_SUCCEEDED"),
            error=None,
            dest=SimpleNamespace(file_name="files/result"),
        )


def test_download_batch_streams_progress_and_writes_atomically(tmp_path: Path) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(
        '{"batch_name":"batches/test","model":"image-model"}',
        encoding="utf-8",
    )
    output_path = tmp_path / "result.jsonl"
    updates = []
    progress_settings = {}

    def create_progress(**settings):
        progress_settings.update(settings)
        return FakeProgress(updates, **settings)

    client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
    download_batch(job_path, output_path, client, create_progress)

    assert output_path.read_bytes() == b"abcdef"
    assert updates == [3, 3]
    assert progress_settings["total"] == 6
    assert progress_settings["unit"] == "B"
    assert not list(tmp_path.glob("*.part"))
