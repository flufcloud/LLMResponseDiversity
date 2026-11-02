from dataclasses import dataclass
from pathlib import Path

from lib.compute_identifiers import slugify


@dataclass(frozen=True)
class ExperimentPaths:
    runtime_root: Path
    result_root: Path
    raw_output_dir: Path

    def generation_requests(self, input_name: str) -> Path:
        return self.runtime_root / "generation_requests" / f"{slugify(input_name)}.jsonl"

    def generation_manifest(self, input_name: str) -> Path:
        return self.runtime_root / "generation_requests" / f"{slugify(input_name)}_manifest.csv"

    def generation_job(self, input_name: str) -> Path:
        return self.runtime_root / "jobs" / f"{slugify(input_name)}.json"

    def generation_results(self, input_name: str) -> Path:
        return self.runtime_root / "batch_results" / f"{slugify(input_name)}.jsonl"

    @property
    def failures(self) -> Path:
        return self.result_root / "data" / "failed_requests.csv"

