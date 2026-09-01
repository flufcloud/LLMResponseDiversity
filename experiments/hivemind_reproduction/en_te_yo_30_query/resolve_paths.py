from dataclasses import dataclass
from pathlib import Path

from lib.compute_identifiers import slugify


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path

    @property
    def source_prompts(self) -> Path:
        return self.root / "source_prompts.csv"

    @property
    def multilingual_prompts(self) -> Path:
        return self.root / "prompts_multilingual.csv"

    @property
    def experiment_manifest(self) -> Path:
        return self.root / "experiment_manifest.json"

    @property
    def generation_config(self) -> Path:
        return self.root / "generation_config.json"

    @property
    def translation_checkpoint(self) -> Path:
        return self.root / "translation_checkpoint.csv"

    @property
    def responses(self) -> Path:
        return self.root / "infinity_chat_queries_gemini_responses.csv"

    @property
    def response_metadata(self) -> Path:
        return self.root / "response_metadata.csv"

    @property
    def failures(self) -> Path:
        return self.root / "failed_requests.csv"

    def generation_requests(self, model: str) -> Path:
        return self.root / "generation_requests" / f"{slugify(model)}.jsonl"

    def generation_manifest(self, model: str) -> Path:
        return self.root / "generation_requests" / f"{slugify(model)}_manifest.csv"

    def generation_job(self, model: str) -> Path:
        return self.root / "jobs" / f"{slugify(model)}.json"

    def generation_results(self, model: str) -> Path:
        return self.root / "batch_results" / f"{slugify(model)}.jsonl"
