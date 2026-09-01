import json
from pathlib import Path


def write_experiment_metadata(
    path: Path,
    experiment_id: str,
    embedding_model: str,
    max_token_length: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str | int] = {
        "experiment_id": experiment_id,
        "embedding_model": embedding_model,
    }
    if max_token_length is not None:
        metadata["max_token_length"] = max_token_length
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
