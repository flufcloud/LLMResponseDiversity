from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from lib.compute_identifiers import create_stable_id
from lib.measure_syntactic_similarity import (
    compute_line_wl_features,
    load_stanza_pipeline,
    split_poetic_lines,
)


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def serialize_features(features: dict[Any, int]) -> dict[str, int]:
    return {repr(feature): count for feature, count in features.items()}


def parse_line_features(
    config_path: Path,
    *,
    pipeline: Any | None = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    syntax_config = config["syntax"]
    responses = pd.read_csv(config["input_csv"], keep_default_na=False)
    pipeline = pipeline or load_stanza_pipeline(syntax_config)
    output_path = Path(config["line_features_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        rows = tqdm(
            responses.itertuples(index=False),
            total=len(responses),
            desc="Parsing syntax",
            unit="poem",
        )
        for row in rows:
            lines = split_poetic_lines(row.response)
            if hasattr(pipeline, "bulk_process"):
                documents = pipeline.bulk_process(lines)
            else:
                documents = [pipeline(line) for line in lines]
            for line_idx, (line, document) in enumerate(
                zip(lines, documents, strict=True), start=1
            ):
                features = compute_line_wl_features(
                    document,
                    syntax_config["wl_iterations"],
                )
                record = {
                    "line_id": create_stable_id(
                        "line", row.poem_id, line_idx
                    ),
                    "poem_id": row.poem_id,
                    "generation_id": row.generation_id,
                    "topic": row.topic,
                    "thinking_level": row.thinking_level,
                    "source_sample_idx": int(row.source_sample_idx),
                    "line_idx": line_idx,
                    "line_text": line,
                    "features": serialize_features(features),
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                output.write("\n")
    print(f"Wrote line features to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse poem lines into syntactic graph features"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    parse_line_features(arguments.config)
