from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Hashable

import pandas as pd

from lib.compute_identifiers import compute_stable_hash


ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "analysis_config.json"
GROUP_LEVELS = {
    "topic": ["topic_id", "topic"],
    "adjective_class": ["topic_id", "topic", "adjective_class"],
    "adjective": ["topic_id", "topic", "adjective_class", "adjective"],
}
METRICS = ("cosine_similarity", "lexical_jaccard", "syntactic_similarity")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_data_path(config: dict[str, Any], name: str) -> Path:
    return Path(config["data_dir"]) / name


def get_table_path(config: dict[str, Any], name: str) -> Path:
    return Path(config["tables_dir"]) / name


def load_responses(config: dict[str, Any]) -> pd.DataFrame:
    responses = pd.read_csv(get_data_path(config, "responses.csv"), keep_default_na=False)
    responses["response_id"] = responses["poem_id"]
    responses["response_sha256"] = responses["response"].map(compute_stable_hash)
    responses["response_input_kind"] = "poem"
    return responses


def freeze_json_value(value: Any) -> Hashable:
    if isinstance(value, list):
        return tuple(freeze_json_value(item) for item in value)
    return value


def serialize_syntax(features: list[Counter]) -> list[list[list[Any]]]:
    return [
        [[feature, count] for feature, count in line.items()]
        for line in features
    ]


def deserialize_syntax(payload: list[list[list[Any]]]) -> list[Counter]:
    return [
        Counter(
            {
                freeze_json_value(feature): int(count)
                for feature, count in line
            }
        )
        for line in payload
    ]


def load_text_features(
    config: dict[str, Any], responses: pd.DataFrame
) -> tuple[list[set[str]], list[list[Counter]]]:
    records = {}
    path = get_data_path(config, "line_text_features.jsonl")
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            records[record["poem_id"]] = record

    lemma_sets = []
    syntax_features = []
    for row in responses.itertuples(index=False):
        record = records[row.poem_id]
        if record["response_sha256"] != row.response_sha256:
            raise ValueError(f"Text feature hash mismatch for {row.poem_id}")
        lemma_sets.append(set(record["lemmas"]))
        syntax_features.append(deserialize_syntax(record["syntax"]))
    return lemma_sets, syntax_features


def get_group_seed(seed: int, level: str, keys: tuple[Any, ...]) -> int:
    payload = json.dumps([seed, level, keys], ensure_ascii=False, default=str)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)
