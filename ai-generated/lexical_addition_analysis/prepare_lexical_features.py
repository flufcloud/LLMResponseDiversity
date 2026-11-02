from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from experiments.multimodal_thinking_experiment.analyze_words import load_spacy_model
from lib.compute_identifiers import compute_stable_hash


DEFAULT_CONFIG = Path(__file__).with_name("analysis_config.json")


def load_completed(path: Path) -> dict[str, str]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            completed[record["poem_id"]] = record["response_sha256"]
    return completed


def load_reusable_features(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            records[record["poem_id"]] = record
    return records


def extract_lemmas(document: Any, config: dict[str, Any]) -> list[str]:
    allowed_pos = set(config["content_pos"])
    excluded_terms = {
        term.lower().replace("’", "'") for term in config["excluded_terms"]
    }
    return [
        token.lemma_.lower().replace("’", "'")
        for token in document
        if token.is_alpha
        and token.pos_ in allowed_pos
        and not token.is_stop
        and token.lower_.replace("’", "'") not in excluded_terms
    ]


def prepare_corpus_features(
    corpus: str,
    corpus_config: dict[str, Any],
    lexical_config: dict[str, Any],
    nlp: Any,
) -> int:
    columns = [
        corpus_config["poem_id_column"],
        corpus_config["topic_column"],
        "response",
    ]
    adjective_column = corpus_config["adjective_column"]
    if adjective_column:
        columns.append(adjective_column)
    poems = pd.read_csv(corpus_config["input_csv"], usecols=columns, keep_default_na=False)
    output = Path(corpus_config["features_jsonl"])
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output)
    reusable_path = corpus_config.get("reuse_features_jsonl")
    reusable = load_reusable_features(
        Path(reusable_path) if reusable_path else None
    )

    pending = []
    reused = []
    for row in poems.itertuples(index=False):
        poem_id = str(getattr(row, corpus_config["poem_id_column"]))
        response_hash = compute_stable_hash(row.response)
        prior_hash = completed.get(poem_id)
        if prior_hash is not None and prior_hash != response_hash:
            raise ValueError(f"Lexical feature hash mismatch for {poem_id}")
        if prior_hash is None:
            reusable_record = reusable.get(poem_id)
            if reusable_record is not None:
                if reusable_record["response_sha256"] != response_hash:
                    raise ValueError(f"Reusable feature hash mismatch for {poem_id}")
                reused.append((row, poem_id, response_hash, reusable_record["lemmas"]))
            else:
                pending.append((row, poem_id, response_hash))

    with output.open("a", encoding="utf-8", buffering=1) as destination:
        for row, poem_id, response_hash, lemmas in tqdm(
            reused,
            desc=f"Reusing {corpus} lemmas",
            unit="poem",
        ):
            write_feature_record(
                destination,
                corpus,
                corpus_config,
                row,
                poem_id,
                response_hash,
                lemmas,
            )
    if not pending:
        return len(reused)

    documents = nlp.pipe(
        (item[0].response for item in pending),
        batch_size=lexical_config["batch_size"],
    )
    with output.open("a", encoding="utf-8", buffering=1) as destination:
        rows = zip(pending, documents, strict=True)
        for (row, poem_id, response_hash), document in tqdm(
            rows,
            total=len(pending),
            desc=f"Tagging {corpus} poems",
            unit="poem",
        ):
            write_feature_record(
                destination,
                corpus,
                corpus_config,
                row,
                poem_id,
                response_hash,
                extract_lemmas(document, lexical_config),
            )
    return len(reused) + len(pending)


def write_feature_record(
    destination: Any,
    corpus: str,
    corpus_config: dict[str, Any],
    row: Any,
    poem_id: str,
    response_hash: str,
    lemmas: list[str],
) -> None:
    adjective_column = corpus_config["adjective_column"]
    record = {
        "corpus": corpus,
        "poem_id": poem_id,
        "topic": getattr(row, corpus_config["topic_column"]),
        "adjective": getattr(row, adjective_column) if adjective_column else None,
        "response_sha256": response_hash,
        "lemmas": lemmas,
    }
    destination.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_lexical_features(config_path: Path, nlp: Any | None = None) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lexical_config = config["lexical"]
    nlp = nlp or load_spacy_model(lexical_config)
    return sum(
        prepare_corpus_features(corpus, corpus_config, lexical_config, nlp)
        for corpus, corpus_config in config["corpora"].items()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract retained content lemmas for lexical-addition analysis"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    count = prepare_lexical_features(parse_args().config)
    print(f"Prepared {count} new poem records")
