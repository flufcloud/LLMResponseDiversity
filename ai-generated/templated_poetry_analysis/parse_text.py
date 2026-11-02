from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from experiments.multimodal_thinking_experiment.analyze_words import load_spacy_model
from lib.measure_syntactic_similarity import (
    compute_line_wl_features,
    load_stanza_pipeline,
    split_poetic_lines,
)

from .process_analysis_data import (
    DEFAULT_CONFIG,
    get_data_path,
    load_config,
    load_responses,
    serialize_syntax,
)


def get_lemmas(document: Any, config: dict[str, Any]) -> list[str]:
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


def read_completed_features(path: Path) -> dict[str, str]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            completed[record["poem_id"]] = record["response_sha256"]
    return completed


def parse_batch(
    batch: list[Any],
    spacy_nlp: Any,
    stanza_nlp: Any,
    lexical_config: dict[str, Any],
    syntax_config: dict[str, Any],
) -> list[dict[str, Any]]:
    texts = [row.response for row in batch]
    spacy_documents = list(
        spacy_nlp.pipe(texts, batch_size=lexical_config["batch_size"])
    )
    poem_lines = [split_poetic_lines(text) for text in texts]
    flattened_lines = [line for lines in poem_lines for line in lines]
    stanza_documents = iter(stanza_nlp.bulk_process(flattened_lines))
    records = []
    for row, spacy_document, lines in zip(
        batch, spacy_documents, poem_lines, strict=True
    ):
        syntax = [
            compute_line_wl_features(
                next(stanza_documents), syntax_config["wl_iterations"]
            )
            for _ in lines
        ]
        records.append(
            {
                "poem_id": row.poem_id,
                "response_sha256": row.response_sha256,
                "lemmas": get_lemmas(spacy_document, lexical_config),
                "syntax": serialize_syntax(syntax),
            }
        )
    if next(stanza_documents, None) is not None:
        raise ValueError("Stanza returned more line documents than expected")
    return records


def parse_text(
    config_path: Path,
    spacy_nlp: Any | None = None,
    stanza_nlp: Any | None = None,
) -> int:
    config = load_config(config_path)
    responses = load_responses(config)
    output = get_data_path(config, "line_text_features.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_features(output)
    for row in responses.itertuples(index=False):
        prior_hash = completed.get(row.poem_id)
        if prior_hash is not None and prior_hash != row.response_sha256:
            raise ValueError(f"Text feature hash mismatch for {row.poem_id}")

    pending = [
        row
        for row in responses.itertuples(index=False)
        if row.poem_id not in completed
    ]
    lexical_config = config["lexical"]
    syntax_config = config["syntax"]
    spacy_nlp = spacy_nlp or load_spacy_model(lexical_config)
    stanza_nlp = stanza_nlp or load_stanza_pipeline(syntax_config)
    batch_size = syntax_config["batch_size"]
    with output.open("a", encoding="utf-8") as output_file:
        progress = tqdm(total=len(pending), desc="Parsing poems", unit="poem")
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            records = parse_batch(
                batch,
                spacy_nlp,
                stanza_nlp,
                lexical_config,
                syntax_config,
            )
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            progress.update(len(batch))
        progress.close()
    return len(pending)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse lexical and syntactic features for adjective poems"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    count = parse_text(parse_args().config)
    print(f"Parsed {count} new poems")
