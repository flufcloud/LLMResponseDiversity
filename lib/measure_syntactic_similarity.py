from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Hashable

from tqdm import tqdm


type WlFeature = tuple[int, Hashable]
type WlFeatures = Counter[WlFeature]


def create_dependency_graph(
    sentence: Any,
) -> tuple[dict[int, str], dict[int, set[int]]]:
    labels = {0: "none"}
    neighbors = {0: set()}
    for word in sentence.words:
        word_id = int(word.id)
        head_id = int(word.head)
        labels[word_id] = word.upos
        neighbors.setdefault(word_id, set()).add(head_id)
        neighbors.setdefault(head_id, set()).add(word_id)
    return labels, neighbors


def compute_wl_features(sentence: Any, iterations: int) -> WlFeatures:
    labels, neighbors = create_dependency_graph(sentence)
    current: dict[int, Hashable] = labels
    features = Counter((0, label) for label in current.values())
    for iteration in range(1, iterations + 1):
        current = {
            node: (
                current[node],
                tuple(sorted((current[item] for item in neighbors[node]), key=repr)),
            )
            for node in current
        }
        features.update((iteration, label) for label in current.values())
    return features


def compute_normalized_wl_kernel(left: WlFeatures, right: WlFeatures) -> float:
    shared = left.keys() & right.keys()
    dot_product = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value**2 for value in left.values()))
    right_norm = math.sqrt(sum(value**2 for value in right.values()))
    return dot_product / (left_norm * right_norm)


def compute_poem_syntactic_similarity(
    left: list[WlFeatures],
    right: list[WlFeatures],
) -> float:
    scores = [
        compute_normalized_wl_kernel(left_line, right_line)
        for left_line in left
        for right_line in right
    ]
    return sum(scores) / len(scores)


def split_poetic_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def compute_line_wl_features(document: Any, iterations: int) -> WlFeatures:
    features: WlFeatures = Counter()
    for sentence in document.sentences:
        features.update(compute_wl_features(sentence, iterations))
    return features


def parse_text_line_syntax(
    text: str,
    pipeline: Any,
    iterations: int,
) -> list[WlFeatures]:
    lines = split_poetic_lines(text)
    if not lines:
        raise ValueError("Enter text containing at least one nonempty line")
    if hasattr(pipeline, "bulk_process"):
        documents = list(pipeline.bulk_process(lines))
    else:
        documents = [pipeline(line) for line in lines]
    if len(documents) != len(lines):
        raise ValueError("Stanza returned an unexpected number of line documents")
    return [
        compute_line_wl_features(document, iterations)
        for document in documents
    ]


def parse_poem_line_syntax(
    responses: list[str],
    pipeline: Any,
    iterations: int,
) -> list[list[WlFeatures]]:
    parsed = []
    for response in tqdm(responses, desc="Parsing poem lines", unit="poem"):
        parsed.append(parse_text_line_syntax(response, pipeline, iterations))
    return parsed


def load_stanza_pipeline(config: dict[str, Any]) -> Any:
    import stanza

    return stanza.Pipeline(
        lang=config["language"],
        dir=str(Path(config["model_dir"])),
        processors=config["processors"],
        package=config["package"],
        use_gpu=config["use_gpu"],
        download_method=None,
        verbose=False,
    )
