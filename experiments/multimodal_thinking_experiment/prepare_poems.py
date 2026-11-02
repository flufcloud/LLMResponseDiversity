from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from lib.compute_identifiers import compute_stable_hash, create_stable_id


EXPERIMENT_ID = "multimodal_thinking_experiment"
EXPERIMENT_ROOT = Path(__file__).parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "analysis_config.json"
DEFAULT_RESULTS_ROOT = Path("results") / EXPERIMENT_ID
DIVIDER_PATTERN = re.compile(r"(?m)^\s*(?:\*{3,}|-{3,}|_{3,})\s*$")
POEM_MARKER_PATTERN = re.compile(
    r"^(?:(?:(?:step|part|phase)\s*)?\d+\s*[:.)-]\s*)?"
    r"(?:(?:a|the)\s+)?poem(?:\s+(?:about|on)\s+.+|\s*[:(].*)?$",
    re.IGNORECASE,
)
VS5_LINE_BREAK_PATTERN = re.compile(r"\s+(?:/|\\)\s+")
VS5_TEXT_KEYS = ("poem", "text", "content", "response", "poem_text")
VS5_PROBABILITY_KEYS = (
    "probability",
    "estimated_probability",
    "probability_score",
    "generative_probability",
    "poem_probability",
)


def load_analysis_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines[1:]).strip()


def is_poem_marker(line: str) -> bool:
    plain = line.strip().lstrip("#").strip().strip("*").strip()
    is_bold_poem_heading = bool(
        re.fullmatch(r"\*\*[^*]+\*\*", line.strip())
        and re.search(r"\bpoem\b", plain, re.I)
    )
    return bool(
        POEM_MARKER_PATTERN.fullmatch(plain)
        or re.search(r"\bhere is (?:the |a )?(?:resulting )?poem\b", plain, re.I)
        or is_bold_poem_heading
    )


def strip_plain_title(text: str) -> str:
    blocks = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    if len(blocks) != 2 or len(blocks[0].splitlines()) != 1:
        return text.strip()
    title = blocks[0].strip()
    looks_like_title = (
        len(title.split()) <= 10
        and title.istitle()
        and not re.search(r"[,;:.!?]$", title)
    )
    return blocks[1].strip() if looks_like_title else text.strip()


def strip_leading_poem_labels(text: str) -> str:
    lines = text.strip().splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines:
        line = lines[0].strip()
        is_heading = line.startswith("#")
        is_bold_title = bool(
            re.fullmatch(r"\*\*[^*]+\*\*", line)
            and not re.match(r"\*\*[A-Za-z0-9]\*\*", line)
        )
        is_section_label = is_poem_marker(line)
        if not (is_heading or is_bold_title or is_section_label):
            break
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    if len(lines) >= 3 and re.match(
        r"^\*(?:Using|Inspired|Constructed)", lines[0].strip(), re.I
    ):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return strip_plain_title("\n".join(lines))


def extract_ssot_poem(text: str) -> tuple[str, str]:
    dividers = list(DIVIDER_PATTERN.finditer(text))
    if dividers:
        poem = text[dividers[-1].end() :]
        return strip_leading_poem_labels(poem), "last_divider"

    lines = text.splitlines()
    marker_indices = [
        index
        for index, line in enumerate(lines[:-1])
        if is_poem_marker(line)
    ]
    if marker_indices:
        poem = "\n".join(lines[marker_indices[-1] + 1 :])
        return strip_leading_poem_labels(poem), "poem_marker"

    heading_indices = [
        index for index, line in enumerate(lines[:-1]) if line.lstrip().startswith("#")
    ]
    start = heading_indices[-1] + 1 if heading_indices else 1
    poem = strip_leading_poem_labels("\n".join(lines[start:]))
    blocks = re.split(r"\n\s*\n", poem, maxsplit=1)
    if len(blocks) == 2 and len(blocks[0].splitlines()) == 1:
        poem = strip_leading_poem_labels(blocks[1])
    return poem, "last_heading"


def find_vs5_responses(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list) and len(value) == 5 and all(
        isinstance(item, dict) for item in value
    ):
        has_expected_fields = all(
            any(key in item for key in VS5_TEXT_KEYS)
            and any(key in item for key in VS5_PROBABILITY_KEYS)
            for item in value
        )
        if has_expected_fields:
            return value
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return None
    for child in children:
        found = find_vs5_responses(child)
        if found is not None:
            return found
    return None


def parse_probability(value: Any) -> float:
    if isinstance(value, str) and value.endswith("%"):
        return float(value.removesuffix("%")) / 100
    return float(value)


def extract_vs5_poems(text: str) -> list[tuple[str, float]]:
    payload = json.loads(strip_json_fence(text))
    responses = find_vs5_responses(payload)
    if responses is None:
        if isinstance(payload, dict) and isinstance(payload.get("poem"), str):
            raise ValueError(
                "Found one poem with five probability descriptions, not five poems"
            )
        raise ValueError("Expected 5 verbalized poem and probability records")

    extracted = []
    for item in responses:
        poem_key = next(key for key in VS5_TEXT_KEYS if key in item)
        probability_key = next(key for key in VS5_PROBABILITY_KEYS if key in item)
        extracted.append(
            (
                strip_leading_poem_labels(
                    VS5_LINE_BREAK_PATTERN.sub("\n", item[poem_key].strip())
                ),
                parse_probability(item[probability_key]),
            )
        )
    return extracted


def extract_generation_poems(row: pd.Series, method_id: str) -> list[dict[str, Any]]:
    if method_id == "poetry_vs5":
        extracted = [
            (poem, probability, "json_response")
            for poem, probability in extract_vs5_poems(row["response"])
        ]
    elif method_id == "poetry_ssot":
        poem, rule = extract_ssot_poem(row["response"])
        extracted = [(poem, None, rule)]
    else:
        extracted = [
            (strip_leading_poem_labels(row["response"]), None, "direct_response")
        ]

    records = []
    for response_idx, (poem, probability, rule) in enumerate(extracted, start=1):
        records.append(
            {
                "poem_id": create_stable_id(
                    "poem", method_id, row["generation_id"], response_idx
                ),
                "generation_id": row["generation_id"],
                "method_id": method_id,
                "topic_id": row["topic_id"],
                "topic": row["topic"],
                "hierarchy_paths": row["hierarchy_paths"],
                "thinking_level": row["thinking_level"],
                "source_sample_idx": int(row["sample_idx"]),
                "response_idx": response_idx,
                "verbalized_probability": probability,
                "response": poem,
                "extraction_rule": rule,
                "model": row["model"],
                "model_version": row["model_version"],
                "generation_config": row["generation_config"],
            }
        )
    return records


def prepare_method(
    method_id: str,
    config: dict[str, Any],
    results_root: Path,
) -> Path:
    method = config["methods"][method_id]
    source = results_root / "data" / method["input_filename"]
    generations = pd.read_csv(source, keep_default_na=False)
    records = []
    failures = []
    rows = tqdm(
        generations.iterrows(),
        total=len(generations),
        desc=f"Extracting {method_id}",
        unit="generation",
    )
    for _, row in rows:
        try:
            records.extend(extract_generation_poems(row, method_id))
        except (json.JSONDecodeError, KeyError, StopIteration, TypeError, ValueError) as error:
            failures.append(
                {
                    "generation_id": row["generation_id"],
                    "method_id": method_id,
                    "topic_id": row["topic_id"],
                    "topic": row["topic"],
                    "thinking_level": row["thinking_level"],
                    "source_sample_idx": int(row["sample_idx"]),
                    "error": str(error),
                }
            )
    expected = len(generations) * method["responses_per_generation"]
    failed_responses = len(failures) * method["responses_per_generation"]
    if len(records) + failed_responses != expected or any(
        not record["response"] for record in records
    ):
        raise ValueError(f"Unexpected extracted response shape for {method_id}")

    output_dir = results_root / "data" / method_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "responses.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    if failures:
        failure_path = output_dir / "extraction_failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=failures[0].keys())
            writer.writeheader()
            writer.writerows(failures)
        print(f"Recorded {len(failures)} extraction failures in {failure_path}")
    print(f"Wrote {len(records)} poems to {output_path}")
    return output_path


def load_embedding_responses(path: Path) -> pd.DataFrame:
    responses = pd.read_csv(path, keep_default_na=False)
    responses["response_id"] = responses["poem_id"]
    responses["response_sha256"] = responses["response"].map(compute_stable_hash)
    responses["response_input_kind"] = "poem"
    return responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract poems for one analysis method")
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analysis_config = load_analysis_config(Path(arguments.config))
    if arguments.method not in analysis_config["methods"]:
        raise ValueError(f"Unknown method: {arguments.method}")
    prepare_method(arguments.method, analysis_config, Path(arguments.results_root))
