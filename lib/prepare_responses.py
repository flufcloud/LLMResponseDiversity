from __future__ import annotations

import pandas as pd

from lib.compute_identifiers import compute_stable_hash


CONTEXT_COLUMNS = ["base_query", "language_code", "language", "query_tag"]


def normalize_responses(responses: pd.DataFrame) -> pd.DataFrame:
    responses = responses.loc[:, ~responses.columns.str.match(r"^Unnamed|^$")].copy()
    for column in ("query", "response", "model"):
        responses[column] = responses[column].fillna("").astype(str)
    responses["sample_idx"] = responses["sample_idx"].astype(int)
    if "generation_config" not in responses:
        responses["generation_config"] = "unknown"
    responses["generation_config"] = (
        responses["generation_config"].fillna("unknown").astype(str)
    )
    if "query_id" not in responses:
        responses["query_id"] = responses["query"].map(
            lambda text: compute_stable_hash(text)[:16]
        )
    responses["query_id"] = responses["query_id"].fillna("").astype(str)
    responses["response_sha256"] = responses["response"].map(compute_stable_hash)
    responses["config_id"] = responses["generation_config"].map(
        lambda value: compute_stable_hash(value)[:16]
    )
    if "generation_id" in responses:
        responses["response_id"] = responses.apply(
            lambda row: compute_stable_hash(row["generation_id"], row["response_sha256"])[:24],
            axis=1,
        )
    else:
        id_columns = [
            "query_id",
            "model",
            "config_id",
            "sample_idx",
            "response_sha256",
        ]
        responses["response_id"] = responses[id_columns].apply(
            lambda row: compute_stable_hash(*row)[:24], axis=1
        )
    columns = [
        "response_id",
        *[
            column
            for column in ("generation_id", "response_input_kind")
            if column in responses
        ],
        "query_id",
        *[column for column in CONTEXT_COLUMNS if column in responses],
        "query",
        "response",
        "response_sha256",
        "model",
        "config_id",
        "generation_config",
        "sample_idx",
    ]
    return responses[columns].sort_values(
        ["query_id", "model", "config_id", "sample_idx"], ignore_index=True
    )
