from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lib.prepare_responses import CONTEXT_COLUMNS


def normalize_matrix(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    matrix = np.stack([embeddings[key] for key in responses["response_id"]])
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def compute_intra_prompt_scores(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    group_columns = [
        *[column for column in CONTEXT_COLUMNS if column in responses],
        "query_id",
        "query",
        "model",
        "config_id",
    ]
    records = []
    for keys, group in responses.groupby(group_columns, sort=False):
        matrix = normalize_matrix(group, embeddings)
        if len(matrix) < 2:
            continue
        row_indices, column_indices = np.triu_indices(len(matrix), k=1)
        scores = (matrix @ matrix.T)[row_indices, column_indices]
        records.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "n_responses": len(matrix),
                "n_pairs": len(scores),
                "mean_similarity": float(scores.mean()),
                "std_similarity": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            }
        )
    return pd.DataFrame(records)


def compute_inter_prompt_scores(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    group_columns = [
        *[column for column in CONTEXT_COLUMNS if column in responses],
        "query_id",
        "query",
        "config_id",
    ]
    records = []
    for keys, query_group in responses.groupby(group_columns, sort=False):
        matrices = {
            model: normalize_matrix(group, embeddings)
            for model, group in query_group.groupby("model", sort=True)
        }
        models = sorted(matrices)
        for index, model_1 in enumerate(models):
            for model_2 in models[index + 1 :]:
                scores = matrices[model_1] @ matrices[model_2].T
                records.append(
                    {
                        **dict(zip(group_columns, keys, strict=True)),
                        "model_1": model_1,
                        "model_2": model_2,
                        "n_responses_1": len(matrices[model_1]),
                        "n_responses_2": len(matrices[model_2]),
                        "n_pairs": scores.size,
                        "mean_similarity": float(scores.mean()),
                        "std_similarity": (
                            float(scores.std(ddof=1)) if scores.size > 1 else 0.0
                        ),
                    }
                )
    return pd.DataFrame(records)


def compute_random_baseline(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    n_pairs: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    matrix = normalize_matrix(responses, embeddings)
    query_ids = responses["query_id"].to_numpy()
    rng = np.random.default_rng(seed)
    records = []
    while len(records) < n_pairs:
        left, right = rng.integers(0, len(responses), size=2)
        if left == right or query_ids[left] == query_ids[right]:
            continue
        record = {
            "response_id_1": responses.iloc[left]["response_id"],
            "response_id_2": responses.iloc[right]["response_id"],
            "query_id_1": query_ids[left],
            "query_id_2": query_ids[right],
            "similarity": float(matrix[left] @ matrix[right]),
        }
        if "language_code" in responses:
            record["language_code"] = responses.iloc[left]["language_code"]
        records.append(record)
    return pd.DataFrame(records)


@dataclass(frozen=True)
class AnalysisPaths:
    normalized: Path
    intra: Path
    inter: Path
    baseline: Path
    summary: Path


def summarize_scores(
    intra: pd.DataFrame,
    inter: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    records = [
        ("intra_model", model, model, group["mean_similarity"])
        for model, group in intra.groupby("model", sort=True)
    ]
    records.extend(
        ("inter_model", model_1, model_2, group["mean_similarity"])
        for (model_1, model_2), group in inter.groupby(
            ["model_1", "model_2"], sort=True
        )
    )
    records.extend(
        [
            ("intra_model_overall", "all", "all", intra["mean_similarity"]),
            ("inter_model_overall", "all", "all", inter["mean_similarity"]),
            ("random_cross_prompt", "all", "all", baseline["similarity"]),
        ]
    )
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "model_1": model_1,
                "model_2": model_2,
                "n": len(values),
                "mean_similarity": values.mean(),
                "std_similarity": values.std(ddof=1),
                "min_similarity": values.min(),
                "max_similarity": values.max(),
            }
            for metric, model_1, model_2, values in records
        ]
    )
