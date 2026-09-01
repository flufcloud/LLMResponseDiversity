from __future__ import annotations

from pathlib import Path

import pandas as pd


LANGUAGES = {"en": "English", "yo": "Yoruba", "te": "Telugu"}
EXPERIMENT_ID = "en_te_yo_30_query"
RESULTS_PATH = Path("results/experiments") / EXPERIMENT_ID
DATA_PATH = RESULTS_PATH / "data"
TABLES_PATH = RESULTS_PATH / "tables"
PLOTS_PATH = RESULTS_PATH / "plots"
METADATA_PATH = RESULTS_PATH / "metadata.json"
GENERATION_CONFIG_PATH = Path(
    "experiments/hivemind_reproduction/en_te_yo_30_query/generation_config.json"
)
GENERATIONS_PATH = DATA_PATH / "generations.csv"
GENERATION_METADATA_PATH = DATA_PATH / "generations_metadata.csv"
EMBEDDINGS_PATH = DATA_PATH / "embeddings.sqlite"

GENERATION_COLUMNS = [
    "generation_id",
    "query_id",
    "query_tag",
    "sample_idx",
    "base_query",
    "base_language_code",
    "base_language",
    "translated_query",
    "translated_language_code",
    "translated_language",
    "query_translation_provider",
    "query_translation_model",
    "model_response",
    "response_translation",
    "response_translation_language_code",
    "response_translation_provider",
    "provider",
    "model_version",
    "generation_config",
    "system_instruction",
]

GENERATION_METADATA_COLUMNS = [
    "generation_id",
    "experiment_id",
    "request_key",
    "batch_job_id",
    "request_status",
    "source_dataset",
    "source_row_index",
    "selection_seed",
    "source_artifact",
    "submitted_at",
    "completed_at",
]


def load_generations(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)[GENERATION_COLUMNS]


def load_generation_metadata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)[GENERATION_METADATA_COLUMNS]


def get_analysis_responses(
    generations: pd.DataFrame,
    response_column: str = "model_response",
) -> pd.DataFrame:
    selected = generations[generations[response_column].notna()].copy()
    selected["query"] = selected["translated_query"].fillna(selected["base_query"])
    selected["language_code"] = selected["translated_language_code"].fillna(
        selected["base_language_code"]
    )
    selected["language"] = selected["translated_language"].fillna(
        selected["base_language"]
    )
    return selected.assign(
        response=selected[response_column],
        model=selected["model_version"],
        response_input_kind=response_column,
    )[
        [
            "generation_id",
            "query_id",
            "query_tag",
            "base_query",
            "query",
            "language_code",
            "language",
            "response",
            "model",
            "generation_config",
            "sample_idx",
            "response_input_kind",
        ]
    ]


def select_multilingual_queries(
    responses: pd.DataFrame,
    languages: list[str],
) -> pd.DataFrame:
    coverage = responses.groupby("query_id")["language_code"].agg(set)
    query_ids = coverage[coverage.map(lambda found: set(languages).issubset(found))].index
    return responses[
        responses["query_id"].isin(query_ids) & responses["language_code"].isin(languages)
    ].copy()


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame[columns].to_csv(temporary, index=False)
    temporary.replace(path)
