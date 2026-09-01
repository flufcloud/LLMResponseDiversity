from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    DATA_PATH,
    EMBEDDINGS_PATH,
    EXPERIMENT_ID,
    GENERATION_CONFIG_PATH,
    GENERATIONS_PATH,
    LANGUAGES,
    METADATA_PATH,
    TABLES_PATH,
)
from lib.embed_with_bge import (
    BGE_DIMENSIONS,
    BGE_MAX_LENGTH,
    BGE_MODEL,
    get_bge_fingerprint,
)
from lib.store_embeddings import EmbeddingCache, load_response_embeddings
from experiments.hivemind_reproduction.en_te_yo_30_query.embed_data import (
    load_multilingual_responses,
    load_translated_responses,
    prepare_responses,
)
from lib.compute_similarity import (
    AnalysisPaths,
    compute_inter_prompt_scores,
    compute_intra_prompt_scores,
    compute_random_baseline,
    normalize_matrix,
    summarize_scores,
)
from lib.write_experiment_metadata import write_experiment_metadata


@dataclass(frozen=True)
class CrossLanguagePaths:
    scores: Path
    baseline: Path
    summary: Path


def get_language_analysis_paths(data_dir: Path, tables_dir: Path) -> AnalysisPaths:
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return AnalysisPaths(
        normalized=data_dir / "responses_normalized.csv",
        intra=tables_dir / "intra_model_prompt_scores.csv",
        inter=tables_dir / "inter_model_prompt_scores.csv",
        baseline=data_dir / "random_pair_baseline.csv",
        summary=tables_dir / "similarity_summary.csv",
    )


def compute_cross_language_scores(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    records = []
    group_columns = ["query_id", "base_query", "model", "config_id"]
    for keys, prompt_group in responses.groupby(group_columns, sort=False):
        language_groups = {
            code: group for code, group in prompt_group.groupby("language_code", sort=True)
        }
        for language_1, language_2 in combinations(sorted(language_groups), 2):
            group_1 = language_groups[language_1]
            group_2 = language_groups[language_2]
            scores = normalize_matrix(group_1, embeddings) @ normalize_matrix(
                group_2, embeddings
            ).T
            records.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "language_code_1": language_1,
                    "language_1": LANGUAGES[language_1],
                    "query_1": group_1["query"].iat[0],
                    "language_code_2": language_2,
                    "language_2": LANGUAGES[language_2],
                    "query_2": group_2["query"].iat[0],
                    "n_responses_1": len(group_1),
                    "n_responses_2": len(group_2),
                    "n_pairs": scores.size,
                    "mean_similarity": float(scores.mean()),
                    "std_similarity": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
                }
            )
    return pd.DataFrame(records)


def compute_cross_language_baseline(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    n_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for model, model_group in responses.groupby("model", sort=True):
        language_groups = {
            code: group.reset_index(drop=True)
            for code, group in model_group.groupby("language_code", sort=True)
        }
        for language_1, language_2 in combinations(sorted(language_groups), 2):
            group_1 = language_groups[language_1]
            group_2 = language_groups[language_2]
            matrix_1 = normalize_matrix(group_1, embeddings)
            matrix_2 = normalize_matrix(group_2, embeddings)
            collected = 0
            while collected < n_pairs:
                draw_count = min(2 * (n_pairs - collected), n_pairs)
                indices_1 = rng.integers(0, len(group_1), size=draw_count)
                indices_2 = rng.integers(0, len(group_2), size=draw_count)
                valid = group_1.iloc[indices_1]["query_id"].to_numpy() != group_2.iloc[
                    indices_2
                ]["query_id"].to_numpy()
                for index_1, index_2 in zip(indices_1[valid], indices_2[valid], strict=True):
                    records.append(
                        {
                            "model": model,
                            "language_code_1": language_1,
                            "language_code_2": language_2,
                            "query_id_1": group_1.iloc[index_1]["query_id"],
                            "query_id_2": group_2.iloc[index_2]["query_id"],
                            "response_id_1": group_1.iloc[index_1]["response_id"],
                            "response_id_2": group_2.iloc[index_2]["response_id"],
                            "similarity": float(matrix_1[index_1] @ matrix_2[index_2]),
                        }
                    )
                    collected += 1
                    if collected == n_pairs:
                        break
    return pd.DataFrame(records)


def summarize_cross_language(
    scores: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    score_groups = scores.groupby(
        ["model", "language_code_1", "language_code_2"], sort=True
    )
    for keys, group in score_groups:
        model, language_1, language_2 = keys
        values = group["mean_similarity"]
        random_values = baseline.loc[
            (baseline["model"] == model)
            & (baseline["language_code_1"] == language_1)
            & (baseline["language_code_2"] == language_2),
            "similarity",
        ]
        for metric, metric_values in (
            ("matched_prompt", values),
            ("random_cross_prompt", random_values),
        ):
            records.append(
                {
                    "metric": metric,
                    "model": model,
                    "language_code_1": language_1,
                    "language_code_2": language_2,
                    "n": len(metric_values),
                    "mean_similarity": metric_values.mean(),
                    "std_similarity": metric_values.std(ddof=1),
                    "min_similarity": metric_values.min(),
                    "max_similarity": metric_values.max(),
                }
            )
    return pd.DataFrame(records)


def write_language_analysis(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    language_code: str,
    args: argparse.Namespace,
) -> AnalysisPaths:
    subset = responses[responses["language_code"] == language_code].copy()
    paths = get_language_analysis_paths(
        Path(args.data_dir) / language_code,
        Path(args.tables_dir) / language_code,
    )
    intra = compute_intra_prompt_scores(subset, embeddings)
    inter = compute_inter_prompt_scores(subset, embeddings)
    baseline = compute_random_baseline(subset, embeddings, args.baseline_pairs, args.seed)
    subset.to_csv(paths.normalized, index=False)
    intra.to_csv(paths.intra, index=False)
    inter.to_csv(paths.inter, index=False)
    baseline.to_csv(paths.baseline, index=False)
    summarize_scores(intra, inter, baseline).to_csv(paths.summary, index=False)
    return paths


def write_cross_language_analysis(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    args: argparse.Namespace,
    output_name: str = "cross_language",
) -> CrossLanguagePaths:
    data_dir = Path(args.data_dir) / output_name
    tables_dir = Path(args.tables_dir) / output_name
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = CrossLanguagePaths(
        scores=tables_dir / "cross_language_prompt_scores.csv",
        baseline=data_dir / "cross_language_random_baseline.csv",
        summary=tables_dir / "cross_language_summary.csv",
    )
    scores = compute_cross_language_scores(responses, embeddings)
    baseline = compute_cross_language_baseline(
        responses, embeddings, args.baseline_pairs, args.seed
    )
    scores.to_csv(paths.scores, index=False)
    baseline.to_csv(paths.baseline, index=False)
    summarize_cross_language(scores, baseline).to_csv(paths.summary, index=False)
    return paths


def prepare_analysis_inputs(
    raw: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    selected = prepare_responses(raw, args)
    fingerprint = get_bge_fingerprint(max_length=args.max_length)
    with EmbeddingCache(Path(args.cache)) as cache:
        embeddings = load_response_embeddings(
            selected,
            cache,
            fingerprint,
            BGE_DIMENSIONS,
        )
    generation_config = json.loads(GENERATION_CONFIG_PATH.read_text())
    write_experiment_metadata(
        Path(args.metadata),
        EXPERIMENT_ID,
        BGE_MODEL,
        generation_config["generation_config"]["max_output_tokens"],
    )
    return selected, embeddings


def run_analysis(
    args: argparse.Namespace,
) -> tuple[dict[str, AnalysisPaths], CrossLanguagePaths]:
    raw = load_multilingual_responses(Path(args.input))
    selected, embeddings = prepare_analysis_inputs(raw, args)
    language_paths = {
        code: write_language_analysis(selected, embeddings, code, args)
        for code in args.languages
    }
    cross_paths = write_cross_language_analysis(selected, embeddings, args)
    return language_paths, cross_paths


def run_translated_analysis(args: argparse.Namespace) -> CrossLanguagePaths:
    raw = load_translated_responses(Path(args.input))
    selected, embeddings = prepare_analysis_inputs(raw, args)
    paths = write_cross_language_analysis(
        selected,
        embeddings,
        args,
        output_name="translated_cross_language",
    )
    data_dir = Path(args.data_dir) / "translated_cross_language"
    tables_dir = paths.scores.parent
    within = pd.concat(
        [
            compute_intra_prompt_scores(
                selected[selected["language_code"] == code], embeddings
            )
            for code in args.languages
        ],
        ignore_index=True,
    )
    selected.to_csv(data_dir / "responses_normalized.csv", index=False)
    within.to_csv(tables_dir / "within_language_prompt_scores.csv", index=False)
    return paths


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        default=str(GENERATIONS_PATH),
    )
    parser.add_argument("--data-dir", default=str(DATA_PATH))
    parser.add_argument("--tables-dir", default=str(TABLES_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--cache", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--max-length", type=int, default=BGE_MAX_LENGTH)
    parser.add_argument("--config-id")
    parser.add_argument("--include-translated", action="store_true")
    parser.add_argument("--baseline-pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BGE-M3 Hivemind similarities by language"
    )
    add_analysis_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    language_outputs, cross_output = run_analysis(arguments)
    for code, paths in language_outputs.items():
        print(f"{code}: {paths.intra.parent}")
    print(f"cross-language: {cross_output.scores.parent}")
    if arguments.include_translated:
        translated_output = run_translated_analysis(arguments)
        print(f"translated cross-language: {translated_output.scores.parent}")
