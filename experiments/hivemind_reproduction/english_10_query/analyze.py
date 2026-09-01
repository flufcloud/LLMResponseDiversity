from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.hivemind_reproduction.english_10_query.load_generation_records import (
    DATA_PATH,
    EMBEDDINGS_PATH,
    EXPERIMENT_ID,
    GENERATION_CONFIG_PATH,
    GENERATIONS_PATH,
    METADATA_PATH,
    TABLES_PATH,
)
from experiments.hivemind_reproduction.english_10_query.embed_data import (
    prepare_responses,
)
from lib.store_embeddings import EmbeddingCache, load_response_embeddings
from lib.embed_with_openai import (
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
)
from lib.compute_similarity import (
    AnalysisPaths,
    compute_inter_prompt_scores,
    compute_intra_prompt_scores,
    compute_random_baseline,
    summarize_scores,
)
from lib.write_experiment_metadata import write_experiment_metadata


def run_analysis(args: argparse.Namespace) -> AnalysisPaths:
    data_dir = Path(args.data_dir)
    tables_dir = Path(args.tables_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    normalized = prepare_responses(Path(args.input), args.config_id)

    paths = AnalysisPaths(
        normalized=data_dir / "responses_normalized.csv",
        intra=tables_dir / "intra_model_prompt_scores.csv",
        inter=tables_dir / "inter_model_prompt_scores.csv",
        baseline=data_dir / "random_pair_baseline.csv",
        summary=tables_dir / "similarity_summary.csv",
    )
    normalized.to_csv(paths.normalized, index=False)
    with EmbeddingCache(Path(args.cache)) as cache:
        embeddings = load_response_embeddings(
            normalized,
            cache,
            args.embedding_model,
            args.dimensions,
        )
    intra = compute_intra_prompt_scores(normalized, embeddings)
    inter = compute_inter_prompt_scores(normalized, embeddings)
    baseline = compute_random_baseline(normalized, embeddings, args.baseline_pairs, args.seed)
    intra.to_csv(paths.intra, index=False)
    inter.to_csv(paths.inter, index=False)
    baseline.to_csv(paths.baseline, index=False)
    summarize_scores(intra, inter, baseline).to_csv(paths.summary, index=False)
    generation_config = json.loads(GENERATION_CONFIG_PATH.read_text())
    write_experiment_metadata(
        Path(args.metadata),
        EXPERIMENT_ID,
        args.embedding_model,
        generation_config["generation_config"]["max_output_tokens"],
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the 10-query English experiment")
    parser.add_argument("--input", default=str(GENERATIONS_PATH))
    parser.add_argument("--data-dir", default=str(DATA_PATH))
    parser.add_argument("--tables-dir", default=str(TABLES_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--cache", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--embedding-model", default=OPENAI_EMBEDDING_MODEL)
    parser.add_argument("--dimensions", type=int, default=OPENAI_EMBEDDING_DIMENSIONS)
    parser.add_argument("--config-id")
    parser.add_argument("--baseline-pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    output_paths = run_analysis(parse_args())
    for output_path in output_paths.__dict__.values():
        print(output_path)
