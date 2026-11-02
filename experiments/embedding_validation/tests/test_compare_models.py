import json
from pathlib import Path

import pandas as pd

from experiments.embedding_validation.compare_models import write_comparison_tables


def write_model_result(
    root: Path,
    validation_name: str,
    result_name: str,
    model: str,
    score: float,
) -> None:
    model_dir = root / validation_name / "models" / result_name
    tables_dir = model_dir / "tables"
    tables_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        json.dumps({"experiment_id": validation_name, "embedding_model": model})
    )
    if validation_name == "cross_lingual":
        tables = {
            "cross_lingual_mean_similarity_by_query.csv": pd.DataFrame(
                [{"query": "query", "mean_pairwise_similarity": score}]
            ),
            "cross_lingual_mean_similarity_by_lang_pair.csv": pd.DataFrame(
                [{"lang_pair": "en_fr", "mean_pairwise_similarity": score}]
            ),
            "cross_lingual_mean_similarity_by_lang.csv": pd.DataFrame(
                [{"lang": "en", "mean_pairwise_similarity": score}]
            ),
        }
    else:
        tables = {
            "intralingual_embedding_scores_mean.csv": pd.DataFrame(
                [{"language_identifier": "en", "mean_similarity_score": score}]
            ),
            "intralingual_embedding_scores_mean_by_query.csv": pd.DataFrame(
                [{"mrpc_sentence_id": 1, "mean_similarity_score": score}]
            ),
        }
    for name, frame in tables.items():
        frame.to_csv(tables_dir / name, index=False)


def test_combines_each_models_mean_tables(tmp_path: Path) -> None:
    for validation_name in ("cross_lingual", "intralingual"):
        write_model_result(
            tmp_path, validation_name, "bge_m3", "BAAI/bge-m3", 0.8
        )
        write_model_result(
            tmp_path, validation_name, "harrier", "microsoft/harrier", 0.9
        )

    outputs = write_comparison_tables(tmp_path)

    assert len(outputs) == 5
    for output in outputs.values():
        combined = pd.read_csv(output)
        assert combined.columns[-2:].tolist() == [
            "bge_m3_similarity",
            "harrier_similarity",
        ]
        assert combined[["bge_m3_similarity", "harrier_similarity"]].iloc[
            0
        ].tolist() == [0.8, 0.9]
