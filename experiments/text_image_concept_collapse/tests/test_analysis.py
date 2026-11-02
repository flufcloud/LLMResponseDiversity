from pathlib import Path

import numpy as np
import pandas as pd

from experiments.text_image_concept_collapse.analyze import (
    compute_cross_modal_scores,
    create_similarity_matrix,
    summarize_scores,
)
from experiments.text_image_concept_collapse.embed_data import embed_vision_inputs
from lib.embed_images_locally import get_feature_tensor
from lib.store_multimodal_embeddings import MultimodalEmbeddingStore


class FakeEncoder:
    def encode_texts(self, values):
        assert values == ["first", "second"]
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class FakeModelOutput:
    def __init__(self, pooled):
        self.pooler_output = pooled


def test_get_feature_tensor_extracts_transformers_pooled_output() -> None:
    pooled = np.array([[1.0, 2.0]], dtype=np.float32)

    assert get_feature_tensor(FakeModelOutput(pooled)) is pooled


def test_embed_vision_inputs_reuses_matching_records(tmp_path: Path) -> None:
    records = pd.DataFrame(
        {
            "input_id": ["text--q1--001", "text--q1--002"],
            "response": ["first", "second"],
        }
    )
    settings = {
        "model": "test/model",
        "revision": "revision",
        "dtype": "float32",
        "dimensions": 2,
    }
    with MultimodalEmbeddingStore(tmp_path / "embeddings.sqlite") as store:
        first = embed_vision_inputs(
            records, "text", "response", store, FakeEncoder(), settings
        )
        second = embed_vision_inputs(
            records, "text", "response", store, FakeEncoder(), settings
        )

    assert np.array_equal(first["text--q1--001"], np.array([1.0, 0.0]))
    assert np.array_equal(second["text--q1--002"], np.array([0.0, 1.0]))


def test_pairwise_scores_matrix_and_summary() -> None:
    left = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "sample_idx": [1, 2],
            "input_id": ["t1", "t2"],
        }
    )
    right = pd.DataFrame(
        {
            "query_id": ["q1", "q1"],
            "image_sample_idx": [1, 2],
            "input_id": ["i1", "i2"],
        }
    )
    embeddings = {
        "t1": np.array([1.0, 0.0]),
        "t2": np.array([0.0, 1.0]),
        "i1": np.array([1.0, 0.0]),
        "i2": np.array([0.0, 1.0]),
    }
    scores = compute_cross_modal_scores(
        left,
        right,
        embeddings,
        embeddings,
        "sample_idx",
        "image_sample_idx",
    )
    matrix = create_similarity_matrix(scores, "sample_idx", "image_sample_idx")
    prompts = pd.DataFrame(
        {"query_id": ["q1"], "text_query": ["Text"], "image_query": ["Image"]}
    )
    summary = summarize_scores(scores, prompts)

    assert len(scores) == 4
    assert matrix.columns.tolist() == [
        "query_id",
        "sample_idx",
        "image_001",
        "image_002",
    ]
    assert summary.loc[0, "mean_pairwise_similarity"] == 0.5
