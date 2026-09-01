from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.store_embeddings import (
    EmbeddingCache,
    compute_embedding_key,
    load_response_embeddings,
)
from lib.embed_with_openai import (
    fetch_openai_embeddings,
)
from lib.prepare_responses import normalize_responses
from lib.compute_similarity import (
    compute_inter_prompt_scores,
    compute_intra_prompt_scores,
)


def make_response_frame() -> pd.DataFrame:
    return normalize_responses(
        pd.DataFrame(
            [
                {"query": "q1", "response": "a", "model": "m1", "sample_idx": 1},
                {"query": "q1", "response": "b", "model": "m1", "sample_idx": 2},
                {"query": "q1", "response": "c", "model": "m2", "sample_idx": 1},
                {"query": "q1", "response": "d", "model": "m2", "sample_idx": 2},
                {"query": "q2", "response": "e", "model": "m1", "sample_idx": 1},
                {"query": "q2", "response": "f", "model": "m1", "sample_idx": 2},
            ]
        )
    )


def test_normalizes_responses() -> None:
    responses = make_response_frame()

    assert responses["response_id"].is_unique
    assert responses["query_id"].nunique() == 2


def test_intra_and_inter_similarity_formulas() -> None:
    responses = make_response_frame()
    vectors_by_text = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
        "c": np.array([1.0, 0.0], dtype=np.float32),
        "d": np.array([1.0, 0.0], dtype=np.float32),
        "e": np.array([1.0, 1.0], dtype=np.float32),
        "f": np.array([1.0, 1.0], dtype=np.float32),
    }
    embeddings = {
        row.response_id: vectors_by_text[row.response] for row in responses.itertuples(index=False)
    }

    intra = compute_intra_prompt_scores(responses, embeddings)
    inter = compute_inter_prompt_scores(responses, embeddings)

    q1_m1 = intra[(intra["query"] == "q1") & (intra["model"] == "m1")].iloc[0]
    q1_m2 = intra[(intra["query"] == "q1") & (intra["model"] == "m2")].iloc[0]
    assert q1_m1["mean_similarity"] == pytest.approx(0)
    assert q1_m2["mean_similarity"] == pytest.approx(1)
    assert inter.iloc[0]["mean_similarity"] == pytest.approx(0.5)
    assert inter.iloc[0]["n_pairs"] == 4


class FakeEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, model: str, dimensions: int, input: list[str]) -> object:
        self.calls += 1
        data = []
        for index, _ in enumerate(input):
            item = type("Embedding", (), {"embedding": [1.0, 0.0], "index": index})()
            data.append(item)
        return type("Response", (), {"data": data})()


class FakeClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


def test_embedding_cache_avoids_repeated_api_calls(tmp_path: Path) -> None:
    responses = make_response_frame().iloc[:2]
    client = FakeClient()
    cache_path = tmp_path / "embeddings.sqlite"

    with EmbeddingCache(cache_path) as cache:
        first = fetch_openai_embeddings(responses, cache, client, dimensions=2)
    with EmbeddingCache(cache_path) as cache:
        second = fetch_openai_embeddings(responses, cache, client, dimensions=2)

    assert client.embeddings.calls == 1
    assert first.keys() == second.keys()
    assert compute_embedding_key("a", "text-embedding-3-small", 2)


def test_analysis_requires_embedding_stage(tmp_path: Path) -> None:
    responses = make_response_frame().iloc[:2]

    with EmbeddingCache(tmp_path / "embeddings.sqlite") as cache:
        with pytest.raises(ValueError, match="run embed_data.py first"):
            load_response_embeddings(
                responses,
                cache,
                "text-embedding-3-small",
                2,
            )
