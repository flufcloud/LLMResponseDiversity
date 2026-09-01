import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.cross_lingual_embedding_validation.analyze import analyze_embeddings
from experiments.cross_lingual_embedding_validation.embed_data import embed_queries


class FakeEncoder:
    vectors = {
        "same": [1.0, 0.0, 0.0],
        "first query": [1.0, 0.0, 0.0],
        "second query": [1.0, 0.0, 0.0],
        "x": [1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [1.0, 0.0, 0.0],
    }

    def get_token_lengths(self, texts: list[str]) -> list[int]:
        return [1] * len(texts)

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def write_inputs(root: Path) -> SimpleNamespace:
    config = root / "embedding_config.json"
    config.write_text(
        json.dumps(
            {
                "model": "test/bge-m3",
                "revision": "test-revision",
                "dimensions": 3,
                "normalize_embeddings": True,
                "max_length": 128,
                "batch_size": 4,
                "device": "cpu",
            }
        )
    )
    translated = root / "translated_queries.csv"
    pd.DataFrame(
        [
            {"query_id": "q001", "lang": lang, "query": text}
            for lang, text in (("en", "first query"), ("fr", "same"), ("yo", "same"))
        ]
        + [
            {"query_id": "q002", "lang": lang, "query": text}
            for lang, text in (("en", "second query"), ("fr", "y"), ("yo", "z"))
        ]
    ).to_csv(translated, index=False)
    return SimpleNamespace(
        config=str(config),
        input=str(translated),
        embeddings=str(root / "embeddings.sqlite"),
        data_dir=str(root / "data"),
        tables_dir=str(root / "tables"),
        metadata=str(root / "metadata.json"),
        allow_model_download=False,
    )


def test_embeds_and_analyzes_language_pairs(tmp_path: Path) -> None:
    args = write_inputs(tmp_path)

    assert embed_queries(args, FakeEncoder()) == 6
    outputs = analyze_embeddings(args)

    assert {path.name for path in outputs.values()} == {
        "raw_multilingual_similarity.csv",
        "cross_lingual_mean_similarity_by_query.csv",
        "cross_lingual_mean_similarity_by_lang_pair.csv",
        "cross_lingual_mean_similarity_by_lang.csv",
    }
    assert outputs["raw"].parent == Path(args.data_dir)
    assert all(
        outputs[name].parent == Path(args.tables_dir)
        for name in ("query", "lang_pair", "lang")
    )
    raw = pd.read_csv(outputs["raw"])
    assert raw.columns.tolist() == ["query_id", "lang_pair", "similarity_score"]
    assert len(raw) == 6
    assert set(raw["lang_pair"]) == {"en_fr", "en_yo", "fr_yo"}

    by_query = pd.read_csv(outputs["query"])
    assert by_query.columns.tolist() == ["query", "mean_pairwise_similarity"]
    assert by_query["query"].tolist() == ["first query", "second query"]

    by_pair = pd.read_csv(outputs["lang_pair"])
    assert by_pair.columns.tolist() == ["lang_pair", "mean_pairwise_similarity"]
    assert by_pair["mean_pairwise_similarity"].is_monotonic_decreasing
    assert by_pair.iloc[0]["lang_pair"] == "en_yo"

    by_language = pd.read_csv(outputs["lang"])
    assert by_language.columns.tolist() == ["lang", "mean_pairwise_similarity"]
    assert by_language["mean_pairwise_similarity"].is_monotonic_decreasing
    assert by_language.iloc[-1]["lang"] == "fr"
    assert json.loads(Path(args.metadata).read_text()) == {
        "experiment_id": "cross_lingual_embedding_validation",
        "embedding_model": "test/bge-m3",
    }
