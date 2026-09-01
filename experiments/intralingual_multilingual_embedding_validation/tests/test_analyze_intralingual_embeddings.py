import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.intralingual_multilingual_embedding_validation.analyze import (
    analyze_embeddings,
)
from experiments.intralingual_multilingual_embedding_validation.embed_data import embed_mrpc


class FakeEncoder:
    vectors = {
        "same": [1.0, 0.0, 0.0],
        "close": [0.8, 0.6, 0.0],
        "different": [0.0, 1.0, 0.0],
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
    translated = root / "mrpc_translated.csv"
    pd.DataFrame(
        [
            {
                "index": 0,
                "mrpc_sentence_id": 10,
                "target_lang_identifier": "en",
                "translated_base": "same",
                "translated_paraphrase": "same",
            },
            {
                "index": 1,
                "mrpc_sentence_id": 10,
                "target_lang_identifier": "fr",
                "translated_base": "same",
                "translated_paraphrase": "close",
            },
            {
                "index": 2,
                "mrpc_sentence_id": 12,
                "target_lang_identifier": "en",
                "translated_base": "same",
                "translated_paraphrase": "same",
            },
            {
                "index": 3,
                "mrpc_sentence_id": 12,
                "target_lang_identifier": "fr",
                "translated_base": "same",
                "translated_paraphrase": "different",
            },
        ]
    ).to_csv(translated, index=False)
    return SimpleNamespace(
        config=str(config),
        input=str(translated),
        embeddings=str(root / "embeddings.sqlite"),
        tables_dir=str(root / "tables"),
        metadata=str(root / "metadata.json"),
        allow_model_download=False,
    )


def test_embeds_and_analyzes_intralingual_pairs(tmp_path: Path) -> None:
    args = write_inputs(tmp_path)

    assert embed_mrpc(args, FakeEncoder()) == 8
    outputs = analyze_embeddings(args)

    raw = pd.read_csv(outputs["raw"])
    assert raw.columns.tolist() == ["index", "language_identifier", "similarity_score"]
    assert raw["index"].tolist() == [0, 1, 2, 3]
    assert np.allclose(raw["similarity_score"], [1.0, 0.8, 1.0, 0.0])

    mean = pd.read_csv(outputs["mean"])
    assert mean.columns.tolist() == ["language_identifier", "mean_similarity_score"]
    assert mean["language_identifier"].tolist() == ["en", "fr"]
    assert np.allclose(mean["mean_similarity_score"], [1.0, 0.4])

    by_query = pd.read_csv(outputs["query"])
    assert by_query.columns.tolist() == ["mrpc_sentence_id", "mean_similarity_score"]
    assert by_query["mrpc_sentence_id"].tolist() == [10, 12]
    assert np.allclose(by_query["mean_similarity_score"], [0.9, 0.5])
    assert json.loads(Path(args.metadata).read_text()) == {
        "experiment_id": "intralingual_multilingual_embedding_validation",
        "embedding_model": "test/bge-m3",
    }
