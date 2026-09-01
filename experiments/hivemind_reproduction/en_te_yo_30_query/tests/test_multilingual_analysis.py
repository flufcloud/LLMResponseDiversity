import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lib.embed_with_bge import (
    BGE_DIMENSIONS,
    fetch_bge_embeddings,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    GENERATION_COLUMNS,
)
from lib.store_embeddings import EmbeddingCache
from lib.prepare_responses import (
    normalize_responses,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.analyze import (
    compute_cross_language_baseline,
    compute_cross_language_scores,
    load_multilingual_responses,
    run_analysis,
    run_translated_analysis,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.create_plots import (
    plot_all_analyses,
)
from experiments.hivemind_reproduction.en_te_yo_30_query.embed_data import embed_data


LANGUAGES = {"en": "English", "yo": "Yoruba", "te": "Telugu"}


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def get_token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 2 for text in texts]

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), BGE_DIMENSIONS), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, hash(text) % BGE_DIMENSIONS] = 1
        return vectors


def write_multilingual_fixture(root: Path) -> Path:
    generations = []
    for prompt_number in (1, 2):
        query_id = f"q{prompt_number:03d}"
        base_query = f"query-{prompt_number}-en"
        for language_code, language in LANGUAGES.items():
            query = f"query-{prompt_number}-{language_code}"
            for model in ("model-a", "model-b"):
                for sample_idx in (1, 2):
                    translated = language_code != "en"
                    response = f"response-{prompt_number}-{language_code}-{model}-{sample_idx}"
                    generations.append(
                        {
                            "generation_id": (
                                f"g-{prompt_number}-{language_code}-{model}-{sample_idx}"
                            ),
                            "query_id": query_id,
                            "query_tag": f"query-{prompt_number}",
                            "sample_idx": sample_idx,
                            "base_query": base_query,
                            "base_language_code": "en",
                            "base_language": "English",
                            "translated_query": query if translated else None,
                            "translated_language_code": (
                                language_code if translated else None
                            ),
                            "translated_language": language if translated else None,
                            "query_translation_provider": (
                                "test-translator" if translated else None
                            ),
                            "query_translation_model": (
                                "test-model" if translated else None
                            ),
                            "model_response": response,
                            "response_translation": (
                                f"English translation of {response}" if translated else None
                            ),
                            "response_translation_language_code": (
                                "en" if translated else None
                            ),
                            "response_translation_provider": (
                                "local-nllb-200-distilled-600m" if translated else None
                            ),
                            "provider": "test-provider",
                            "model_version": model,
                            "generation_config": "config",
                            "system_instruction": None,
                        }
                    )
    generation_path = root / "generations.csv"
    pd.DataFrame(generations)[GENERATION_COLUMNS].to_csv(generation_path, index=False)
    return generation_path


def test_loads_canonical_generations(tmp_path: Path) -> None:
    generation_path = write_multilingual_fixture(tmp_path)
    loaded = load_multilingual_responses(generation_path)

    assert len(loaded) == 24
    assert set(loaded["language_code"]) == set(LANGUAGES)


def test_cross_language_scores_hold_prompt_and_model_constant(tmp_path: Path) -> None:
    generation_path = write_multilingual_fixture(tmp_path)
    responses = normalize_responses(load_multilingual_responses(generation_path))
    embeddings = {
        row.response_id: np.array([1.0, 0.0], dtype=np.float32)
        for row in responses.itertuples(index=False)
    }

    scores = compute_cross_language_scores(responses, embeddings)
    baseline = compute_cross_language_baseline(responses, embeddings, n_pairs=5, seed=0)

    assert len(scores) == 12
    assert scores["n_pairs"].eq(4).all()
    assert scores.groupby(["query_id", "model"]).size().eq(3).all()
    assert len(baseline) == 30
    assert baseline["query_id_1"].ne(baseline["query_id_2"]).all()


def test_bge_cache_avoids_reencoding(tmp_path: Path) -> None:
    generation_path = write_multilingual_fixture(tmp_path)
    responses = normalize_responses(load_multilingual_responses(generation_path))
    encoder = FakeEncoder()

    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        first, fingerprint = fetch_bge_embeddings(responses, cache, encoder, batch_size=8)
    first_call_count = encoder.calls
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        second, repeated_fingerprint = fetch_bge_embeddings(responses, cache, encoder, batch_size=8)

    assert first_call_count > 0
    assert encoder.calls == first_call_count
    assert first.keys() == second.keys()
    assert fingerprint == repeated_fingerprint


def test_bge_rejects_overlength_responses(tmp_path: Path) -> None:
    class OverlengthEncoder(FakeEncoder):
        def get_token_lengths(self, texts: list[str]) -> list[int]:
            return [8193 for _ in texts]

    generation_path = write_multilingual_fixture(tmp_path)
    responses = normalize_responses(load_multilingual_responses(generation_path))
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises(ValueError, match="exceeds 8192 tokens"):
            fetch_bge_embeddings(responses, cache, OverlengthEncoder(), batch_size=8)


def test_full_analysis_and_plot_outputs(tmp_path: Path) -> None:
    generation_path = write_multilingual_fixture(tmp_path)
    data_dir = tmp_path / "data"
    tables_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    cache_path = tmp_path / "cache.sqlite"
    args = SimpleNamespace(
        input=str(generation_path),
        data_dir=str(data_dir),
        tables_dir=str(tables_dir),
        metadata=str(tmp_path / "metadata.json"),
        cache=str(cache_path),
        languages=list(LANGUAGES),
        device="mps",
        batch_size=8,
        max_length=8192,
        config_id=None,
        baseline_pairs=5,
        seed=0,
        allow_model_download=False,
        include_translated=True,
    )

    embed_data(args, FakeEncoder())
    language_paths, cross_paths = run_analysis(args)

    assert json.loads(Path(args.metadata).read_text()) == {
        "experiment_id": "en_te_yo_30_query",
        "embedding_model": "BAAI/bge-m3",
        "max_token_length": 2048,
    }
    assert set(language_paths) == set(LANGUAGES)
    assert language_paths["en"].normalized.parent == data_dir / "en"
    assert language_paths["en"].intra.parent == tables_dir / "en"
    assert cross_paths.baseline.parent == data_dir / "cross_language"
    assert cross_paths.scores.parent == tables_dir / "cross_language"
    assert pd.read_csv(cross_paths.scores)["n_pairs"].eq(4).all()
    translated_paths = run_translated_analysis(args)

    assert pd.read_csv(translated_paths.scores)["n_pairs"].eq(4).all()
    assert (translated_paths.scores.parent / "within_language_prompt_scores.csv").exists()
    plot_all_analyses(
        SimpleNamespace(
            data_dir=str(data_dir),
            tables_dir=str(tables_dir),
            output_dir=str(figure_dir),
            cache=str(cache_path),
            metadata=str(tmp_path / "metadata.json"),
            languages=list(LANGUAGES),
            query_id="q001",
            seed=0,
        )
    )
    for language_code in LANGUAGES:
        assert len(list((figure_dir / language_code).glob("*.png"))) == 6
    assert len(list((figure_dir / "cross_language").glob("*.png"))) == 6
    assert len(list((figure_dir / "translated_cross_language").glob("*.png"))) == 6
    assert not list(figure_dir.rglob("*.pdf"))
