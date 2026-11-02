from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glasgow_concreteness_regression.embed_data import embed_words
from experiments.glasgow_concreteness_regression.prepare_data import (
    load_regression_settings,
    prepare_training_rows,
)
from experiments.glasgow_concreteness_regression.predict_norms import (
    predict_words,
)
from experiments.glasgow_concreteness_regression.train_mlp import train_mlp
from experiments.glasgow_concreteness_regression.train_mlp_ensemble import (
    train_mlp_ensemble,
)
from experiments.glasgow_concreteness_regression.train_regressor import train_regressor


class FakeEncoder:
    def get_token_lengths(self, texts: list[str]) -> list[int]:
        return [1] * len(texts)

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        del batch_size
        indices = np.array([int(text.removeprefix("word")) + 1 for text in texts])
        return np.column_stack((indices, indices % 7)).astype(np.float32)


def write_fixture(tmp_path: Path) -> Path:
    words = [f"word{index}" for index in range(100)] + ["word1 (sense)"]
    scores = [1 + 6 * index / 99 for index in range(100)] + [2.0]
    source_path = tmp_path / "glasgow.csv"
    targets = [
        "arousal_mean",
        "valence_mean",
        "dominance_mean",
        "concrete_mean",
        "imagine_mean",
        "familiar_mean",
        "aoa_mean",
        "size_mean",
        "gender_mean",
    ]
    source = {"word_english": words}
    source.update(
        {
            target: np.roll(scores, shift) for shift, target in enumerate(targets)
        }
    )
    pd.DataFrame(source).to_csv(source_path, index=False)
    config = {
        "experiment_id": "test_concreteness",
        "source_csv": str(source_path),
        "word_column": "word_english",
        "target_columns": targets,
        "exclude_parenthetical_senses": True,
        "train_fraction": 0.8,
        "dev_fraction": 0.1,
        "test_fraction": 0.1,
        "split_seed": 0,
        "stratify_bins": 5,
        "ridge_alphas": [0.01, 0.1, 1.0],
        "result_name": "fake",
        "model": "fake/model",
        "revision": "test-revision",
        "dimensions": 2,
        "normalize_embeddings": False,
        "max_length": 8,
        "batch_size": 16,
        "checkpoint_batch_size": 100,
        "device": "cpu",
        "mlp": {
            "hidden_dimension": 12,
            "dropout": 0.2,
            "learning_rate": 0.01,
            "weight_decay": 0.01,
            "batch_size": 32,
            "max_epochs": 12,
            "patience": 3,
            "minimum_improvement": 0.0001,
            "seed": 0,
            "device": "cpu",
        },
        "mlp_ensemble": {"seeds": [0, 1]},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_prepare_training_rows_excludes_senses_and_splits_deterministically(
    tmp_path: Path,
) -> None:
    config_path = write_fixture(tmp_path)
    settings = load_regression_settings(config_path)
    first = prepare_training_rows(settings)
    second = prepare_training_rows(settings)

    assert len(first) == 100
    assert first["split"].value_counts().to_dict() == {
        "train": 80,
        "dev": 10,
        "test": 10,
    }
    pd.testing.assert_frame_equal(first, second)
    assert not first["word"].str.contains("(", regex=False).any()


def test_pipeline_trains_and_predicts_with_fake_encoder(tmp_path: Path) -> None:
    config_path = write_fixture(tmp_path)
    results_dir = tmp_path / "results"

    assert embed_words(config_path, results_dir, encoder=FakeEncoder()) == 100
    selected_alphas = train_regressor(config_path, results_dir)
    selected_epoch = train_mlp(config_path, results_dir)
    ensemble_epochs = train_mlp_ensemble(config_path, results_dir)
    predictions = predict_words(
        ["word10", "word90"],
        config_path,
        results_dir,
        encoder=FakeEncoder(),
        model_name="mlp_ensemble",
    )

    assert set(selected_alphas) == set(
        load_regression_settings(config_path).target_columns
    )
    assert predictions.shape == (2, 10)
    assert selected_epoch >= 1
    assert len(ensemble_epochs) == 2
    assert (results_dir / "data" / "ridge_regressors.pkl").exists()
    assert (results_dir / "data" / "mlp_regressor.pt").exists()
    assert (results_dir / "data" / "mlp_ensemble.pt").exists()
    assert (results_dir / "tables" / "regression_metrics.csv").exists()
    assert (results_dir / "tables" / "model_comparison.csv").exists()
    comparison = pd.read_csv(results_dir / "tables" / "model_comparison.csv")
    assert set(comparison["model"]) == {
        "mean_baseline",
        "ridge",
        "mlp",
        "mlp_ensemble",
    }
    metadata = json.loads((results_dir / "metadata.json").read_text())
    assert metadata == {
        "experiment_id": "test_concreteness",
        "embedding_model": "fake/model",
    }
