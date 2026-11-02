from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.glasgow_concreteness_regression.embed_data import DEFAULT_CONFIG
from experiments.glasgow_concreteness_regression.prepare_data import (
    load_regression_settings,
    normalize_word,
)
from experiments.glasgow_concreteness_regression.train_mlp import NormsMLP
from lib.embed_locally import LocalSentenceEncoder, load_local_embedding_settings


def predict_with_ridge(
    vectors: np.ndarray,
    bundle: dict,
    targets: tuple[str, ...],
) -> dict[str, np.ndarray]:
    return {
        target: bundle["models"][target].predict(vectors)
        for target in targets
    }


def predict_with_mlp(
    vectors: np.ndarray,
    bundle: dict,
) -> dict[str, np.ndarray]:
    model = NormsMLP(
        bundle["input_dimension"],
        bundle["hidden_dimension"],
        bundle["output_dimension"],
        bundle["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    inputs = (torch.as_tensor(vectors) - bundle["input_mean"]) / bundle["input_scale"]
    with torch.no_grad():
        normalized = model(inputs)
        predictions = normalized * bundle["target_scale"] + bundle["target_mean"]
    values = predictions.numpy()
    return {
        target: values[:, index]
        for index, target in enumerate(bundle["target_columns"])
    }


def predict_with_mlp_ensemble(
    vectors: np.ndarray,
    bundle: dict,
) -> dict[str, np.ndarray]:
    inputs = (torch.as_tensor(vectors) - bundle["input_mean"]) / bundle["input_scale"]
    member_predictions: list[np.ndarray] = []
    for member in bundle["members"]:
        model = NormsMLP(
            bundle["input_dimension"],
            bundle["hidden_dimension"],
            bundle["output_dimension"],
            bundle["dropout"],
        )
        model.load_state_dict(member["state_dict"])
        model.eval()
        with torch.no_grad():
            normalized = model(inputs)
            predictions = normalized * bundle["target_scale"] + bundle["target_mean"]
        member_predictions.append(predictions.numpy())
    values = np.stack(member_predictions).mean(axis=0)
    return {
        target: values[:, index]
        for index, target in enumerate(bundle["target_columns"])
    }


def predict_words(
    words: list[str],
    config_path: Path,
    results_dir: Path | None = None,
    encoder: Any | None = None,
    allow_model_download: bool = False,
    model_name: str = "mlp_ensemble",
) -> pd.DataFrame:
    regression = load_regression_settings(config_path)
    embedding = load_local_embedding_settings(config_path)
    results_dir = results_dir or Path("results") / regression.experiment_id
    encoder = encoder or LocalSentenceEncoder(
        embedding,
        allow_model_download=allow_model_download,
    )
    normalized = [normalize_word(word) for word in words]
    vectors = encoder.encode(normalized, batch_size=embedding.batch_size)
    if model_name == "ridge":
        with (results_dir / "data" / "ridge_regressors.pkl").open("rb") as source:
            bundle = pickle.load(source)
        predictions = predict_with_ridge(vectors, bundle, regression.target_columns)
    elif model_name == "mlp":
        bundle = torch.load(
            results_dir / "data" / "mlp_regressor.pt",
            map_location="cpu",
            weights_only=True,
        )
        predictions = predict_with_mlp(vectors, bundle)
    else:
        bundle = torch.load(
            results_dir / "data" / "mlp_ensemble.pt",
            map_location="cpu",
            weights_only=True,
        )
        predictions = predict_with_mlp_ensemble(vectors, bundle)
    return pd.DataFrame({"word": normalized, **predictions})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict all Glasgow Norms ratings for words"
    )
    parser.add_argument("words", nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--model",
        choices=("mlp_ensemble", "mlp", "ridge"),
        default="mlp_ensemble",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = predict_words(
        arguments.words,
        arguments.config,
        allow_model_download=arguments.allow_model_download,
        model_name=arguments.model,
    )
    print(output.to_csv(index=False), end="")
