from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tqdm import tqdm

from experiments.glasgow_concreteness_regression.embed_data import (
    DEFAULT_CONFIG,
)
from experiments.glasgow_concreteness_regression.prepare_data import (
    load_regression_settings,
    prepare_embedding_inputs,
)
from lib.embed_locally import (
    get_local_embedding_fingerprint,
    load_local_embedding_settings,
)
from lib.store_embeddings import EmbeddingCache, load_response_embeddings


def compute_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if np.ptp(predicted) == 0:
        return float("nan")
    observed_ranks = pd.Series(observed).rank(method="average").to_numpy()
    predicted_ranks = pd.Series(predicted).rank(method="average").to_numpy()
    return float(np.corrcoef(observed_ranks, predicted_ranks)[0, 1])


def evaluate_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(root_mean_squared_error(observed, predicted)),
        "spearman": compute_spearman(observed, predicted),
        "r2": float(r2_score(observed, predicted)),
    }


def select_ridge_alpha(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    dev_embeddings: np.ndarray,
    dev_targets: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, Ridge, pd.DataFrame]:
    candidates: list[tuple[float, Ridge, dict[str, float]]] = []
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(train_embeddings, train_targets)
        metrics = evaluate_predictions(dev_targets, model.predict(dev_embeddings))
        candidates.append((alpha, model, metrics))

    selected_alpha, selected_model, _ = min(
        candidates,
        key=lambda candidate: (candidate[2]["mae"], candidate[0]),
    )
    search = pd.DataFrame(
        [
            {"alpha": alpha, **{f"dev_{key}": value for key, value in metrics.items()}}
            for alpha, _, metrics in candidates
        ]
    )
    return selected_alpha, selected_model, search


def load_embedding_matrix(
    rows: pd.DataFrame,
    embeddings_path: Path,
    fingerprint: str,
    dimensions: int,
) -> np.ndarray:
    inputs = prepare_embedding_inputs(rows)
    with EmbeddingCache(embeddings_path) as cache:
        embeddings = load_response_embeddings(
            inputs,
            cache,
            fingerprint,
            dimensions,
        )
    return np.vstack([embeddings[word_id] for word_id in rows["word_id"]])


def build_metric_row(
    model_name: str,
    split: str,
    training_rows: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    alpha: float | None,
) -> dict[str, float | int | str | None]:
    return {
        "model": model_name,
        "split": split,
        "training_rows": training_rows,
        "evaluation_rows": len(observed),
        "alpha": alpha,
        **evaluate_predictions(observed, predicted),
    }


def train_regressor(
    config_path: Path,
    results_dir: Path | None = None,
) -> dict[str, float]:
    regression = load_regression_settings(config_path)
    embedding = load_local_embedding_settings(config_path)
    results_dir = results_dir or Path("results") / regression.experiment_id
    data_dir = results_dir / "data"
    tables_dir = results_dir / "tables"
    rows = pd.read_csv(data_dir / "split_assignments.csv")
    matrix = load_embedding_matrix(
        rows,
        data_dir / "embeddings.sqlite",
        get_local_embedding_fingerprint(embedding),
        embedding.dimensions,
    )

    train_mask = rows["split"].eq("train").to_numpy()
    dev_mask = rows["split"].eq("dev").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()
    fit_mask = train_mask | dev_mask
    models: dict[str, Ridge] = {}
    selected_alphas: dict[str, float] = {}
    searches: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str | None]] = []
    prediction_rows: list[pd.DataFrame] = []

    for target_name in tqdm(
        regression.target_columns,
        desc="Training Glasgow targets",
        unit="target",
    ):
        target = rows[target_name].to_numpy(dtype=np.float64)
        selected_alpha, selection_model, search = select_ridge_alpha(
            matrix[train_mask],
            target[train_mask],
            matrix[dev_mask],
            target[dev_mask],
            regression.ridge_alphas,
        )
        search.insert(0, "target", target_name)
        searches.append(search)

        dev_prediction = selection_model.predict(matrix[dev_mask])
        train_mean = float(target[train_mask].mean())
        final_model = Ridge(alpha=selected_alpha)
        final_model.fit(matrix[fit_mask], target[fit_mask])
        test_prediction = final_model.predict(matrix[test_mask])
        fit_mean = float(target[fit_mask].mean())
        metric_rows.extend(
            [
                {
                    "target": target_name,
                    **build_metric_row(
                        "mean_baseline",
                        "dev",
                        int(train_mask.sum()),
                        target[dev_mask],
                        np.full(dev_mask.sum(), train_mean),
                        None,
                    ),
                },
                {
                    "target": target_name,
                    **build_metric_row(
                        "ridge",
                        "dev",
                        int(train_mask.sum()),
                        target[dev_mask],
                        dev_prediction,
                        selected_alpha,
                    ),
                },
                {
                    "target": target_name,
                    **build_metric_row(
                        "mean_baseline",
                        "test",
                        int(fit_mask.sum()),
                        target[test_mask],
                        np.full(test_mask.sum(), fit_mean),
                        None,
                    ),
                },
                {
                    "target": target_name,
                    **build_metric_row(
                        "ridge",
                        "test",
                        int(fit_mask.sum()),
                        target[test_mask],
                        test_prediction,
                        selected_alpha,
                    ),
                },
            ]
        )
        target_predictions = rows.loc[test_mask, ["word_id", "word"]].copy()
        target_predictions["target"] = target_name
        target_predictions["observed"] = target[test_mask]
        target_predictions["predicted"] = test_prediction
        target_predictions["residual"] = target[test_mask] - test_prediction
        prediction_rows.append(target_predictions)
        models[target_name] = final_model
        selected_alphas[target_name] = selected_alpha

    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(searches, ignore_index=True).to_csv(
        tables_dir / "ridge_alpha_search.csv",
        index=False,
    )
    pd.DataFrame(metric_rows).to_csv(
        tables_dir / "regression_metrics.csv",
        index=False,
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        data_dir / "test_predictions.csv",
        index=False,
    )
    bundle = {
        "models": models,
        "embedding_model": embedding.model,
        "embedding_revision": embedding.revision,
        "embedding_dimensions": embedding.dimensions,
        "target_columns": regression.target_columns,
        "selected_alphas": selected_alphas,
    }
    with (data_dir / "ridge_regressors.pkl").open("wb") as destination:
        pickle.dump(bundle, destination)
    return selected_alphas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ridge regressors for all Glasgow ratings"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    alphas = train_regressor(arguments.config)
    for target, alpha in alphas.items():
        print(f"{target}: selected ridge alpha {alpha:g}")
