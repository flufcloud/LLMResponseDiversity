from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from experiments.glasgow_concreteness_regression.embed_data import DEFAULT_CONFIG
from experiments.glasgow_concreteness_regression.prepare_data import (
    load_regression_settings,
)
from experiments.glasgow_concreteness_regression.train_mlp import (
    compute_standardization,
    fit_fixed_epochs,
    load_mlp_settings,
    predict_array,
    select_training_epoch,
)
from experiments.glasgow_concreteness_regression.train_regressor import (
    build_metric_row,
    load_embedding_matrix,
)
from lib.embed_locally import (
    get_local_embedding_fingerprint,
    load_local_embedding_settings,
)


def load_ensemble_seeds(path: Path) -> tuple[int, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(raw["mlp_ensemble"]["seeds"])


def build_ensemble_predictions(
    rows: pd.DataFrame,
    test_mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    prediction_std: np.ndarray,
    target_names: tuple[str, ...],
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for index, target_name in enumerate(target_names):
        target_rows = rows.loc[test_mask, ["word_id", "word"]].copy()
        target_rows["target"] = target_name
        target_rows["observed"] = targets[test_mask, index]
        target_rows["predicted"] = predictions[:, index]
        target_rows["prediction_std"] = prediction_std[:, index]
        target_rows["residual"] = target_rows["observed"] - target_rows["predicted"]
        output.append(target_rows)
    return pd.concat(output, ignore_index=True)


def train_mlp_ensemble(
    config_path: Path,
    results_dir: Path | None = None,
) -> tuple[int, ...]:
    regression = load_regression_settings(config_path)
    embedding = load_local_embedding_settings(config_path)
    base_settings = load_mlp_settings(config_path)
    seeds = load_ensemble_seeds(config_path)
    results_dir = results_dir or Path("results") / regression.experiment_id
    data_dir = results_dir / "data"
    tables_dir = results_dir / "tables"
    rows = pd.read_csv(data_dir / "split_assignments.csv")
    matrix = load_embedding_matrix(
        rows,
        data_dir / "embeddings.sqlite",
        get_local_embedding_fingerprint(embedding),
        embedding.dimensions,
    ).astype(np.float32)
    targets = rows[list(regression.target_columns)].to_numpy(dtype=np.float32)
    train_mask = rows["split"].eq("train").to_numpy()
    dev_mask = rows["split"].eq("dev").to_numpy()
    test_mask = rows["split"].eq("test").to_numpy()
    fit_mask = train_mask | dev_mask

    input_mean, input_scale = compute_standardization(matrix[train_mask])
    target_mean, target_scale = compute_standardization(targets[train_mask])
    standardized_inputs = (matrix - input_mean) / input_scale
    standardized_targets = (targets - target_mean) / target_scale
    final_input_mean, final_input_scale = compute_standardization(matrix[fit_mask])
    final_target_mean, final_target_scale = compute_standardization(targets[fit_mask])
    final_inputs = (matrix - final_input_mean) / final_input_scale
    final_targets = (targets - final_target_mean) / final_target_scale

    members: list[dict] = []
    histories: list[pd.DataFrame] = []
    selected_epochs: list[int] = []
    dev_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    for seed in tqdm(seeds, desc="Training MLP ensemble", unit="member"):
        settings = replace(base_settings, seed=seed)
        selection_model, best_epoch, history = select_training_epoch(
            standardized_inputs[train_mask],
            standardized_targets[train_mask],
            standardized_inputs[dev_mask],
            standardized_targets[dev_mask],
            settings,
        )
        history.insert(0, "seed", seed)
        histories.append(history)
        selected_epochs.append(best_epoch)
        device = torch.device(settings.device)
        dev_predictions.append(
            predict_array(selection_model, standardized_inputs[dev_mask], device)
            * target_scale
            + target_mean
        )

        final_model = fit_fixed_epochs(
            final_inputs[fit_mask],
            final_targets[fit_mask],
            best_epoch,
            settings,
        )
        test_predictions.append(
            predict_array(final_model, final_inputs[test_mask], device)
            * final_target_scale
            + final_target_mean
        )
        members.append(
            {
                "seed": seed,
                "selected_epoch": best_epoch,
                "state_dict": final_model.cpu().state_dict(),
            }
        )

    dev_stack = np.stack(dev_predictions)
    test_stack = np.stack(test_predictions)
    dev_mean = dev_stack.mean(axis=0)
    test_mean = test_stack.mean(axis=0)
    test_std = test_stack.std(axis=0, ddof=1)
    metric_rows: list[dict[str, float | int | str | None]] = []
    for index, target_name in enumerate(regression.target_columns):
        metric_rows.extend(
            [
                {
                    "target": target_name,
                    **build_metric_row(
                        "mlp_ensemble",
                        "dev",
                        int(train_mask.sum()),
                        targets[dev_mask, index],
                        dev_mean[:, index],
                        None,
                    ),
                },
                {
                    "target": target_name,
                    **build_metric_row(
                        "mlp_ensemble",
                        "test",
                        int(fit_mask.sum()),
                        targets[test_mask, index],
                        test_mean[:, index],
                        None,
                    ),
                },
            ]
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables_dir / "mlp_ensemble_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        tables_dir / "mlp_ensemble_training_history.csv",
        index=False,
    )
    predictions = build_ensemble_predictions(
        rows,
        test_mask,
        targets,
        test_mean,
        test_std,
        regression.target_columns,
    )
    predictions.to_csv(data_dir / "mlp_ensemble_test_predictions.csv", index=False)

    comparison_path = tables_dir / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    comparison = comparison[~comparison["model"].eq("mlp_ensemble")]
    comparison = pd.concat(
        [comparison, metrics[metrics["split"].eq("test")]],
        ignore_index=True,
    ).sort_values(["target", "model"])
    comparison.to_csv(comparison_path, index=False)

    bundle = {
        "members": members,
        "input_mean": torch.from_numpy(final_input_mean),
        "input_scale": torch.from_numpy(final_input_scale),
        "target_mean": torch.from_numpy(final_target_mean),
        "target_scale": torch.from_numpy(final_target_scale),
        "input_dimension": matrix.shape[1],
        "hidden_dimension": base_settings.hidden_dimension,
        "output_dimension": len(regression.target_columns),
        "dropout": base_settings.dropout,
        "target_columns": list(regression.target_columns),
        "embedding_model": embedding.model,
        "embedding_revision": embedding.revision,
    }
    torch.save(bundle, data_dir / "mlp_ensemble.pt")
    return tuple(selected_epochs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the five-member frozen-embedding MLP ensemble"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    epochs = train_mlp_ensemble(arguments.config)
    print(f"Selected ensemble epochs: {epochs}")
