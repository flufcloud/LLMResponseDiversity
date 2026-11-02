from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.glasgow_concreteness_regression.embed_data import DEFAULT_CONFIG
from experiments.glasgow_concreteness_regression.prepare_data import (
    load_regression_settings,
)
from experiments.glasgow_concreteness_regression.train_regressor import (
    build_metric_row,
    load_embedding_matrix,
)
from lib.embed_locally import (
    get_local_embedding_fingerprint,
    load_local_embedding_settings,
)


@dataclass(frozen=True)
class MLPSettings:
    hidden_dimension: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    minimum_improvement: float
    seed: int
    device: str


class NormsMLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def load_mlp_settings(path: Path) -> MLPSettings:
    raw = json.loads(path.read_text(encoding="utf-8"))["mlp"]
    return MLPSettings(**raw)


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    settings: MLPSettings,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(inputs.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(settings.seed)
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
    )


def train_epoch(
    model: NormsMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def compute_mse(
    model: NormsMLP,
    inputs: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(inputs.astype(np.float32)).to(device)
        target_tensor = torch.from_numpy(targets.astype(np.float32)).to(device)
        return float(nn.functional.mse_loss(model(input_tensor), target_tensor).cpu())


def predict_array(
    model: NormsMLP,
    inputs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(inputs.astype(np.float32)).to(device)
        return model(tensor).cpu().numpy()


def create_model(
    input_dimension: int,
    output_dimension: int,
    settings: MLPSettings,
    device: torch.device,
) -> NormsMLP:
    model = NormsMLP(
        input_dimension,
        settings.hidden_dimension,
        output_dimension,
        settings.dropout,
    )
    return model.to(device)


def select_training_epoch(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    dev_inputs: np.ndarray,
    dev_targets: np.ndarray,
    settings: MLPSettings,
) -> tuple[NormsMLP, int, pd.DataFrame]:
    set_training_seed(settings.seed)
    device = torch.device(settings.device)
    model = create_model(
        train_inputs.shape[1],
        train_targets.shape[1],
        settings,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loader = build_loader(train_inputs, train_targets, settings)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    progress = tqdm(
        range(1, settings.max_epochs + 1),
        desc="Selecting MLP epoch",
        unit="epoch",
    )
    for epoch in progress:
        train_loss = train_epoch(model, loader, optimizer, device)
        dev_loss = compute_mse(model, dev_inputs, dev_targets, device)
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": train_loss,
                "dev_standardized_mse": dev_loss,
            }
        )
        progress.set_postfix(dev_mse=f"{dev_loss:.4f}")
        if dev_loss < best_loss - settings.minimum_improvement:
            best_loss = dev_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, pd.DataFrame(history)


def fit_fixed_epochs(
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    settings: MLPSettings,
) -> NormsMLP:
    set_training_seed(settings.seed)
    device = torch.device(settings.device)
    model = create_model(inputs.shape[1], targets.shape[1], settings, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loader = build_loader(inputs, targets, settings)
    for _ in tqdm(range(epochs), desc="Fitting final MLP", unit="epoch"):
        train_epoch(model, loader, optimizer, device)
    return model


def build_prediction_rows(
    rows: pd.DataFrame,
    test_mask: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: tuple[str, ...],
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for index, target_name in enumerate(target_names):
        target_rows = rows.loc[test_mask, ["word_id", "word"]].copy()
        target_rows["target"] = target_name
        target_rows["observed"] = targets[test_mask, index]
        target_rows["predicted"] = predictions[:, index]
        target_rows["residual"] = target_rows["observed"] - target_rows["predicted"]
        output.append(target_rows)
    return pd.concat(output, ignore_index=True)


def train_mlp(config_path: Path, results_dir: Path | None = None) -> int:
    regression = load_regression_settings(config_path)
    embedding = load_local_embedding_settings(config_path)
    settings = load_mlp_settings(config_path)
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

    input_mean, input_scale = compute_standardization(matrix[train_mask])
    target_mean, target_scale = compute_standardization(targets[train_mask])
    standardized_inputs = (matrix - input_mean) / input_scale
    standardized_targets = (targets - target_mean) / target_scale
    selection_model, best_epoch, history = select_training_epoch(
        standardized_inputs[train_mask],
        standardized_targets[train_mask],
        standardized_inputs[dev_mask],
        standardized_targets[dev_mask],
        settings,
    )
    device = torch.device(settings.device)
    dev_predictions = (
        predict_array(selection_model, standardized_inputs[dev_mask], device)
        * target_scale
        + target_mean
    )

    fit_mask = train_mask | dev_mask
    final_input_mean, final_input_scale = compute_standardization(matrix[fit_mask])
    final_target_mean, final_target_scale = compute_standardization(targets[fit_mask])
    final_inputs = (matrix - final_input_mean) / final_input_scale
    final_targets = (targets - final_target_mean) / final_target_scale
    final_model = fit_fixed_epochs(
        final_inputs[fit_mask],
        final_targets[fit_mask],
        best_epoch,
        settings,
    )
    test_predictions = (
        predict_array(final_model, final_inputs[test_mask], device)
        * final_target_scale
        + final_target_mean
    )

    metric_rows: list[dict[str, float | int | str | None]] = []
    for index, target_name in enumerate(regression.target_columns):
        metric_rows.extend(
            [
                {
                    "target": target_name,
                    **build_metric_row(
                        "mlp",
                        "dev",
                        int(train_mask.sum()),
                        targets[dev_mask, index],
                        dev_predictions[:, index],
                        None,
                    ),
                },
                {
                    "target": target_name,
                    **build_metric_row(
                        "mlp",
                        "test",
                        int(fit_mask.sum()),
                        targets[test_mask, index],
                        test_predictions[:, index],
                        None,
                    ),
                },
            ]
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables_dir / "mlp_regression_metrics.csv", index=False)
    history.to_csv(tables_dir / "mlp_training_history.csv", index=False)
    predictions = build_prediction_rows(
        rows,
        test_mask,
        targets,
        test_predictions,
        regression.target_columns,
    )
    predictions.to_csv(data_dir / "mlp_test_predictions.csv", index=False)

    ridge_metrics = pd.read_csv(tables_dir / "regression_metrics.csv")
    comparison = pd.concat(
        [
            ridge_metrics[ridge_metrics["split"].eq("test")],
            metrics[metrics["split"].eq("test")],
        ],
        ignore_index=True,
    ).sort_values(["target", "model"])
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)

    bundle = {
        "state_dict": final_model.cpu().state_dict(),
        "input_mean": torch.from_numpy(final_input_mean),
        "input_scale": torch.from_numpy(final_input_scale),
        "target_mean": torch.from_numpy(final_target_mean),
        "target_scale": torch.from_numpy(final_target_scale),
        "input_dimension": matrix.shape[1],
        "hidden_dimension": settings.hidden_dimension,
        "output_dimension": len(regression.target_columns),
        "dropout": settings.dropout,
        "target_columns": list(regression.target_columns),
        "selected_epoch": best_epoch,
        "embedding_model": embedding.model,
        "embedding_revision": embedding.revision,
    }
    torch.save(bundle, data_dir / "mlp_regressor.pt")
    return best_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen-embedding Glasgow Norms MLP"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    epoch = train_mlp(arguments.config)
    print(f"Selected MLP epoch: {epoch}")
