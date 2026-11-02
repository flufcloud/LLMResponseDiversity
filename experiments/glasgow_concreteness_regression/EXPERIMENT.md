# Glasgow Norms Regression

This experiment predicts all nine continuous Glasgow Norms ratings from frozen
`sentence-transformers/all-MiniLM-L6-v2` embeddings. Separately tuned ridge regressors provide the
linear control. A shared shallow MLP provides nonlinear predictions.

The source CSV remains unchanged. Rows whose word contains a parenthetical sense are excluded.
The remaining words receive a fixed 80/10/10 train/dev/test split, stratified on the mean
percentile across all targets. Each target's ridge alpha is selected by dev MAE. Final models are
fitted on train and dev, then evaluated once on test.

Run from the repository root:

```bash
uv run --frozen python -m experiments.glasgow_concreteness_regression.embed_data --allow-model-download
uv run --frozen python -m experiments.glasgow_concreteness_regression.train_regressor
uv run --frozen python -m experiments.glasgow_concreteness_regression.train_mlp
uv run --frozen python -m experiments.glasgow_concreteness_regression.train_mlp_ensemble
uv run --frozen python -m experiments.glasgow_concreteness_regression.predict_norms ocean justice memory
```

The first and fifth commands run local embedding inference. `--allow-model-download` permits the
first command to download the pinned public model if it is not already available locally. None of
the commands call a paid API.

Outputs are written under `results/glasgow_concreteness_regression`:

- `data/split_assignments.csv`
- `data/embeddings.sqlite`
- `data/ridge_regressors.pkl`
- `data/mlp_regressor.pt`
- `data/mlp_ensemble.pt`
- `data/test_predictions.csv`
- `data/mlp_test_predictions.csv`
- `data/mlp_ensemble_test_predictions.csv`
- `tables/ridge_alpha_search.csv`
- `tables/regression_metrics.csv`
- `tables/mlp_regression_metrics.csv`
- `tables/mlp_training_history.csv`
- `tables/mlp_ensemble_metrics.csv`
- `tables/mlp_ensemble_training_history.csv`
- `tables/model_comparison.csv`
