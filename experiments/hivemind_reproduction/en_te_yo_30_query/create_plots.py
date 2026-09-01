from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm-response-diversity-matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from experiments.hivemind_reproduction.en_te_yo_30_query.manage_generation_records import (
    DATA_PATH,
    EMBEDDINGS_PATH,
    LANGUAGES,
    METADATA_PATH,
    PLOTS_PATH,
    TABLES_PATH,
)
from lib.compute_similarity import (
    normalize_matrix,
)
from lib.compute_identifiers import slugify
from lib.embed_with_bge import (
    BGE_DIMENSIONS,
    BGE_MAX_LENGTH,
    BGE_REVISION,
    get_bge_fingerprint,
)
from lib.store_embeddings import EmbeddingCache, load_response_embeddings

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def format_contextual_title(title: str, context: str | None) -> str:
    return f"{title} — {context}" if context else title


def create_annotated_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    color_label: str,
    value_format: str,
    vmin: float,
    vmax: float,
    mask: np.ndarray | None = None,
) -> plt.Figure:
    width = max(7, 1.2 * len(column_labels))
    height = max(5, 0.55 * len(row_labels))
    figure, axis = plt.subplots(figsize=(width, height))
    cmap = matplotlib.colormaps["YlOrRd"].with_extremes(bad="white")
    displayed_values = np.ma.array(values, mask=mask) if mask is not None else values
    image = axis.imshow(displayed_values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(column_labels)), labels=column_labels, rotation=35, ha="right")
    axis.set_yticks(range(len(row_labels)), labels=row_labels)
    axis.set_title(title)
    if mask is not None:
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if mask is not None and mask[row, column]:
                continue
            value = values[row, column]
            color = "white" if value > vmin + 0.65 * (vmax - vmin) else "black"
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                color=color,
            )
    figure.colorbar(image, ax=axis, label=color_label, shrink=0.85)
    figure.tight_layout()
    return figure


def plot_intra_bins(
    intra: pd.DataFrame,
    output_dir: Path,
    context: str | None = None,
) -> None:
    models = sorted(intra["model"].unique())
    edges = np.linspace(0, 1, 11)
    values = np.column_stack(
        [
            np.histogram(intra.loc[intra["model"] == model, "mean_similarity"], edges)[0]
            for model in models
        ]
    ).astype(float)
    values /= np.maximum(values.sum(axis=0, keepdims=True), 1)
    values = 100 * values[::-1]
    labels = [f"{edges[index]:.1f}-{edges[index + 1]:.1f}" for index in range(9, -1, -1)]
    figure = create_annotated_heatmap(
        values,
        labels,
        models,
        format_contextual_title("Intra-model repetition by prompt", context),
        "Prompts (%)",
        ".0f",
        0,
        100,
    )
    save_figure(figure, output_dir, "intra_model_similarity_bins")


def plot_model_matrix(
    intra: pd.DataFrame,
    inter: pd.DataFrame,
    output_dir: Path,
    matrix_output: Path,
    context: str | None = None,
) -> None:
    models = sorted(set(intra["model"]) | set(inter["model_1"]) | set(inter["model_2"]))
    model_index = {model: index for index, model in enumerate(models)}
    matrix = np.eye(len(models))
    for model, group in intra.groupby("model"):
        matrix[model_index[model], model_index[model]] = group["mean_similarity"].mean()
    for pair, group in inter.groupby(["model_1", "model_2"]):
        left, right = map(model_index.get, pair)
        matrix[left, right] = matrix[right, left] = group["mean_similarity"].mean()
    lower = min(0.5, float(matrix.min()))
    figure = create_annotated_heatmap(
        matrix,
        models,
        models,
        format_contextual_title("Mean response similarity within and across models", context),
        "Cosine similarity",
        ".3f",
        lower,
        1,
        mask=np.triu(np.ones_like(matrix, dtype=bool), k=1),
    )
    save_figure(figure, output_dir, "inter_model_similarity_heatmap")
    pd.DataFrame(matrix, index=models, columns=models).to_csv(matrix_output)


def plot_intra_distributions(
    intra: pd.DataFrame,
    output_dir: Path,
    context: str | None = None,
) -> None:
    models = sorted(intra["model"].unique())
    values = [intra.loc[intra["model"] == model, "mean_similarity"] for model in models]
    figure, axis = plt.subplots(figsize=(max(8, len(models) * 1.4), 5))
    axis.boxplot(values, tick_labels=models, showmeans=True)
    axis.set_ylabel("Prompt-level mean cosine similarity")
    axis.set_title(format_contextual_title("Intra-model similarity across prompts", context))
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_dir, "intra_model_distributions")


def compute_bootstrap_interval(
    values: np.ndarray,
    seed: int,
    iterations: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def plot_intra_inter_comparison(
    intra: pd.DataFrame,
    inter: pd.DataFrame,
    output_dir: Path,
    seed: int,
    context: str | None = None,
) -> None:
    prompt_values = [
        intra.groupby("query_id")["mean_similarity"].mean().to_numpy(),
        inter.groupby("query_id")["mean_similarity"].mean().to_numpy(),
    ]
    means = np.array([values.mean() for values in prompt_values])
    intervals = np.array(
        [
            compute_bootstrap_interval(values, seed + index)
            for index, values in enumerate(prompt_values)
        ]
    )
    errors = np.vstack([means - intervals[:, 0], intervals[:, 1] - means])
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.bar(
        ["Intra-model", "Inter-model"],
        means,
        yerr=errors,
        capsize=6,
        color=["#4C78A8", "#F58518"],
    )
    axis.set_ylabel("Mean cosine similarity")
    axis.set_title(format_contextual_title("Prompt-bootstrap 95% confidence intervals", context))
    axis.set_ylim(max(0, means.min() - 0.2), min(1, means.max() + 0.1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, output_dir, "intra_vs_inter_similarity")


def plot_random_baseline(
    baseline: pd.DataFrame,
    intra: pd.DataFrame,
    inter: pd.DataFrame,
    output_dir: Path,
    context: str | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(baseline["similarity"], bins=np.linspace(-0.2, 1, 49), color="#72B7B2", alpha=0.85)
    axis.axvline(intra["mean_similarity"].mean(), color="#4C78A8", label="Mean intra-model")
    axis.axvline(inter["mean_similarity"].mean(), color="#F58518", label="Mean inter-model")
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Random cross-prompt pairs")
    axis.set_title(format_contextual_title("Random-pair similarity baseline", context))
    axis.legend()
    figure.tight_layout()
    save_figure(figure, output_dir, "random_pair_baseline")


def plot_pca(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    output_dir: Path,
    query_match: str | None,
    selected_query_id: str | None = None,
    context: str | None = None,
) -> None:
    if selected_query_id:
        candidates = responses[responses["query_id"] == selected_query_id]
        query_id = candidates.iloc[0]["query_id"]
    elif query_match:
        matches = responses["query"].str.contains(
            query_match, case=False, regex=False
        )
        candidates = responses[matches]
        query_id = candidates.iloc[0]["query_id"]
    else:
        query_id = responses.iloc[0]["query_id"]
    group = responses[responses["query_id"] == query_id]
    matrix = normalize_matrix(group, embeddings)
    centered = matrix - matrix.mean(axis=0)
    left_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = left_vectors[:, :2] * singular_values[:2]
    models = sorted(group["model"].unique())
    figure, axis = plt.subplots(figsize=(8, 6))
    for model in models:
        mask = group["model"].eq(model).to_numpy()
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], label=model, alpha=0.8, s=35)
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    title = group.iloc[0].get("base_query") or group.iloc[0]["query"]
    axis.set_title(format_contextual_title(title, context))
    axis.legend(fontsize=8)
    figure.tight_layout()
    label = group.iloc[0].get("query_tag") or group.iloc[0]["query"]
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:60]
    save_figure(figure, output_dir, f"pca_{slug}")


def format_language_pair_label(language_1: str, language_2: str) -> str:
    return f"{LANGUAGES[language_1]}–{LANGUAGES[language_2]}"


def plot_language_analysis(
    data_dir: Path,
    tables_dir: Path,
    output_dir: Path,
    cache_path: Path,
    metadata_path: Path,
    language_code: str,
    query_id: str,
    seed: int,
) -> None:
    language = LANGUAGES[language_code]
    responses = pd.read_csv(data_dir / "responses_normalized.csv")
    intra = pd.read_csv(tables_dir / "intra_model_prompt_scores.csv")
    inter = pd.read_csv(tables_dir / "inter_model_prompt_scores.csv")
    baseline = pd.read_csv(data_dir / "random_pair_baseline.csv")
    embeddings = load_cached_analysis_embeddings(responses, metadata_path, cache_path)
    plot_intra_bins(intra, output_dir, language)
    plot_model_matrix(
        intra,
        inter,
        output_dir,
        tables_dir / "model_similarity_matrix.csv",
        language,
    )
    plot_intra_distributions(intra, output_dir, language)
    plot_intra_inter_comparison(intra, inter, output_dir, seed, language)
    plot_random_baseline(baseline, intra, inter, output_dir, language)
    plot_pca(responses, embeddings, output_dir, None, query_id, language)


def plot_cross_language_bins(
    scores: pd.DataFrame,
    output_dir: Path,
    context: str | None = None,
) -> None:
    edges = np.linspace(0, 1, 11)
    for model, model_scores in scores.groupby("model", sort=True):
        pairs = sorted(
            set(zip(model_scores["language_code_1"], model_scores["language_code_2"]))
        )
        values = np.column_stack(
            [
                np.histogram(
                    model_scores.loc[
                        (model_scores["language_code_1"] == language_1)
                        & (model_scores["language_code_2"] == language_2),
                        "mean_similarity",
                    ],
                    edges,
                )[0]
                for language_1, language_2 in pairs
            ]
        ).astype(float)
        values /= np.maximum(values.sum(axis=0, keepdims=True), 1)
        values = 100 * values[::-1]
        row_labels = [
            f"{edges[index]:.1f}-{edges[index + 1]:.1f}" for index in range(9, -1, -1)
        ]
        column_labels = [format_language_pair_label(*pair) for pair in pairs]
        figure = create_annotated_heatmap(
            values,
            row_labels,
            column_labels,
            format_contextual_title(f"Cross-language similarity by prompt — {model}", context),
            "Prompts (%)",
            ".0f",
            0,
            100,
        )
        save_figure(
            figure,
            output_dir,
            f"cross_language_similarity_bins_{slugify(model)}",
        )


def plot_language_matrices(
    scores: pd.DataFrame,
    intra: dict[str, pd.DataFrame],
    matrix_output_dir: Path,
    output_dir: Path,
    languages: list[str],
    context: str | None = None,
) -> None:
    models = sorted(scores["model"].unique())
    for model in models:
        matrix = np.eye(len(languages))
        for index, code in enumerate(languages):
            values = intra[code].loc[intra[code]["model"] == model, "mean_similarity"]
            matrix[index, index] = values.mean()
        for left, language_1 in enumerate(languages):
            for right, language_2 in enumerate(languages[left + 1 :], start=left + 1):
                pair = scores[
                    (scores["model"] == model)
                    & (scores["language_code_1"] == min(language_1, language_2))
                    & (scores["language_code_2"] == max(language_1, language_2))
                ]
                matrix[left, right] = matrix[right, left] = pair["mean_similarity"].mean()
        labels = [LANGUAGES[code] for code in languages]
        lower = min(0.5, float(matrix.min()))
        figure = create_annotated_heatmap(
            matrix,
            labels,
            labels,
            format_contextual_title(
                f"Mean within- and cross-language similarity — {model}", context
            ),
            "Cosine similarity",
            ".3f",
            lower,
            1,
            mask=np.triu(np.ones_like(matrix, dtype=bool), k=1),
        )
        name = f"language_similarity_matrix_{slugify(model)}"
        save_figure(figure, output_dir, name)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            matrix_output_dir / f"{name}.csv"
        )


def plot_cross_language_analysis(
    tables_dir: Path,
    output_dir: Path,
    languages: list[str],
    intra: dict[str, pd.DataFrame],
    context: str | None = None,
) -> None:
    scores = pd.read_csv(tables_dir / "cross_language_prompt_scores.csv")
    plot_cross_language_bins(scores, output_dir, context)
    plot_language_matrices(scores, intra, tables_dir, output_dir, languages, context)


def plot_aggregate_pca(
    responses: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    output_dir: Path,
    languages: list[str],
    title: str,
    seed: int,
) -> None:
    markers = {"en": "o", "yo": "s", "te": "^"}
    matrix = np.stack([embeddings[response_id] for response_id in responses["response_id"]])
    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    coordinates = pca.fit_transform(matrix)
    models = sorted(responses["model"].unique())
    model_markers = dict(zip(models, ("o", "s"), strict=True))
    palette = plt.get_cmap("tab10").colors
    variants = (
        (
            "aggregate_pca_by_language_and_model",
            dict(zip(models, palette)),
            markers,
            True,
        ),
        (
            "aggregate_pca_language_colors_model_markers",
            dict(zip(languages, palette)),
            model_markers,
            False,
        ),
    )
    explained = 100 * pca.explained_variance_ratio_
    for name, colors, point_markers, color_by_model in variants:
        figure, axis = plt.subplots(figsize=(10, 7))
        for model in models:
            for language_code in languages:
                mask = (
                    responses["model"].eq(model)
                    & responses["language_code"].eq(language_code)
                ).to_numpy()
                color_key = model if color_by_model else language_code
                marker_key = language_code if color_by_model else model
                label_parts = (
                    (model, LANGUAGES[language_code])
                    if color_by_model
                    else (LANGUAGES[language_code], model)
                )
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    color=colors[color_key],
                    marker=point_markers[marker_key],
                    label=" — ".join(label_parts),
                    alpha=0.45,
                    edgecolors="none",
                    s=22,
                )
        axis.set_xlabel(f"Principal component 1 ({explained[0]:.1f}% variance)")
        axis.set_ylabel(f"Principal component 2 ({explained[1]:.1f}% variance)")
        axis.set_title(title)
        axis.legend(fontsize=8, ncols=2)
        axis.grid(alpha=0.15)
        figure.tight_layout()
        save_figure(figure, output_dir, name)


def load_cached_analysis_embeddings(
    responses: pd.DataFrame,
    metadata_path: Path,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    metadata = json.loads(metadata_path.read_text())
    fingerprint = get_bge_fingerprint(
        model=metadata["embedding_model"],
        revision=BGE_REVISION,
        max_length=BGE_MAX_LENGTH,
    )
    with EmbeddingCache(cache_path) as cache:
        return load_response_embeddings(
            responses,
            cache,
            fingerprint,
            BGE_DIMENSIONS,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multilingual Hivemind analyses")
    parser.add_argument("--data-dir", default=str(DATA_PATH))
    parser.add_argument("--tables-dir", default=str(TABLES_PATH))
    parser.add_argument("--output-dir", default=str(PLOTS_PATH))
    parser.add_argument("--cache", default=str(EMBEDDINGS_PATH))
    parser.add_argument("--metadata", default=str(METADATA_PATH))
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--query-id", default="q003")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def plot_all_analyses(args: argparse.Namespace) -> None:
    data_root = Path(args.data_dir)
    tables_root = Path(args.tables_dir)
    output_root = Path(args.output_dir)
    metadata_path = Path(args.metadata)
    for code in args.languages:
        plot_language_analysis(
            data_root / code,
            tables_root / code,
            output_root / code,
            Path(args.cache),
            metadata_path,
            code,
            args.query_id,
            args.seed,
        )
    intra = {
        code: pd.read_csv(tables_root / code / "intra_model_prompt_scores.csv")
        for code in args.languages
    }
    plot_cross_language_analysis(
        tables_root / "cross_language",
        output_root / "cross_language",
        args.languages,
        intra,
    )
    aggregate_responses = pd.concat(
        [
            pd.read_csv(data_root / code / "responses_normalized.csv")
            for code in args.languages
        ],
        ignore_index=True,
    )
    aggregate_embeddings = load_cached_analysis_embeddings(
        aggregate_responses,
        metadata_path,
        Path(args.cache),
    )
    plot_aggregate_pca(
        aggregate_responses,
        aggregate_embeddings,
        output_root / "cross_language",
        args.languages,
        "Aggregate PCA — multilingual responses",
        args.seed,
    )
    translated_tables = tables_root / "translated_cross_language"
    translated_data = data_root / "translated_cross_language"
    if translated_tables.exists():
        translated_intra = pd.read_csv(
            translated_tables / "within_language_prompt_scores.csv"
        )
        translated_by_language = {
            code: translated_intra[translated_intra["language_code"] == code]
            for code in args.languages
        }
        plot_cross_language_analysis(
            translated_tables,
            output_root / "translated_cross_language",
            args.languages,
            translated_by_language,
            "responses translated to English",
        )
        translated_responses = pd.read_csv(translated_data / "responses_normalized.csv")
        translated_embeddings = load_cached_analysis_embeddings(
            translated_responses,
            metadata_path,
            Path(args.cache),
        )
        plot_aggregate_pca(
            translated_responses,
            translated_embeddings,
            output_root / "translated_cross_language",
            args.languages,
            "Aggregate PCA — responses translated to English",
            args.seed,
        )


if __name__ == "__main__":
    plot_all_analyses(parse_args())
