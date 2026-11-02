from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.glasgow_concreteness_regression.predict_norms import predict_words
from experiments.multimodal_thinking_experiment.analyze_words import (
    extract_content_terms,
    load_spacy_model,
)


DEFAULT_CONFIG = Path(__file__).with_name("poem_imageability_config.json")
NORM_TARGETS = {
    "arousal": "arousal_mean",
    "valence": "valence_mean",
    "dominance": "dominance_mean",
    "concreteness": "concrete_mean",
    "imageability": "imagine_mean",
    "familiarity": "familiar_mean",
    "aoa": "aoa_mean",
    "size": "size_mean",
    "gender": "gender_mean",
}
NORM_LABELS = {
    "arousal": "Arousal",
    "valence": "Valence",
    "dominance": "Dominance",
    "concreteness": "Concreteness",
    "imageability": "Imageability",
    "familiarity": "Familiarity",
    "aoa": "Age of acquisition",
    "size": "Size",
    "gender": "Gender association",
}


def select_poem(responses: pd.DataFrame, seed: int | None) -> pd.Series:
    generator = np.random.default_rng(seed)
    return responses.iloc[int(generator.integers(len(responses)))]


def extract_poem_lemmas(
    response: str,
    nlp: Any,
    lexical_config: dict[str, Any],
) -> list[str]:
    _, _, lemma_terms = extract_content_terms(
        pd.Series([response]),
        nlp,
        lexical_config,
    )
    return lemma_terms[0]


def build_norm_table(
    lemmas: list[str],
    predictions: pd.DataFrame,
    norm: str,
) -> pd.DataFrame:
    scores = predictions.set_index("word")[NORM_TARGETS[norm]]
    return pd.DataFrame(
        {
            "position": range(1, len(lemmas) + 1),
            "word": lemmas,
            norm: [scores[word] for word in lemmas],
        }
    )


def plot_norm(
    values: pd.DataFrame,
    poem: pd.Series,
    norm: str,
) -> plt.Figure:
    width = min(18, max(9, 0.4 * len(values)))
    figure, axis = plt.subplots(figsize=(width, 5))
    axis.plot(
        values["position"],
        values[norm],
        color="#4C78A8",
        marker="o",
        linewidth=1.6,
        markersize=4,
    )
    axis.set_xticks(
        values["position"],
        labels=values["word"],
        rotation=60,
        ha="right",
    )
    axis.set_ylim(1, 9)
    axis.set_xlabel("Content lemmas in poem order")
    axis.set_ylabel(f"Predicted {NORM_LABELS[norm].lower()} (1–9)")
    axis.set_title(
        f"Poem {NORM_LABELS[norm].lower()}: "
        f"{poem['topic']} ({poem['thinking_level']} thinking)"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def analyze_poem_norm(
    config_path: Path,
    allow_model_download: bool = False,
    seed: int | None = None,
    norm: str = "imageability",
    show_plot: bool = True,
    nlp: Any | None = None,
    predictor: Callable[..., pd.DataFrame] = predict_words,
) -> tuple[pd.Series, pd.DataFrame]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    responses = pd.read_csv(config["input_csv"])
    lexical_config = json.loads(
        Path(config["lexical_config"]).read_text(encoding="utf-8")
    )
    progress = tqdm(total=4, desc="Analyzing poem norm", unit="stage")

    poem = select_poem(responses, seed)
    progress.update()

    nlp = nlp or load_spacy_model(lexical_config)
    lemmas = extract_poem_lemmas(poem["response"], nlp, lexical_config)
    progress.update()

    unique_lemmas = list(dict.fromkeys(lemmas))
    predictions = predictor(
        unique_lemmas,
        Path(config["regression_config"]),
        Path(config["regression_results"]),
        allow_model_download=allow_model_download,
        model_name="mlp_ensemble",
    )
    values = build_norm_table(lemmas, predictions, norm)
    progress.update()

    figure = plot_norm(values, poem, norm)
    progress.update()
    progress.close()
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return poem, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one predicted norm across a randomly selected poem"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--norm",
        choices=tuple(NORM_TARGETS),
        default="imageability",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    selected, values = analyze_poem_norm(
        arguments.config,
        allow_model_download=arguments.allow_model_download,
        seed=arguments.seed,
        norm=arguments.norm,
    )
    print(
        f"Selected {selected['poem_id']} for {selected['topic']} "
        f"({selected['thinking_level']} thinking)"
    )
    print(
        values.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
