from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from lib.measure_syntactic_similarity import (
    compute_poem_syntactic_similarity,
    load_stanza_pipeline,
    parse_text_line_syntax,
)


DEFAULT_CONFIG = Path(
    "experiments/multimodal_thinking_experiment/"
    "homogeneity_analysis_config.json"
)


def load_syntax_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["syntax"]


def compute_text_syntactic_similarity(
    first: str,
    second: str,
    pipeline: Any,
    iterations: int,
) -> tuple[float, int, int]:
    first_features = parse_text_line_syntax(first, pipeline, iterations)
    second_features = parse_text_line_syntax(second, pipeline, iterations)
    similarity = compute_poem_syntactic_similarity(
        first_features,
        second_features,
    )
    return similarity, len(first_features), len(second_features)


def main() -> None:
    root = tk.Tk()
    root.title("Syntactic Similarity")
    root.geometry("600x430")
    root.columnconfigure(0, weight=1)

    ttk.Label(root, text="First text").grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    first = tk.Text(root, height=7, wrap="word")
    first.grid(row=1, column=0, sticky="nsew", padx=12)

    ttk.Label(root, text="Second text").grid(
        row=2, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    second = tk.Text(root, height=7, wrap="word")
    second.grid(row=3, column=0, sticky="nsew", padx=12)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(3, weight=1)

    result = tk.StringVar(value="Similarity: —")
    detail = tk.StringVar(value="")
    pipeline = None
    syntax_config = load_syntax_config()

    def compare() -> None:
        nonlocal pipeline
        try:
            result.set("Loading parser…")
            detail.set("")
            root.update_idletasks()
            pipeline = pipeline or load_stanza_pipeline(syntax_config)
            similarity, first_lines, second_lines = (
                compute_text_syntactic_similarity(
                    first.get("1.0", "end-1c"),
                    second.get("1.0", "end-1c"),
                    pipeline,
                    syntax_config["wl_iterations"],
                )
            )
            result.set(f"Similarity: {similarity:.6f}")
            detail.set(
                f"Mean across {first_lines} × {second_lines} line pairs"
            )
        except Exception as error:
            result.set("Similarity: —")
            detail.set("")
            messagebox.showerror("Error", str(error))

    ttk.Button(root, text="Compute similarity", command=compare).grid(
        row=4, column=0, pady=12
    )
    ttk.Label(root, textvariable=result, font=("TkDefaultFont", 14, "bold")).grid(
        row=5, column=0
    )
    ttk.Label(root, textvariable=detail).grid(
        row=6, column=0, pady=(4, 12)
    )
    root.mainloop()


if __name__ == "__main__":
    main()
