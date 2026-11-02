import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from lib.embed_with_bge import BGEM3Encoder


def main() -> None:
    root = tk.Tk()
    root.title("Text Similarity")
    root.geometry("600x400")
    root.columnconfigure(0, weight=1)

    ttk.Label(root, text="First phrase").grid(
        row=0, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    first = tk.Text(root, height=6, wrap="word")
    first.grid(row=1, column=0, sticky="nsew", padx=12)

    ttk.Label(root, text="Second phrase").grid(
        row=2, column=0, sticky="w", padx=12, pady=(12, 4)
    )
    second = tk.Text(root, height=6, wrap="word")
    second.grid(row=3, column=0, sticky="nsew", padx=12)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(3, weight=1)

    result = tk.StringVar(value="Similarity: —")
    encoder = None

    def compare() -> None:
        nonlocal encoder
        try:
            result.set("Loading model…")
            root.update_idletasks()
            encoder = encoder or BGEM3Encoder(allow_model_download=True)
            texts = [first.get("1.0", "end-1c"), second.get("1.0", "end-1c")]
            vectors = encoder.encode(texts, batch_size=2)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            result.set(f"Similarity: {float(vectors[0] @ vectors[1]):.6f}")
        except Exception as error:
            result.set("Similarity: —")
            messagebox.showerror("Error", str(error))

    ttk.Button(root, text="Compute similarity", command=compare).grid(
        row=4, column=0, pady=12
    )
    ttk.Label(root, textvariable=result, font=("TkDefaultFont", 14, "bold")).grid(
        row=5, column=0, pady=(0, 12)
    )
    root.mainloop()


if __name__ == "__main__":
    main()
