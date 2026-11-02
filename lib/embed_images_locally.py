from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_feature_tensor(output: Any) -> Any:
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    if hasattr(output, "norm"):
        return output
    raise TypeError(f"Embedding model returned unsupported output type: {type(output)}")


class LocalVisionLanguageEncoder:
    def __init__(self, settings: dict[str, Any], allow_model_download: bool) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install project dependencies with `uv sync --frozen`") from error

        self.torch = torch
        self.settings = settings
        load_options = {
            "revision": settings["revision"],
            "local_files_only": not allow_model_download,
            "trust_remote_code": False,
        }
        self.processor = AutoProcessor.from_pretrained(settings["model"], **load_options)
        self.model = AutoModel.from_pretrained(
            settings["model"],
            torch_dtype=getattr(torch, settings["dtype"]),
            **load_options,
        ).to(settings["device"])
        self.model.eval()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._encode_batches(texts, "text", self.settings["text_batch_size"])

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        return self._encode_batches(paths, "image", self.settings["image_batch_size"])

    def _encode_batches(
        self,
        values: list[str] | list[Path],
        modality: str,
        batch_size: int,
    ) -> np.ndarray:
        vectors = []
        for start in tqdm(range(0, len(values), batch_size), desc=f"Embedding {modality}s"):
            batch = values[start : start + batch_size]
            if modality == "text":
                inputs = self.processor(
                    text=batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.settings["max_text_tokens"],
                    return_tensors="pt",
                )
                feature_method = self.model.get_text_features
            else:
                with_images = []
                for path in batch:
                    with Image.open(path) as image:
                        with_images.append(image.convert("RGB"))
                inputs = self.processor(images=with_images, return_tensors="pt")
                feature_method = self.model.get_image_features
            inputs = {key: value.to(self.settings["device"]) for key, value in inputs.items()}
            with self.torch.inference_mode():
                output = get_feature_tensor(feature_method(**inputs))
                output = self.torch.nn.functional.normalize(output, dim=-1)
            vectors.append(output.float().cpu().numpy())
        return np.concatenate(vectors).astype(np.float32)
