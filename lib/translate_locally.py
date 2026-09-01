from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from lib.prepare_translation_batches import (
    get_text_batches,
    get_translation_batches,
    split_text,
)


TRANSLATION_PROVIDER = "local-nllb-200-distilled-600m"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
TRANSLATION_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
MARIAN_TRANSLATION_PROVIDER = "local-marian"
MARIAN_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"
MARIAN_TRANSLATION_REVISION = "848eae0c1676cfce9bb791c200e8228e5a6396ff"
LANGUAGE_CODES = {
    "az": "azj_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "la": "lat_Latn",
    "lg": "lug_Latn",
    "mi": "mri_Latn",
    "mr": "mar_Deva",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "sn": "sna_Latn",
    "st": "sot_Latn",
    "te": "tel_Telu",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "yo": "yor_Latn",
    "zh": "zho_Hans",
    "zu": "zul_Latn",
}


@dataclass(frozen=True)
class LocalTranslationSettings:
    model_id: str
    revision: str
    device: str
    batch_size: int
    max_batch_characters: int
    max_batch_items: int
    max_chunk_characters: int
    max_input_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class MarianSettings:
    model_id: str
    revision: str
    device: str
    dtype: str
    model_batch_size: int
    max_batch_characters: int
    max_batch_items: int
    max_chunk_characters: int
    max_input_tokens: int
    max_output_tokens: int


def load_local_translation_settings(raw: dict[str, Any]) -> LocalTranslationSettings:
    expected = (TRANSLATION_PROVIDER, TRANSLATION_MODEL, TRANSLATION_REVISION)
    configured = (
        raw["translation_provider"],
        raw["translation_model"],
        raw["translation_revision"],
    )
    if configured != expected:
        raise ValueError("The local translation model configuration is not pinned")
    return LocalTranslationSettings(
        model_id=raw["translation_model"],
        revision=raw["translation_revision"],
        device=raw["device"],
        batch_size=raw["batch_size"],
        max_batch_characters=raw["max_batch_characters"],
        max_batch_items=raw["max_batch_items"],
        max_chunk_characters=raw["max_chunk_characters"],
        max_input_tokens=raw["max_input_tokens"],
        max_output_tokens=raw["max_output_tokens"],
    )


def load_marian_settings(raw: dict[str, Any]) -> MarianSettings:
    expected = (
        MARIAN_TRANSLATION_PROVIDER,
        MARIAN_TRANSLATION_MODEL,
        MARIAN_TRANSLATION_REVISION,
    )
    configured = (
        raw["translation_provider"],
        raw["translation_model"],
        raw["translation_revision"],
    )
    if configured != expected:
        raise ValueError("The local translation model configuration is not pinned")
    return MarianSettings(
        model_id=raw["translation_model"],
        revision=raw["translation_revision"],
        device=raw["device"],
        dtype=raw["dtype"],
        model_batch_size=raw["model_batch_size"],
        max_batch_characters=raw["max_batch_characters"],
        max_batch_items=raw["max_batch_items"],
        max_chunk_characters=raw["max_chunk_characters"],
        max_input_tokens=raw["max_input_tokens"],
        max_output_tokens=raw["max_output_tokens"],
    )


class LocalTranslator:
    def __init__(self, settings: LocalTranslationSettings) -> None:
        if settings.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; set translation device to cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id, revision=settings.revision
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.model_id, revision=settings.revision
        ).to(settings.device)
        self.model.eval()
        self.device = settings.device
        self.batch_size = settings.batch_size
        self.max_chunk_characters = settings.max_chunk_characters
        self.max_input_tokens = settings.max_input_tokens
        self.max_output_tokens = settings.max_output_tokens

    def translate(
        self, texts: list[str], source_language: str, target_language: str
    ) -> list[str]:
        self.tokenizer.src_lang = LANGUAGE_CODES[source_language]
        chunks_by_text = [
            split_text(text, self.max_chunk_characters) for text in texts
        ]
        chunks = [chunk for text_chunks in chunks_by_text for chunk in text_chunks]
        translated_chunks: list[str] = []
        target_token = self.tokenizer.convert_tokens_to_ids(
            LANGUAGE_CODES[target_language]
        )
        for offset in range(0, len(chunks), self.batch_size):
            inputs = self.tokenizer(
                chunks[offset : offset + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_token,
                    max_length=self.max_output_tokens,
                )
            translated_chunks.extend(
                self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )
        translations = []
        offset = 0
        for text_chunks in chunks_by_text:
            end = offset + len(text_chunks)
            translations.append("\n".join(translated_chunks[offset:end]))
            offset = end
        return translations


class MarianTranslator:
    def __init__(self, settings: MarianSettings) -> None:
        if settings.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; set device to cpu in the config")
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on this host")
        self.device = torch.device(settings.device)
        self.dtype = getattr(torch, settings.dtype)
        self.batch_size = settings.model_batch_size
        self.max_input_tokens = settings.max_input_tokens
        self.max_output_tokens = settings.max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_id, revision=settings.revision
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.model_id, revision=settings.revision
        ).to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def split_by_tokens(self, text: str) -> list[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.max_input_tokens:
            return [text]
        return [
            self.tokenizer.decode(
                token_ids[offset : offset + self.max_input_tokens],
                skip_special_tokens=True,
            ).strip()
            for offset in range(0, len(token_ids), self.max_input_tokens)
        ]

    def translate(
        self, texts: list[str], source_language: str, target_language: str
    ) -> list[str]:
        if target_language != "en":
            raise ValueError("The configured Marian model only translates to English")
        chunks_by_text = [self.split_by_tokens(text) for text in texts]
        chunks = [chunk for text_chunks in chunks_by_text for chunk in text_chunks]
        translated_chunks: list[str] = []
        for offset in range(0, len(chunks), self.batch_size):
            inputs = self.tokenizer(
                chunks[offset : offset + self.batch_size],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs, max_length=self.max_output_tokens
                )
            translated_chunks.extend(
                self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            )
        translations = []
        offset = 0
        for text_chunks in chunks_by_text:
            end = offset + len(text_chunks)
            translations.append("\n".join(translated_chunks[offset:end]))
            offset = end
        return translations


def get_local_translator(
    settings: LocalTranslationSettings, translator: Any | None = None
) -> Any:
    if translator is not None:
        return translator
    return LocalTranslator(settings)


def get_marian_translator(
    settings: MarianSettings, translator: Any | None = None
) -> Any:
    if translator is not None:
        return translator
    return MarianTranslator(settings)


def translate_texts(
    translator: Any,
    texts: list[str],
    target_language_code: str,
    source_language_code: str = "en",
) -> list[str]:
    translations = translator.translate(
        texts, source_language_code, target_language_code
    )
    if len(translations) != len(texts) or any(
        not text.strip() for text in translations
    ):
        raise RuntimeError(
            f"Expected {len(texts)} non-empty {target_language_code} translations, "
            f"received {len(translations)}"
        )
    return [text.strip() for text in translations]
