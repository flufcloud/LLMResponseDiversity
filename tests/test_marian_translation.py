import pytest

from lib.translate_locally import translate_texts


class Translator:
    def translate(self, texts, source_language, target_language):
        return [f"{target_language}:{text}" for text in texts]


def test_translates_without_hosted_client() -> None:
    translated = translate_texts(Translator(), ["hola", "adios"], "en", "es")

    assert translated == ["en:hola", "en:adios"]


def test_rejects_empty_translation() -> None:
    class EmptyTranslator:
        def translate(self, texts, source_language, target_language):
            return [""]

    with pytest.raises(RuntimeError, match="non-empty"):
        translate_texts(EmptyTranslator(), ["hola"], "en", "es")
