from types import SimpleNamespace

import pytest

from compare_syntax import compute_text_syntactic_similarity


def make_document() -> SimpleNamespace:
    word = SimpleNamespace(id=1, head=0, upos="NOUN")
    sentence = SimpleNamespace(words=[word])
    return SimpleNamespace(sentences=[sentence])


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def bulk_process(self, lines):
        self.calls.append(lines)
        return [make_document() for _ in lines]


def test_similarity_parses_each_nonempty_line_independently() -> None:
    pipeline = FakePipeline()

    similarity, first_lines, second_lines = compute_text_syntactic_similarity(
        "First line\nSecond line",
        "Third line",
        pipeline,
        iterations=2,
    )

    assert similarity == pytest.approx(1.0)
    assert (first_lines, second_lines) == (2, 1)
    assert pipeline.calls == [["First line", "Second line"], ["Third line"]]


def test_similarity_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="at least one nonempty line"):
        compute_text_syntactic_similarity("", "Second line", FakePipeline(), 2)
