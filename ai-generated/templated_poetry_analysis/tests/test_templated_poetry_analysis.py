from __future__ import annotations

import importlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest


data_module = importlib.import_module(
    "ai-generated.templated_poetry_analysis.process_analysis_data"
)
prepare_module = importlib.import_module(
    "ai-generated.templated_poetry_analysis.prepare_responses"
)
embedding_module = importlib.import_module(
    "ai-generated.templated_poetry_analysis.embed_data"
)
similarity_module = importlib.import_module(
    "ai-generated.templated_poetry_analysis.analyze_similarity"
)
form_module = importlib.import_module(
    "ai-generated.templated_poetry_analysis.analyze_form"
)


def write_config(tmp_path: Path, **updates) -> Path:
    config = {
        "input_csv": str(tmp_path / "generations.csv"),
        "data_dir": str(tmp_path / "data"),
        "tables_dir": str(tmp_path / "tables"),
        "embedding": {
            "model": "fake-model",
            "output_dimensionality": 2,
            "task_prefix": "query: {content}",
            "batch_size": 2,
            "user_approved_for_api": False,
        },
        "pair_sampling": {
            "corpus_pairs": 3,
            "maximum_pairs_per_group": 2,
            "random_seed": 7,
        },
    }
    config.update(updates)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_prepare_responses_renames_adjective_group(tmp_path: Path) -> None:
    source = pd.DataFrame(
        [
            {
                "generation_id": "g1",
                "topic_id": "t1",
                "topic": "Love",
                "adjective_group": "happy",
                "adjective": "joyful",
                "sample_idx": 1,
                "response": "Bright morning.",
            }
        ]
    )
    source.to_csv(tmp_path / "generations.csv", index=False)
    config_path = write_config(tmp_path)

    result = prepare_module.prepare_responses(config_path)

    assert result.loc[0, "poem_id"] == "g1"
    assert result.loc[0, "adjective_class"] == "happy"
    assert (tmp_path / "data" / "responses.csv").exists()


def test_embedding_requires_both_authorization_controls(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)

    with pytest.raises(PermissionError, match="--confirm-api"):
        embedding_module.embed_responses(config_path, False)
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        embedding_module.embed_responses(config_path, True)


def test_syntax_features_round_trip() -> None:
    features = [
        Counter(
            {
                (0, "NOUN"): 2,
                (1, ("NOUN", ("VERB", "none"))): 1,
            }
        )
    ]

    payload = data_module.serialize_syntax(features)
    restored = data_module.deserialize_syntax(
        json.loads(json.dumps(payload))
    )

    assert restored == features


def test_parse_batch_keeps_poetic_lines_separate() -> None:
    parse_module = importlib.import_module(
        "ai-generated.templated_poetry_analysis.parse_text"
    )

    class FakeSpacy:
        def pipe(self, texts, batch_size):
            return [[] for _ in texts]

    class FakeStanza:
        def bulk_process(self, lines):
            documents = []
            for line in lines:
                word = SimpleNamespace(id=1, head=0, upos="NOUN")
                sentence = SimpleNamespace(words=[word])
                documents.append(SimpleNamespace(sentences=[sentence]))
            return documents

    batch = [
        SimpleNamespace(
            poem_id="p1",
            response_sha256="hash",
            response="First line,\nSecond line,",
        )
    ]
    lexical_config = {
        "batch_size": 8,
        "content_pos": ["NOUN"],
        "excluded_terms": [],
    }
    syntax_config = {"wl_iterations": 2}

    records = parse_module.parse_batch(
        batch,
        FakeSpacy(),
        FakeStanza(),
        lexical_config,
        syntax_config,
    )

    assert len(records[0]["syntax"]) == 2


def test_pair_sampling_is_bounded_and_deterministic() -> None:
    first = similarity_module.sample_pairs(range(10), 7, 3)
    second = similarity_module.sample_pairs(range(10), 7, 3)

    assert first == second
    assert len(first) == 7
    assert similarity_module.sample_pairs(range(3), 10, 3) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]


def test_similarity_level_uses_topic_adjective_groups() -> None:
    responses = pd.DataFrame(
        [
            ("p1", "t1", "Love", "happy", "joyful"),
            ("p2", "t1", "Love", "happy", "joyful"),
            ("p3", "t1", "Love", "sad", "somber"),
            ("p4", "t1", "Love", "sad", "somber"),
        ],
        columns=["poem_id", "topic_id", "topic", "adjective_class", "adjective"],
    )
    embeddings = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    lemma_sets = [{"sun"}, {"sun"}, {"rain"}, {"rain"}]
    syntax = [[Counter({(0, "NOUN"): 1})] for _ in range(4)]
    config = {
        "pair_sampling": {
            "corpus_pairs": 2,
            "maximum_pairs_per_group": 5,
            "random_seed": 0,
        }
    }

    scores, summary = similarity_module.analyze_level(
        "adjective",
        responses,
        embeddings,
        lemma_sets,
        syntax,
        config,
    )

    assert len(scores) == 2
    assert len(summary) == 2
    assert summary["mean_cosine_similarity"].eq(1.0).all()


def test_rhyme_distribution_includes_zero_categories() -> None:
    frame = pd.DataFrame(
        {
            "topic_id": ["t1", "t1", "t2"],
            "topic": ["Love", "Love", "Nature"],
            "rhyme_scheme": ["AA", "AA", "AB"],
        }
    )

    result = form_module.create_distribution(
        frame,
        ["topic_id", "topic"],
        "rhyme_scheme",
        "rhyme_scheme",
        ("AA", "AB", "ABC", "Other / non-rhyming"),
    )

    assert len(result) == 8
    t1_abc = result[
        (result["topic_id"] == "t1") & (result["rhyme_scheme"] == "ABC")
    ].iloc[0]
    assert t1_abc["count"] == 0
    assert t1_abc["fraction"] == 0


def test_form_summary_handles_no_scorable_meter_lines() -> None:
    frame = pd.DataFrame(
        {
            "line_count": [2],
            "stanza_count": [1],
            "four_line_stanza_count": [0],
            "rhymed_line_count": [0],
            "known_end_word_count": [0],
            "scorable_meter_line_count": [0],
            "iambic_pentameter_line_count": [0],
            "dominant_iambic_pentameter": [False],
            "iambic_alignment_score_sum": [0.0],
        }
    )

    result = form_module.summarize_form_metrics(frame)

    assert result["iambic_pentameter_line_fraction"] == 0
    assert result["mean_iambic_alignment_score"] == 0


class FakeToken:
    def __init__(self, lemma: str, pos: str, *, stop: bool = False) -> None:
        self.lemma_ = lemma
        self.lower_ = lemma.lower()
        self.pos_ = pos
        self.is_alpha = True
        self.is_stop = stop


def test_lemma_filter_removes_stopwords_and_negation() -> None:
    document = [
        FakeToken("Light", "NOUN"),
        FakeToken("never", "ADV"),
        FakeToken("the", "DET", stop=True),
    ]
    config = {
        "content_pos": ["NOUN", "VERB", "ADJ", "ADV"],
        "excluded_terms": ["no", "not", "never", "n't"],
    }
    parse_module = importlib.import_module(
        "ai-generated.templated_poetry_analysis.parse_text"
    )

    assert parse_module.get_lemmas(document, config) == ["light"]
