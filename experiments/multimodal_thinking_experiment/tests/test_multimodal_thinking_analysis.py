import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.multimodal_thinking_experiment.analyze import (
    classify_rhyme_scheme,
    compute_cross_thinking_similarity,
    compute_poem_metrics,
    compute_within_condition_similarity,
    split_stanzas,
)
from experiments.multimodal_thinking_experiment.analyze_words import (
    count_unique_words,
    create_combined_topic_tables,
    create_topic_usage_tables,
    create_word_topics_table,
    extract_content_terms,
)
from experiments.multimodal_thinking_experiment.browse_basic_words import (
    get_topics as get_basic_lexical_topics,
    load_usage_tables,
    select_topic_words,
)
from experiments.multimodal_thinking_experiment.compare_vs5_lexical_diversity import (
    compare_vs5_lexical_diversity,
)
from experiments.multimodal_thinking_experiment.generate_responses import read_requests
from experiments.multimodal_thinking_experiment.judge_pii import (
    finalize_results as finalize_pii_results,
    parse_judgment,
    prepare_requests as prepare_pii_requests,
    submit_requests as submit_pii_requests,
    validate_prepared_requests as validate_pii_requests,
)
from experiments.multimodal_thinking_experiment.prepare_poems import (
    extract_ssot_poem,
    extract_vs5_poems,
    prepare_method,
)
from experiments.multimodal_thinking_experiment.plot_poem_imageability import (
    analyze_poem_norm,
)
from experiments.multimodal_thinking_experiment.summarize_lemmas import (
    summarize_lemmas as summarize_basic_lemmas,
)
from lib.compute_identifiers import compute_stable_hash
from lib.embed_with_gemini import fetch_gemini_embeddings
from lib.store_embeddings import EmbeddingCache


def test_extract_vs5_expands_five_poems_and_line_separators() -> None:
    payload = {
        "responses": [
            {"poem": f"Line {index} / Next {index}", "probability": 0.2}
            for index in range(5)
        ]
    }
    extracted = extract_vs5_poems(f"```json\n{json.dumps(payload)}\n```")

    assert len(extracted) == 5
    assert extracted[0] == ("Line 0\nNext 0", 0.2)


def test_extract_vs5_accepts_nested_provider_key_variants() -> None:
    payload = {
        "topic": "Love",
        "generated_poems": [
            {"poem_text": f"Poem {index}", "probability_score": "20%"}
            for index in range(5)
        ],
    }

    extracted = extract_vs5_poems(json.dumps(payload))

    assert extracted == [(f"Poem {index}", 0.2) for index in range(5)]


def test_extract_vs5_rejects_one_poem_with_five_descriptions() -> None:
    payload = {
        "poem": "Only one poem",
        "responses": [
            {"id": index, "style": f"Style {index}", "probability": 0.2}
            for index in range(1, 6)
        ],
    }

    with pytest.raises(ValueError, match="one poem with five probability descriptions"):
        extract_vs5_poems(json.dumps(payload))


def test_prepare_vs5_writes_one_normalized_row_per_poem(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "responses": [
            {
                "poem": f"Poem {index} / Second line",
                "probability": index / 10,
            }
            for index in range(1, 6)
        ]
    }
    pd.DataFrame(
        [
            {
                "generation_id": "gen-1",
                "method_id": "poetry_vs5",
                "topic_id": "topic-1",
                "topic": "Love",
                "hierarchy_paths": '[["SUBJECTS", "Love"]]',
                "thinking_level": "minimal",
                "sample_idx": 3,
                "response": json.dumps(payload),
                "model": "test-model",
                "model_version": "test-version",
                "generation_config": "{}",
            }
        ]
    ).to_csv(data_dir / "source.csv", index=False)
    config = {
        "methods": {
            "poetry_vs5": {
                "input_filename": "source.csv",
                "responses_per_generation": 5,
            }
        }
    }

    output_path = prepare_method("poetry_vs5", config, tmp_path)
    normalized = pd.read_csv(output_path)

    assert len(normalized) == 5
    assert normalized["generation_id"].tolist() == ["gen-1"] * 5
    assert normalized["source_sample_idx"].tolist() == [3] * 5
    assert normalized["response_idx"].tolist() == [1, 2, 3, 4, 5]
    assert normalized["verbalized_probability"].tolist() == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
    assert normalized["response"].tolist() == [
        f"Poem {index}\nSecond line" for index in range(1, 6)
    ]


def test_extract_ssot_uses_last_divider_without_llm() -> None:
    response = """Random seed: x7Q

**Manipulation:** reverse it

***

### A Poem About Love

**Quiet Thread**

First line,
Second line.
"""
    poem, rule = extract_ssot_poem(response)

    assert rule == "last_divider"
    assert poem == "First line,\nSecond line."


def test_poem_metrics_count_lines_stanzas_quatrains_and_rhyme() -> None:
    responses = pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "generation_id": "g1",
                "method_id": "poetry_basic",
                "topic_id": "t1",
                "topic": "Love",
                "thinking_level": "minimal",
                "source_sample_idx": 1,
                "response_idx": 1,
                "response": "Rose is red\nRest in bed\nOcean blue\nPromise true\n\nFifth line",
            }
        ]
    )
    metrics = compute_poem_metrics(responses).iloc[0]

    assert metrics["line_count"] == 5
    assert metrics["stanza_count"] == 2
    assert metrics["four_line_stanza_count"] == 1
    assert metrics["non_four_line_stanza_count"] == 1
    assert metrics["rhyme_scheme"] == "AABB"
    assert classify_rhyme_scheme(split_stanzas("red\nblue\nbed\ntrue")) == "ABAB"


def test_unique_words_count_occurrences_and_poem_frequency() -> None:
    responses = pd.Series(["Moon, moon—light.", "Moon’s light and moon."])

    words = count_unique_words(responses).set_index("word")

    assert words.loc["moon", "count"] == 3
    assert words.loc["moon", "decimal_frequency"] == 1.0
    assert words.loc["light", "count"] == 2
    assert words.loc["moon's", "count"] == 1
    assert words.loc["moon's", "decimal_frequency"] == 0.5


def test_content_words_filter_stop_words_pos_and_negation() -> None:
    def make_token(text, lemma, pos, *, is_stop=False, is_alpha=True):
        return SimpleNamespace(
            lower_=text,
            lemma_=lemma,
            pos_=pos,
            is_stop=is_stop,
            is_alpha=is_alpha,
        )

    docs = [
        [
            make_token("the", "the", "DET", is_stop=True),
            make_token("never", "never", "ADV"),
            make_token("n’t", "not", "ADV"),
            make_token("dreams", "dream", "NOUN"),
            make_token("brightly", "brightly", "ADV"),
        ],
        [
            make_token("dreaming", "dream", "VERB"),
            make_token("with", "with", "ADP", is_stop=True),
            make_token("stars", "star", "NOUN"),
        ],
    ]

    class FakeNlp:
        def pipe(self, responses, batch_size):
            assert list(responses) == ["poem one", "poem two"]
            assert batch_size == 64
            return iter(docs)

    config = {
        "batch_size": 64,
        "content_pos": ["NOUN", "PROPN", "VERB", "ADJ", "ADV"],
        "excluded_terms": ["no", "not", "never", "n't"],
    }
    words, lemmas, lemma_terms = extract_content_terms(
        pd.Series(["poem one", "poem two"]), FakeNlp(), config
    )

    assert set(words["word"]) == {"brightly", "dreaming", "dreams", "stars"}
    assert "never" not in set(words["word"])
    dream = lemmas.set_index("lemma").loc["dream"]
    assert dream["count"] == 2
    assert dream["decimal_frequency"] == 1.0

    generations = pd.DataFrame(
        {
            "topic": ["Love", "Love"],
            "thinking_level": ["minimal", "minimal"],
        }
    )
    usage = create_topic_usage_tables(generations, lemma_terms)["minimal"]
    dream_usage = usage.set_index(["word", "topic"]).loc[("dream", "Love")]
    assert dream_usage["num_usages"] == 2

    word_topics = create_word_topics_table(generations, lemma_terms).set_index("word")
    assert json.loads(word_topics.loc["dream", "topics"]) == ["Love"]


def test_poem_imageability_writes_ordered_table_and_plot(tmp_path: Path) -> None:
    responses_path = tmp_path / "responses.csv"
    pd.DataFrame(
        [
            {
                "poem_id": "p1",
                "topic": "Love",
                "thinking_level": "minimal",
                "response": "Dreaming stars dream",
            }
        ]
    ).to_csv(responses_path, index=False)
    lexical_path = tmp_path / "lexical.json"
    lexical_path.write_text(
        json.dumps(
            {
                "batch_size": 8,
                "content_pos": ["NOUN", "VERB"],
                "excluded_terms": [],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "imageability.json"
    config_path.write_text(
        json.dumps(
            {
                "selection_seed": 0,
                "input_csv": str(responses_path),
                "lexical_config": str(lexical_path),
                "regression_config": str(tmp_path / "regression.json"),
                "regression_results": str(tmp_path / "regression_results"),
            }
        ),
        encoding="utf-8",
    )

    def make_token(text, lemma, pos):
        return SimpleNamespace(
            lower_=text,
            lemma_=lemma,
            pos_=pos,
            is_stop=False,
            is_alpha=True,
        )

    class FakeNlp:
        def pipe(self, responses, batch_size):
            assert list(responses) == ["Dreaming stars dream"]
            assert batch_size == 8
            return iter(
                [
                    [
                        make_token("dreaming", "dream", "VERB"),
                        make_token("stars", "star", "NOUN"),
                        make_token("dream", "dream", "VERB"),
                    ]
                ]
            )

    def fake_predictor(words, *args, **kwargs):
        del args, kwargs
        assert words == ["dream", "star"]
        return pd.DataFrame(
            {
                "word": words,
                "imagine_mean": [6.0, 7.5],
                "valence_mean": [4.5, 6.5],
            }
        )

    selected, values = analyze_poem_norm(
        config_path,
        seed=0,
        norm="valence",
        show_plot=False,
        nlp=FakeNlp(),
        predictor=fake_predictor,
    )

    assert selected["poem_id"] == "p1"
    assert values.to_dict("records") == [
        {"position": 1, "word": "dream", "valence": 4.5},
        {"position": 2, "word": "star", "valence": 6.5},
        {"position": 3, "word": "dream", "valence": 4.5},
    ]
    assert list(tmp_path.glob("*.csv")) == [responses_path]
    assert not list(tmp_path.glob("*.png"))


def test_lemma_topic_tables_count_poems_and_combine_topics() -> None:
    generations = pd.DataFrame(
        {
            "topic": ["Love", "Love", "Nature"],
            "thinking_level": ["minimal", "minimal", "high"],
        }
    )
    lemma_terms = [["dream", "dream"], ["dream", "star"], ["dream"]]

    usage = create_topic_usage_tables(generations, lemma_terms)
    minimal = usage["minimal"].set_index(["word", "topic"])
    high = usage["high"].set_index(["word", "topic"])
    topics = create_word_topics_table(generations, lemma_terms).set_index("word")

    assert list(usage["minimal"].columns) == ["word", "topic", "num_usages"]
    assert minimal.loc[("dream", "Love"), "num_usages"] == 2
    assert high.loc[("dream", "Nature"), "num_usages"] == 1
    assert json.loads(topics.loc["dream", "topics"]) == ["Love", "Nature"]

    combined, coverage = create_combined_topic_tables(usage)
    combined = combined.set_index(["word", "topic"])
    coverage = coverage.set_index(["word", "topic"])
    assert combined.loc[
        ("dream", "Love"), "num_usages_within_topic_generations"
    ] == 2
    assert combined.loc[
        ("dream", "Nature"), "num_usages_within_topic_generations"
    ] == 1
    assert coverage.loc[("dream", "Love"), "num_topics_covered"] == 2
    assert coverage.loc[("dream", "Nature"), "num_topics_covered"] == 2


def test_basic_summary_counts_unique_lemmas_by_topic(tmp_path: Path) -> None:
    input_path = tmp_path / "lemma_topic_sample_usage.csv"
    output_path = tmp_path / "num_unique_lemmas_by_topic.csv"
    pd.DataFrame(
        [
            {
                "word": "dream",
                "topic": "Love",
                "num_usages_within_topic_generations": 4,
            },
            {
                "word": "star",
                "topic": "Love",
                "num_usages_within_topic_generations": 2,
            },
            {
                "word": "dream",
                "topic": "Nature",
                "num_usages_within_topic_generations": 1,
            },
        ]
    ).to_csv(input_path, index=False)
    config_path = tmp_path / "lexical_analysis_config.json"
    config_path.write_text(
        json.dumps(
            {
                "lemma_topic_sample_usage_output": str(input_path),
                "num_unique_lemmas_by_topic_output": str(output_path),
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_basic_lemmas(config_path)

    assert summary.loc[0, "topic"] == "Love"
    assert summary.loc[0, "num_unique_lemmas"] == 2
    assert summary.loc[0, "entropy"] == pytest.approx(0.918295834)
    assert summary.loc[1, "topic"] == "Nature"
    assert summary.loc[1, "num_unique_lemmas"] == 1
    assert summary.loc[1, "entropy"] == 0.0
    assert pd.read_csv(output_path).loc[0, "entropy"] == pytest.approx(0.918296)


def test_vs5_comparison_resamples_equal_poem_counts(tmp_path: Path) -> None:
    basic_path = tmp_path / "basic.csv"
    vs5_path = tmp_path / "vs5.csv"
    output_path = tmp_path / "matched.csv"
    pd.DataFrame(
        [
            {"topic": "Love", "response": "sun sun"},
            {"topic": "Love", "response": "moon"},
        ]
    ).to_csv(basic_path, index=False)
    pd.DataFrame(
        [{"topic": "Love", "response": "single"} for _ in range(4)]
    ).to_csv(vs5_path, index=False)
    config_path = tmp_path / "lexical_analysis_config.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_size": 8,
                "content_pos": ["NOUN"],
                "excluded_terms": [],
                "matched_vs5_comparison": {
                    "basic_input_csv": str(basic_path),
                    "vs5_input_csv": str(vs5_path),
                    "sample_size": 2,
                    "num_resamples": 5,
                    "confidence_level": 0.95,
                    "random_seed": 0,
                    "output_csv": str(output_path),
                },
            }
        ),
        encoding="utf-8",
    )

    def make_token(word):
        return SimpleNamespace(
            lower_=word,
            lemma_=word,
            pos_="NOUN",
            is_stop=False,
            is_alpha=True,
        )

    class FakeNlp:
        def pipe(self, responses, batch_size):
            assert batch_size == 8
            for response in responses:
                words = response.split() if response != "single" else ["sun"]
                yield [make_token(word) for word in words]

    result = compare_vs5_lexical_diversity(config_path, nlp=FakeNlp())

    assert result.loc[0, "basic_num_unique_lemmas"] == 2
    assert result.loc[0, "vs5_mean_num_unique_lemmas"] == 1
    assert result.loc[0, "mean_num_unique_lemmas_difference"] == -1
    assert result.loc[0, "basic_entropy"] == pytest.approx(0.918295834)
    assert result.loc[0, "vs5_mean_entropy"] == 0.0
    assert result.loc[0, "mean_entropy_difference"] == pytest.approx(-0.918295834)
    assert pd.read_csv(output_path).loc[0, "basic_entropy"] == pytest.approx(
        0.918296
    )


def test_basic_lexical_browser_selects_topic_and_thinking(tmp_path: Path) -> None:
    minimal_path = tmp_path / "minimal.csv"
    high_path = tmp_path / "high.csv"
    combined_path = tmp_path / "combined.csv"
    pd.DataFrame(
        [
            {"word": "dream", "topic": "Love", "num_usages": 4},
            {"word": "star", "topic": "Love", "num_usages": 2},
        ]
    ).to_csv(minimal_path, index=False)
    pd.DataFrame(
        [
            {"word": "dream", "topic": "Love", "num_usages": 3},
            {"word": "moon", "topic": "Love", "num_usages": 1},
        ]
    ).to_csv(high_path, index=False)
    pd.DataFrame(
        [
            {
                "word": "dream",
                "topic": "Love",
                "num_usages_within_topic_generations": 7,
            },
            {
                "word": "star",
                "topic": "Love",
                "num_usages_within_topic_generations": 2,
            },
        ]
    ).to_csv(combined_path, index=False)
    config = {
        "thinking_tables": {
            "minimal": str(minimal_path),
            "high": str(high_path),
        },
        "combined_table": str(combined_path),
    }

    tables = load_usage_tables(config)
    minimal = select_topic_words(tables, "Love", "minimal", 10)
    combined = select_topic_words(tables, "Love", "combined", 10)

    assert get_basic_lexical_topics(tables) == ["Love"]
    assert minimal.to_dict("records")[0] == {
        "lemma": "dream",
        "count": 4,
        "decimal_frequency": 0.4,
    }
    assert combined.to_dict("records")[0] == {
        "lemma": "dream",
        "count": 7,
        "decimal_frequency": 0.35,
    }


def test_similarity_keeps_methods_separate_and_compares_thinking_levels() -> None:
    responses = pd.DataFrame(
        [
            {
                "poem_id": poem_id,
                "topic_id": "t1",
                "topic": "Love",
                "thinking_level": thinking,
            }
            for poem_id, thinking in (
                ("m1", "minimal"),
                ("m2", "minimal"),
                ("h1", "high"),
                ("h2", "high"),
            )
        ]
    )
    embeddings = {
        "m1": np.array([1.0, 0.0]),
        "m2": np.array([1.0, 0.0]),
        "h1": np.array([0.0, 1.0]),
        "h2": np.array([0.0, 1.0]),
    }

    within = compute_within_condition_similarity(responses, embeddings)
    cross = compute_cross_thinking_similarity(responses, embeddings)

    assert len(within) == 2
    assert set(within["similarity"]) == {1.0}
    assert len(cross) == 4
    assert set(cross["similarity"]) == {0.0}


def test_gemini_embeddings_apply_sentence_similarity_prefix(tmp_path) -> None:
    calls = []

    class FakeModels:
        def embed_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0, 0.0])]
            )

    client = SimpleNamespace(models=FakeModels())
    responses = pd.DataFrame(
        [
            {
                "generation_id": "g1",
                "response_id": "p1",
                "response_sha256": compute_stable_hash("A poem"),
                "response_input_kind": "poem",
                "response": "A poem",
            }
        ]
    )
    with EmbeddingCache(tmp_path / "embeddings.sqlite") as cache:
        embeddings = fetch_gemini_embeddings(
            responses,
            cache,
            client,
            model="gemini-embedding-2",
            dimensions=2,
            task_prefix="task: sentence similarity | query: {content}",
            batch_size=100,
        )

    embedded_text = calls[0]["contents"][0].parts[0].text
    assert embedded_text == "task: sentence similarity | query: A poem"
    assert embeddings["p1"].tolist() == [1.0, 0.0]


def write_prepared_poem(tmp_path) -> Path:
    data_dir = tmp_path / "results/data/poetry_basic"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "poem_id": "poem-1",
                "generation_id": "gen-1",
                "method_id": "poetry_basic",
                "topic_id": "topic-1",
                "topic": "Love",
                "thinking_level": "minimal",
                "source_sample_idx": 1,
                "response_idx": 1,
                "response": "Call me at 555-0100.",
            }
        ]
    ).to_csv(data_dir / "responses.csv", index=False)
    return data_dir


def make_pii_args(tmp_path, config_path=None) -> Namespace:
    return Namespace(
        method="poetry_basic",
        config=str(
            config_path
            or "experiments/multimodal_thinking_experiment/pii_judge_config.json"
        ),
        runtime_root=str(tmp_path / "runtime"),
        results_root=str(tmp_path / "results"),
        confirm_api=False,
    )


def test_prepare_pii_requests_uses_approved_prompt_and_json_schema(tmp_path) -> None:
    write_prepared_poem(tmp_path)
    args = make_pii_args(tmp_path)

    prepare_pii_requests(args)
    validate_pii_requests(args)

    request_path = tmp_path / "runtime/poetry_basic/pii_requests.jsonl"
    request = read_requests(request_path)["pii--poem-1"]
    prompt = request["contents"][0]["parts"][0]["text"]
    generation_config = request["generationConfig"]
    assert prompt.endswith("Poem:\nCall me at 555-0100.\n")
    assert generation_config["maxOutputTokens"] == 512
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["thinkingConfig"] == {
        "includeThoughts": False,
        "thinkingLevel": "MINIMAL",
    }
    assert generation_config["responseSchema"]["required"] == [
        "count",
        "evidence_phrases",
    ]
    assert not {"temperature", "topP", "topK"} & set(generation_config)


def test_pii_submission_requires_confirmation_and_user_flag(tmp_path) -> None:
    write_prepared_poem(tmp_path)
    config_source = Path(
        "experiments/multimodal_thinking_experiment/pii_judge_config.json"
    )
    source_config = json.loads(config_source.read_text(encoding="utf-8"))
    source_config["user_approved_for_api"] = False
    config_path = tmp_path / "pii_config.json"
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    args = make_pii_args(tmp_path, config_path)
    prepare_pii_requests(args)

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_pii_requests(args)
    args.confirm_api = True
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_pii_requests(args)


def test_finalize_pii_results_materializes_count_and_evidence(tmp_path) -> None:
    data_dir = write_prepared_poem(tmp_path)
    args = make_pii_args(tmp_path)
    prepare_pii_requests(args)
    runtime_dir = tmp_path / "runtime/poetry_basic"
    result = {
        "key": "pii--poem-1",
        "response": {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "count": 1,
                                        "evidence_phrases": ["555-0100"],
                                    }
                                )
                            }
                        ]
                    },
                }
            ],
            "modelVersion": "gemini-3.1-flash-lite-001",
            "usageMetadata": {
                "promptTokenCount": 80,
                "candidatesTokenCount": 15,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 100,
            },
        },
    }
    (runtime_dir / "pii_results.jsonl").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )

    finalize_pii_results(args)

    judgments = pd.read_csv(data_dir / "pii_judgments.csv")
    assert judgments.loc[0, "count"] == 1
    assert json.loads(judgments.loc[0, "evidence_phrases"]) == ["555-0100"]
    assert judgments.loc[0, "thought_token_count"] == 5


def test_parse_pii_judgment_rejects_extra_fields() -> None:
    assert parse_judgment('{"count": 0, "evidence_phrases": []}') == (0, [])
    with pytest.raises(ValueError, match="only count"):
        parse_judgment('{"count": 0, "evidence_phrases": [], "extra": true}')
    with pytest.raises(ValueError, match="must equal"):
        parse_judgment('{"count": 1, "evidence_phrases": []}')
