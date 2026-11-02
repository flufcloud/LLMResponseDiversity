# Templated Prompts Compositionality

This experiment applies 15 adjectives from five semantic groups to the existing 203 Poetry
Foundation topic strings. It requests 10 poems for every adjective-topic pair, for 30,450 Gemini
Batch API requests.

The exact prompt is `Write a(n) {ADJECTIVE} poem about {TOPIC}.` The text model is
`gemini-3.1-flash-lite` with minimal thinking, no returned thoughts, and a 2,048-token output cap.
Temperature, top-p, and top-k are unset.

Run from the repository root:

```bash
uv run --frozen python -m experiments.templated_prompts_compositionality.generate_responses prepare
uv run --frozen python -m experiments.templated_prompts_compositionality.generate_responses validate
```

Review `generation_config.json` and
`.cache/experiments/templated_prompts_compositionality/generation_manifest.csv`. Only the user may
then change `user_approved_for_api` to `true`.

The following commands contact Gemini. Submission is a paid action:

```bash
uv run --frozen python -m experiments.templated_prompts_compositionality.generate_responses submit --confirm-api
uv run --frozen python -m experiments.templated_prompts_compositionality.generate_responses status --confirm-api
uv run --frozen python -m experiments.templated_prompts_compositionality.generate_responses download --confirm-api
```

After download, materialize results locally:

```bash
uv run --frozen python -m experiments.templated_prompts_compositionality.generate_responses finalize
```

Responses are written to
`results/templated_prompts_compositionality/data/generations.csv`. Provider failures are written to
`results/templated_prompts_compositionality/data/failed_requests.csv`.

After materializing responses, compute the pruned lemma-frequency tables locally:

```bash
uv run --frozen python -m experiments.templated_prompts_compositionality.analyze_words
```

This writes:

- `tables/lemma_frequency_by_topic.csv`
- `tables/lemma_frequency_by_topic_adjective_class.csv`
- `tables/lemma_frequency_by_topic_adjective.csv`

Each table records total lemma occurrences and the fraction of poems in that grouping containing
the lemma. The analysis retains nouns, proper nouns, verbs, adjectives, and adverbs after removing
stop words and explicit negation terms.

Create the unique-lemma count tables from the existing lexical tables:

```bash
uv run --frozen python -m \
  experiments.templated_prompts_compositionality.summarize_lemmas
```

This writes topic, topic-adjective-class, and topic-adjective summaries under the experiment's
`tables/` directory. Each table includes Shannon entropy in bits, calculated from the lemma
occurrence-count distribution within each group.

Compare the basic and adjective-conditioned corpora using equal poem counts:

```bash
uv run --frozen python -m \
  experiments.templated_prompts_compositionality.compare_topic_lexical_diversity
```

For each topic, this uses all 20 basic poems and 1,000 random samples of 20 poems from the
adjective-conditioned corpus. It writes the basic statistics, adjective-sample means, 95%
intervals, and adjective-minus-basic differences to
`tables/matched_basic_comparison_by_topic.csv`.

Open an interactive lemma-usage plot at any supported grouping level:

```bash
uv run --frozen python -m \
  experiments.templated_prompts_compositionality.plot_word_frequencies \
  --topic "Love"

uv run --frozen python -m \
  experiments.templated_prompts_compositionality.plot_word_frequencies \
  --topic "Love" --adjective-class happy

uv run --frozen python -m \
  experiments.templated_prompts_compositionality.plot_word_frequencies \
  --topic "Love" --adjective-class happy --adjective joyful
```

The plot assigns one x-position to every lemma, ranks the lemmas by total occurrences, and hides
individual x-axis labels. The y-axis reports total occurrences. Highlighted bars identify common,
typical, and uncommon example lemmas. The figure opens without being saved.

Open a table containing the 10 most common lemmas with the same selection arguments:

```bash
uv run --frozen python -m \
  experiments.templated_prompts_compositionality.show_top_words \
  --topic "Love"

uv run --frozen python -m \
  experiments.templated_prompts_compositionality.show_top_words \
  --topic "Love" --adjective-class happy

uv run --frozen python -m \
  experiments.templated_prompts_compositionality.show_top_words \
  --topic "Love" --adjective-class happy --adjective joyful
```

The table reports each lemma's rank, total occurrences, and fraction of poems containing it. It
opens without being saved.

Both commands open the same viewer. Use the topic, adjective class, adjective, and view selectors
to update the current window. The original command arguments set the initial selection. The viewer
can also be opened directly:

```bash
uv run --frozen python -m \
  experiments.templated_prompts_compositionality.browse_word_frequencies \
  --topic "Love" --view plot
```
