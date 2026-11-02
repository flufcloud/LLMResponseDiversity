# Adjective-poetry analysis

This directory contains the unsupervised extension of the basic-poetry analysis to the templated
adjective corpus. It leaves the source generations unchanged.

## Aggregation levels

- Corpus: deterministic random poem pairs across the full corpus.
- Topic: poem pairs within each topic; results report the mean across topics.
- Adjective class: poem pairs within each topic and adjective class; results report the mean across
  those groups.
- Adjective: poem pairs within each topic and adjective; results report the mean across those
  groups.

The pairwise metrics are Gemini embedding cosine similarity, lexical Jaccard similarity over
pruned content lemmas, and dependency-tree syntactic similarity. Pair sampling is bounded by
`analysis_config.json` to control runtime.

The form analysis measures rhyme class, rhymed-line fraction, estimated iambic-pentameter usage,
line count, stanza count, stanza length, and four-line-stanza usage.

## Output layout

- `data/`: normalized poems, embedding database, text features, pair scores, and poem-level form
  measurements.
- `tables/`: group summaries, distributions, comparison tables, and
  `homogeneity_comparisons.csv`.

`line_text_features.jsonl` is resumable. Each nonempty poetic line is parsed independently, so no
dependency graph crosses a line boundary. Existing rows are checked against their response hashes
before new poems are appended.
