# Line-level syntactic-form analysis

The earlier sentence-level results are stale. They allowed Stanza to join poetic lines across line
breaks and do not measure the intended unit.

The corrected pipeline performs these steps:

1. Split each poem at nonempty poetic lines.
2. Parse every line independently.
3. Build one dependency-feature representation for the complete line.
4. Compare, cluster, and accumulate line-level syntactic forms.

If a single poetic line contains multiple grammatical sentences, their dependency-feature counts
are combined into one line representation. A dependency graph never crosses a poetic line break.

Run the corrected pipeline before reporting syntactic-form findings. The example selector restricts
each line pair to one topic. The new row-level outputs are:

- `data/line_features.jsonl`
- `data/line_form_assignments.csv`
- `tables/line_similarity_summary.csv`

The accumulation and distribution filenames are unchanged, but rerunning the analysis overwrites
them with line-level results.
