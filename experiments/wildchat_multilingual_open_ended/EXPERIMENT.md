# WildChat Multilingual Open-Ended Classification

## Purpose

Apply an LLM judge classifier to translated WildChat queries and surface selected queries for
further analysis.

## Inputs

Read these pilot outputs without copying them into this experiment's result directory:

- `results/wildchat_multilingual_pilot/data/Spanish.csv`
- `results/wildchat_multilingual_pilot/data/Chinese.csv`

Classify the value in the `untranslated` column. Do not use the English translation.
Submit each exact `untranslated` value once across both files. Each output CSV contains one row per
distinct query in its corresponding input file. When the same query occurs in both files, reuse its
single classifier result in both outputs.

Expected pilot columns:

- `turn_idx`
- `untranslated`
- `role`
- `translated`

## Outputs

- One raw JSON file per submitted query:
  `results/wildchat_multilingual_open_ended/data/classifier_outputs`
- One CSV per input file in `results/wildchat_multilingual_open_ended/data`, using the
  input filename with `_classifier_output` appended
- Aggregate summaries: `results/wildchat_multilingual_open_ended/tables`
- Figures, if later required: `results/wildchat_multilingual_open_ended/plots`

Each output CSV has these columns:

- `original_query`
- `revised_query`
- `meaningful_information`
- `greeting_model_inquiry`
- `response_type`
- `language`

Accept the classifier's `original_query` value even when it differs from the submitted query.
Exclude responses that cannot be parsed into the required JSON object or contain no usable model
text. Record all excluded responses in `data/failed_requests.csv` and materialize the remaining
responses.

Use the Gemini Batch API with `gemini-3.8-flash` and one response per query. The approved prompt is
stored in `judge_prompt.txt`. API submission also requires `user_approved_for_api` in
`generation_config.json` and the explicit `--confirm-api` command flag.

## Execution

The Google Gen AI SDK reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the environment. Do not store
the key in this repository.

Prepare and validate the requests locally:

```bash
uv run python -m experiments.wildchat_multilingual_open_ended.generate_responses prepare
uv run python -m experiments.wildchat_multilingual_open_ended.generate_responses validate
```

Commands that contact Gemini require `--confirm-api`:

```bash
uv run python -m experiments.wildchat_multilingual_open_ended.generate_responses submit --confirm-api
uv run python -m experiments.wildchat_multilingual_open_ended.generate_responses status --confirm-api
uv run python -m experiments.wildchat_multilingual_open_ended.generate_responses download --confirm-api
```

Materialize per-query JSON files and classifier CSV files locally:

```bash
uv run python -m experiments.wildchat_multilingual_open_ended.generate_responses finalize
```
