# Project Context

## Purpose

This repository reproduces the intra-model and inter-model response-similarity analysis from the
Artificial Hivemind paper and extends it to English, Telugu, and Yoruba.

## Safety

- Never access credentials, inspect API-key values, submit jobs, or call an external API without
  the user's explicit instruction for that specific action.
- Before any action that can charge a credit card, explicitly disclose that it may incur charges
  and obtain approval for that specific paid action.
- Local preparation, validation, tests, and plotting do not imply authorization for API calls.
- Do not introduce or change model-facing prompts, system instructions, or other behavior-shaping
  request fields without first discussing the exact wording and experimental rationale with the
  user and receiving explicit approval.
- Every Gemini generation experiment must contain an experiment-local generation configuration as
  the authoritative record of models, sample count, system instructions, and generation config. Do
  not add code defaults or command-line overrides for those fields.
- Only the user may change `user_approved_for_api` in a generation config from `false` to
  `true`. Agents must never infer approval or make that change.
- Gemini commands that contact the service require `--confirm-api`.
- Translation must run locally and must not call a hosted translation API or service.
- Translation experiment settings must be explicit in the experiment-local configuration.

## Data

- Assume pipeline inputs are reasonably formatted, non-empty, produced by the expected dataset, and
  consumed in the documented stage order. Do not add defensive validation for expected columns,
  row counts, duplicate rows, query coverage, language coverage, model coverage, or sample counts.
- Retain checks only at external boundaries: explicit API authorization and exact approved request
  configuration, provider response integrity, model output shape and token limits, and embedding
  cache integrity needed for safe resumption.
- Store every experiment under `results/experiments/<experiment_name>`, with `data/`, `tables/`,
  and `plots/` subdirectories. Do not create a repository-level `data/`, `results/analysis/`, or
  `results/plots/` directory.
- Keep `results/` ignored by Git. Experiment code and configuration must reproduce its contents.
- Do not store CSV files under `experiments/`; source datasets must be loaded from their configured
  upstream source, and generated CSV files belong under `results/` or ignored `.cache/` checkpoints.
- `data/` contains raw model responses, translations, embedding databases, normalized row-level
  data, and row-level similarity scores. `tables/` contains grouped scores, means, summaries,
  matrices, and validation tables. `plots/` contains figures.
- Each experiment result root contains one `metadata.json`. It records only `experiment_id` and
  `embedding_model`, plus generation `max_token_length` when the experiment generates responses.
  Embedding-only experiments omit `max_token_length`. Do not create analysis manifests.
- `experiment_id` must match the directory name under both `results/experiments/` and
  `.cache/experiments/` so it identifies the experiment's disposable runtime artifacts.
- The multilingual generations are stored at
  `results/experiments/en_te_yo_30_query/data/generations.csv`; primary multilingual analysis reads
  only this file. Its provenance is
  `results/experiments/en_te_yo_30_query/data/generations_metadata.csv`, joined only by
  `generation_id`.
- `generation_id` is unique in both files; neither file uses unnamed or positional indexes.
- Each canonical row includes its base query, optional translated query, original model response,
  optional English response translation, provider, model version, configuration, and query tag.
- Base-language rows leave every query-translation column null.
- Translations of the same base query share one stable `query_id`; `q003` is the time-metaphor
  query.
- Raw requests, manifests, jobs, and checkpoints are disposable implementation artifacts. Keep
  them under ignored `.cache/experiments`. Raw model responses are experimental data and belong
  under the experiment's `results/experiments/<experiment_name>/data`; provider transport files do
  not. Experiment configurations are tracked with their code under `experiments/`.
- Call resumable progress files checkpoints, not caches. Use "cache" only for reusable derived data,
  such as embeddings keyed for reuse.
- `english_10_query` contains 10 English prompts, 4 Gemini models, and 10 responses per group, for
  400 responses total.
- `en_te_yo_30_query` contains 30 prompts, 3 languages, 2 Gemini models, and 30 responses per
  group, for 5,400 responses total.
- `results/experiments/en_te_yo_30_query/data/generations.csv` contains the 5,400 multilingual
  responses only.

## Embeddings and Analysis

- Multilingual analysis uses dense normalized `BAAI/bge-m3` embeddings at revision
  `5617a9f61b028005a4858fdac845db406aefb181`.
- Default runtime is MPS float32, dimension 1,024, maximum sequence length 8,192, batch size 16.
- Store each experiment's active embedding database under its
  `results/experiments/<experiment_name>/data` directory. Embeddings are data, even when they are
  reproducible and ignored by Git; do not place them under `.cache/` or `tables/`.
- The embedding database links vectors to `generation_id` and input kind while retaining internal
  text hashes for validation and reuse.
- Run within-language analyses independently for `en`, `yo`, and `te`; do not aggregate languages.
- Cross-language analysis holds source `query_id`, Gemini model, and generation configuration
  fixed.
- Cross-language pairs are English–Telugu, English–Yoruba, and Telugu–Yoruba.
- Back-translated analysis first converts Telugu and Yoruba responses to English with the pinned
  local NLLB model. Its disposable resumable checkpoint is under
  `.cache/experiments/en_te_yo_30_query`; materialized translations are stored directly in the
  experiment's generation row.
- Multilingual data, tables, and plots live under
  `results/experiments/en_te_yo_30_query/{data,tables,plots}`. Language-specific outputs use
  `{en,yo,te,cross_language}` subdirectories within those categories.
- Back-translated outputs use the `translated_cross_language` subdirectory within each category
  and must only consume the materialized translations during embedding and plotting.
- PCA figures use `q003`, the time-metaphor query.
- Cross-language and back-translated outputs each include two aggregate response-level PCA
  variants: model colors with language marker shapes, and language colors with model marker shapes.
  Each has six model-language point types.
- Generate figures as PNG files only; do not create PDF copies.

## Cross-Lingual Embedding Validation

- Use all 100 rows of `liweijiang/infinite-chats-eval` in source order, with stable IDs `q001`
  through `q100`; do not sample or exclude rows.
- Translation settings live in
  `experiments/cross_lingual_embedding_validation/translation_config.json`.
- Embedding settings live in
  `experiments/cross_lingual_embedding_validation/embedding_config.json` and use the pinned
  `BAAI/bge-m3` revision specified there.
- Load the source queries directly from the configured Infinite Chats dataset; do not maintain a
  separate tracked `queries.csv`. Raw translations and embeddings are stored under
  `results/experiments/cross_lingual_embedding_validation/data`. The resumable translation
  checkpoint remains under ignored
  `.cache/experiments/cross_lingual_embedding_validation`.
- `raw_multilingual_similarity.csv` is a row-level score file and belongs in the experiment's
  `data/` directory. `tables/` contains `cross_lingual_mean_similarity_by_query.csv`,
  `cross_lingual_mean_similarity_by_lang_pair.csv`, and
  `cross_lingual_mean_similarity_by_lang.csv`.
- Sort `cross_lingual_mean_similarity_by_lang_pair.csv` and
  `cross_lingual_mean_similarity_by_lang.csv` from highest to lowest mean similarity.

## Intralingual Multilingual Embedding Validation

- Load `nyu-mll/glue` MRPC training rows directly and select only label `1` paraphrase pairs.
  Sample 1,000 pairs with random seed `0`.
- Translation and embedding settings live in
  `experiments/intralingual_multilingual_embedding_validation/`
  `{translation_config,embedding_config}.json`.
  Use the pinned local NLLB model for the 19 non-English languages and the pinned `BAAI/bge-m3`
  configuration for all 20 languages.
- Store `mrpc_translated.csv` and `embeddings.sqlite` under the experiment's `data/` directory.
  `mrpc_translated.csv` contains one row per sampled MRPC pair and language, with a unique `index`
  and the source MRPC `idx` recorded as `mrpc_sentence_id`.
- Store `intralingual_embedding_scores_raw.csv` and
  `intralingual_embedding_scores_mean.csv` under the experiment's `tables/` directory. Also store
  `intralingual_embedding_scores_mean_by_query.csv`, which averages all language scores for each
  `mrpc_sentence_id`.

## WildChat Multilingual Pilot

- Process source rows `0` through `99,999` from `allenai/WildChat-4.8M` in source order.
- Translate only the initial query. Do not translate model-response turns.
- Include Spanish, Chinese, Telugu, and Yoruba, with separate output files for each language.
- Translate to English locally with `Helsinki-NLP/opus-mt-mul-en` at revision
  `848eae0c1676cfce9bb791c200e8228e5a6396ff`. Do not call a hosted translation service.
- Use `translation_config.json` for MPS and `translation_config_cuda.json` for CUDA. The CUDA
  configuration uses float16 inference and a larger model batch.

## Code Organization

- Store experiment-specific scripts, configurations, tracked inputs, and tests together under
  `experiments/<experiment_name>` so the complete procedure can be reviewed in one directory.
- Keep `english_10_query` and `en_te_yo_30_query` self-contained. Do not introduce a shared or
  common Hivemind code package; duplicate the small amount of supporting code when necessary.
- Keep experiment scripts directly in the experiment root; `tests/` is the only code subdirectory.
  Use stage-oriented entry points named `generate_responses.py`, `embed_data.py`, `analyze.py`, and
  `create_plots.py` when those stages apply. Keep explicitly named translation scripts in the root.
- Do not create empty `__init__.py` files. Namespace-package imports work from repository-root
  module execution.
- Experiment reruns must separate stages: generation writes responses, embedding populates the
  embedding database, analysis reads existing embeddings and writes tables, and plotting reads the
  analysis tables and existing embeddings. Analysis must never call an embedding provider or model.
- Keep `lib/` limited to reusable primitives shared by multiple experiments. Do not place CLI entry
  points, experiment defaults, or experiment orchestration in `lib/`.
- Keep local translation model loading, batching, inference, and output validation in
  `lib/translate_locally.py`; experiment directories retain only experiment-specific translation
  orchestration, checkpoint schemas, and materialization.
- Keep provider-level Gemini request construction, batch submission, status checks, downloads, and
  response parsing in `lib/generate_with_gemini.py`; experiment directories retain
  experiment-specific configuration, request manifests, orchestration, and canonical
  materialization.
- Keep embedding persistence, provider adapters, and reusable model wrappers under `lib/`;
  experiment directories retain only experiment-specific model selection, validation, analysis,
  and plotting.
- Keep shared response normalization in `lib/prepare_responses.py`, shared similarity calculations
  in `lib/compute_similarity.py`, and shared hashes and stable identifiers in
  `lib/compute_identifiers.py`. Do not duplicate these APIs inside experiment directories.
- Name functions with a verb that states what they do, such as `get_batch`; avoid noun-only
  function names. Properties and required protocol methods are exempt.
- Name Python files with a verb that states what the file does; do not use noun-only filenames such
  as `plots.py`, `records.py`, `responses.py`, `similarity.py`, `paths.py`, or `utils.py`.
- Store ignored runtime artifacts under `.cache/experiments/<experiment_name>`. Store experiment
  outputs only under `results/experiments/<experiment_name>/{data,tables,plots}` according to the
  raw-versus-aggregate distinction above.

## Verification

- This is a uv-managed project. Run Python scripts, tests, and tools through `uv run`; do not
  recommend or invoke `.venv/bin/python`, `.venv/bin/pytest`, or other `.venv/bin/*` executables
  directly.
- Run experiment entry points as repository modules from the repository root, using
  `uv run python -m experiments.<module>`. Do not recommend `uv run python experiments/...py`;
  direct file execution does not provide the repository root required by namespace-package
  imports.
- Use `uv sync --frozen` when setting up the locked environment reproducibly, and use
  `uv run --frozen` when a command must not update the lockfile.
- Run tests with `uv run pytest -q`.
- Do not download BGE-M3 or run real embedding inference during routine tests; tests use fake local
  encoders.
- After coding, perform a compactness, readability, and style review.

All `README.md` files are off limits to agents unless the user explicitly asks for a README edit.
This includes every README under `experiments/`. Use a short experiment-local `EXPERIMENT.md` for
agent-facing protocol notes when needed.
