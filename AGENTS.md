# Project Context

## Purpose

This repository reproduces the intra-model and inter-model response-similarity analysis from the
Artificial Hivemind paper and extends it to English, Telugu, and Yoruba.

## Safety

- Agents must never execute experiment jobs, including local model inference. Prepare and verify
  code and configuration, then provide the exact terminal commands for the user to run.
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
- Store every experiment under `results/<experiment_name>`, with `data/`, `tables/`,
  and `plots/` subdirectories. Do not create a repository-level `data/`, `results/analysis/`, or
  `results/plots/` directory.
- Keep `results/` ignored by Git. Experiment code and configuration must reproduce its contents.
- Do not store CSV files under `experiments/`; source datasets must be loaded from their configured
  upstream source, and generated CSV files belong under `results/` or ignored `.cache/` checkpoints.
- `data/` contains raw model responses, translations, embedding databases, normalized row-level
  data, and row-level similarity scores. `tables/` contains grouped scores, means, summaries,
  matrices, and validation tables. `plots/` contains figures.
- Each experiment result root contains one `metadata.json`. Embedding validation instead contains
  one in each model-specific result directory. It records only `experiment_id` and
  `embedding_model`, plus generation `max_token_length` when the experiment generates responses.
  Embedding-only experiments omit `max_token_length`. Do not create analysis manifests.
- `experiment_id` must match the directory name under both `results/` and `.cache/experiments/` so
  it identifies the experiment's disposable runtime artifacts. Embedding validation uses the
  grouped model-specific layout described below.
- The multilingual generations are stored at
  `results/en_te_yo_30_query/data/generations.csv`; primary multilingual analysis reads
  only this file. Its provenance is
  `results/en_te_yo_30_query/data/generations_metadata.csv`, joined only by
  `generation_id`.
- `generation_id` is unique in both files; neither file uses unnamed or positional indexes.
- Each canonical row includes its base query, optional translated query, original model response,
  optional English response translation, provider, model version, configuration, and query tag.
- Base-language rows leave every query-translation column null.
- Translations of the same base query share one stable `query_id`; `q003` is the time-metaphor
  query.
- Raw requests, manifests, jobs, and checkpoints are disposable implementation artifacts. Keep
  them under ignored `.cache/experiments`. Raw model responses are experimental data and belong
  under the experiment's `results/<experiment_name>/data`; provider transport files do
  not. Experiment configurations are tracked with their code under `experiments/`.
- Call resumable progress files checkpoints, not caches. Use "cache" only for reusable derived data,
  such as embeddings keyed for reuse.
- `english_10_query` contains 10 English prompts, 4 Gemini models, and 10 responses per group, for
  400 responses total.
- `en_te_yo_30_query` contains 30 prompts, 3 languages, 2 Gemini models, and 30 responses per
  group, for 5,400 responses total.
- `results/en_te_yo_30_query/data/generations.csv` contains the 5,400 multilingual
  responses only.

## Embeddings and Analysis

- Multilingual analysis uses dense normalized `BAAI/bge-m3` embeddings at revision
  `5617a9f61b028005a4858fdac845db406aefb181`.
- Default runtime is MPS float32, dimension 1,024, maximum sequence length 8,192, batch size 16.
- Store each experiment's active embedding database under its
  `results/<experiment_name>/data` directory. Embeddings are data, even when they are
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
  `results/en_te_yo_30_query/{data,tables,plots}`. Language-specific outputs use
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
  `experiments/embedding_validation/cross_lingual/translation_config.json`.
- Model-specific embedding settings live in `experiments/embedding_validation/cross_lingual/`.
  Use each pinned model revision and keep model outputs separate.
- Harrier validation uses `microsoft/harrier-oss-v1-0.6b` at revision
  `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`, with the approved `sts_query` instruction
  `Instruct: Retrieve semantically similar text\nQuery: `.
- Load the source queries directly from the configured Infinite Chats dataset; do not maintain a
  separate tracked `queries.csv`. Raw translations and embeddings are stored under
  `results/embedding_validation/cross_lingual/data`. Model-specific embeddings, row-level scores,
  tables, and metadata live under `results/embedding_validation/cross_lingual/models/<model_name>`.
  The resumable translation
  checkpoint remains under ignored
  `.cache/experiments/cross_lingual_embedding_validation`.
- `raw_multilingual_similarity.csv` is a row-level score file and belongs in the model's `data/`
  directory. The model's `tables/` contains `cross_lingual_mean_similarity_by_query.csv`,
  `cross_lingual_mean_similarity_by_lang_pair.csv`, and
  `cross_lingual_mean_similarity_by_lang.csv`.
- Sort `cross_lingual_mean_similarity_by_lang_pair.csv` and
  `cross_lingual_mean_similarity_by_lang.csv` from highest to lowest mean similarity.

## Intralingual Multilingual Embedding Validation

- Load `nyu-mll/glue` MRPC training rows directly and select only label `1` paraphrase pairs.
  Sample 1,000 pairs with random seed `0`.
- Translation and model-specific embedding settings live in
  `experiments/embedding_validation/intralingual/`.
  Use the pinned local NLLB model for the 19 non-English languages and each pinned embedding model
  configuration for all 20 languages.
- Store `mrpc_translated.csv` under `results/embedding_validation/intralingual/data` and each
  `embeddings.sqlite` under the corresponding model's `data/` directory.
  `mrpc_translated.csv` contains one row per sampled MRPC pair and language, with a unique `index`
  and the source MRPC `idx` recorded as `mrpc_sentence_id`.
- Store `intralingual_embedding_scores_raw.csv` and
  `intralingual_embedding_scores_mean.csv` under the model's `tables/` directory. Also store
  `intralingual_embedding_scores_mean_by_query.csv`, which averages all language scores for each
  `mrpc_sentence_id`.

## WildChat Multilingual Pilot

- Use `wildchat_multilingual_pilot` as the experiment ID under `experiments/`, `.cache/experiments/`,
  and `results/`.
- Process source rows `0` through `99,999` from `allenai/WildChat-4.8M` in source order.
- Translate only the initial query. Do not translate model-response turns.
- Include Spanish, Chinese, Telugu, and Yoruba, with separate output files for each language.
- Translate to English locally with `Helsinki-NLP/opus-mt-mul-en` at revision
  `848eae0c1676cfce9bb791c200e8228e5a6396ff`. Do not call a hosted translation service.
- Use `translation_config.json` for MPS and `translation_config_cuda.json` for CUDA. The CUDA
  configuration uses float16 inference and a larger model batch.

## WildChat Multilingual Open-Ended Classification

- Use `wildchat_multilingual_open_ended` as the experiment ID under `experiments/`,
  `.cache/experiments/`, and `results/`.
- Read `Spanish.csv` and `Chinese.csv` from
  `results/wildchat_multilingual_pilot/data`; do not duplicate the pilot data. Classify
  the original-language query in `untranslated`, not the English translation.
- Submit each exact `untranslated` value once across both input files. Retain one output row per
  distinct query per source file and reuse a classifier result for cross-file duplicates.
- Store row-level classifier outputs under the experiment's `data/` directory and aggregate
  summaries under its `tables/` directory.
- Accept classifier `original_query` differences. Exclude responses with invalid classifier JSON
  or no usable model text, record them in `data/failed_requests.csv`, and materialize all remaining
  responses.
- Use one Gemini Batch API response per query with `gemini-3.8-flash`. Keep the approved judge
  prompt, model, and generation settings under the experiment directory.
- Do not submit judge requests until the user approves the exact prompt and request configuration.

## Test-Time Diversity Scaling

- Use `100_ideas_scaling` as the experiment ID under `experiments/`, `.cache/experiments/`, and
  `results/`.
- Use `gemini-3.8-flash` for all 10 prompts and request 10 responses per prompt.
- Append the approved 100-ideas instruction to each base prompt, separated by one blank line.
- Store row-level responses in `results/100_ideas_scaling/data/generations.csv`.

## Templated Prompts Compositionality

- Use `templated_prompts_compositionality` as the experiment ID under `experiments/`,
  `.cache/experiments/`, and `results/`.
- Reuse the Poetry Foundation hierarchy from the multimodal thinking experiment, selecting every
  leaf topic and each second-level `SUBJECTS` topic. Submit each distinct topic string once per
  adjective and sample while retaining every hierarchy path.
- Use the exact template `Write a(n) {ADJECTIVE} poem about {TOPIC}.` with the 15 approved
  adjectives grouped under happy, sad, unique, aesthetic, and ugly.
- Generate 10 responses for every adjective-topic pair with `gemini-3.1-flash-lite` through the
  Gemini Batch API. Use minimal thinking, do not return thoughts, cap output at 2,048 tokens, and
  leave temperature, top-p, and top-k unset.
- Store materialized responses in
  `results/templated_prompts_compositionality/data/generations.csv`, including adjective group,
  adjective, topic hierarchy paths, model version, finish reason, and token counts.
- For lexical analysis, retain the existing pruned content POS set, remove stop words and explicit
  negations, and lemmatize tokens. Store lemma count and poem-level decimal frequency at the topic,
  topic-adjective-class, and topic-adjective levels under the experiment's `tables/` directory.
- The lexical frequency plot accepts a required topic and optional adjective class and adjective,
  selects the matching aggregation level, and opens a ranked lemma-occurrence bar chart without
  saving it. Every lemma receives an unlabeled x-position; the y-axis is its total usage count.
- The top-words table accepts the same selection arguments and opens the 10 highest-count lemmas
  with rank, total occurrences, and poem-level decimal frequency without saving it.
- Both lexical views share one minimal interactive window with topic, adjective class, adjective,
  and plot-or-table selectors. Changing a selector updates the open window.
- Store unique-lemma counts in `num_unique_lemmas_by_topic.csv`,
  `num_unique_lemmas_by_topic_adjective_class.csv`, and
  `num_unique_lemmas_by_topic_adjective.csv`. The last table groups by topic and adjective without
  repeating adjective class. Append Shannon entropy in bits using each group's lemma occurrence
  counts as weights.
- Compare overall topic-level lexical diversity against the basic-poetry corpus using all 20 basic
  poems and 1,000 random 20-poem samples from the adjective corpus for each topic. Use random seed
  `0`, report 95% intervals and adjective-minus-basic differences, and compute both corpora from
  retained lemma occurrence counts.

## Multimodal Thinking Experiment

- Use `multimodal_thinking_experiment` as the experiment ID under `experiments/`,
  `.cache/experiments/`, and `results/`.
- Store the Poetry Foundation hierarchy in
  `experiments/multimodal_thinking_experiment/data/poetry_topics.json`.
- Generate text for every leaf topic and each second-level `SUBJECTS` topic. Submit each distinct
  topic string once per condition and sample while retaining every hierarchy path in the manifest.
- Use the exact prompt `Write a poem on the following topic: {TOPIC}`.
- Use `gemini-3.1-flash-lite` through the Gemini Batch API with 10 responses per topic at both
  `minimal` and `high` thinking levels. Request thought summaries and use a maximum output length
  of 16,384 tokens. Leave temperature, top-p, and top-k unset.
- Store materialized poems, thought summaries, returned model versions, finish reasons, and token
  counts in
  `results/multimodal_thinking_experiment/data/poetry_basic_generations.csv`.
- Use the exact VS5 prompt
  `Write a poem about {TOPIC}. Generate 5 responses and their corresponding probabilities. Respond in JSON format.`
  and store its rows in
  `results/multimodal_thinking_experiment/data/poetry_vs5_generations.csv`.
- Use the exact SSOT prompt
  `Generate a random string, then manipulate it to generate a diverse response. Write a poem about {TOPIC}.`
  and store its rows in
  `results/multimodal_thinking_experiment/data/poetry_ssot_generations.csv`.
- Keep separate request manifests, batch jobs, downloads, and failure tables for `poetry_basic`,
  `poetry_vs5`, and `poetry_ssot` under the experiment runtime and result directories.
- Run poetry analysis separately for `poetry_basic`, `poetry_ssot`, and `poetry_vs5`, using
  method-specific directories under each result category.
- Treat all five valid poems in each VS5 response as separate samples. Normalize slash separators
  to line breaks, retain the verbalized probability, and record malformed five-poem outputs as
  extraction failures.
- Extract SSOT poems locally from their final markdown poem section; do not use a model for SSOT
  extraction.
- Embed poem text with `gemini-embedding-2` at 768 dimensions and prefix every text with
  `task: sentence similarity | query: `. Compute within-thinking and minimal-to-high similarities
  independently for each method and topic.
- Judge poem PII with `gemini-3.1-flash-lite` through the Gemini Batch API, minimal thinking,
  no returned thoughts, a 512-token output limit, and the approved prompt stored in
  `experiments/multimodal_thinking_experiment/pii_judge_prompt.txt`. Require JSON containing only
  `count` and `evidence_phrases`, and store method-specific judgments and failures under each
  method's result data directory.
- For basic-poem lexical analysis, lowercase alphabetic tokens, preserve internal apostrophes,
  normalize curly apostrophes, and count both total occurrences and the fraction of poems containing
  each word. Store the aggregate table at
  `results/multimodal_thinking_experiment/tables/poetry_basic/unique_words.csv`.
- Use the pinned spaCy English pipeline for content-word analysis. Retain nouns, proper nouns,
  verbs, adjectives, and adverbs after removing stop words and explicit negation terms. Store
  surface-word and lemma aggregates as `content_words.csv` and `content_lemmas.csv` beside the
  unfiltered table.
- Store per-topic lemma poem counts separately for minimal and high thinking in
  `lemma_topic_usage_minimal.csv` and `lemma_topic_usage_high.csv`. Store each lemma's combined
  topic list in `lemma_topics.csv`.
- Store combined-thinking word-topic poem counts in `lemma_topic_sample_usage.csv`. Store each
  observed word-topic pair with the word's total distinct-topic count in
  `lemma_topic_coverage.csv`.
- The basic-poetry lexical browser opens a top-10 lemma table with topic and thinking-level
  selectors. Rank by the number of poems containing each lemma; calculate frequency over 10 poems
  for one thinking level and 20 poems for the combined view.
- Store the combined-thinking `(topic, num_unique_lemmas)` summary in
  `results/multimodal_thinking_experiment/tables/poetry_basic/num_unique_lemmas_by_topic.csv`.
  Append Shannon entropy in bits using each lemma's combined poem-presence count as its weight.
- Compare VS5 lexical diversity against the basic-poetry corpus using all 20 basic poems and 1,000
  random 20-poem samples from the extracted VS5 poems for each topic. Use random seed `0`, report
  95% intervals and VS5-minus-basic differences, and compute both corpora from retained lemma
  occurrence counts. Do not include SSOT in this matched lexical comparison.
- For basic-poetry homogeneity, sample 1,000 distinct unordered corpus-wide poem pairs with random
  seed `42` and score all 190 unordered pairs among each topic's 20 poems. Store row-level pair
  scores under `data/poetry_basic` and corpus, topic, and macro summaries under
  `tables/poetry_basic`. Measure semantic homogeneity with the existing normalized Gemini
  embeddings and lexical homogeneity with Jaccard similarity over the existing pruned content
  lemmas. Report the corpus and topic means separately; do not subtract one from the other.
- Measure poem-pair syntactic homogeneity by parsing sentences with the local Stanza English
  Universal Dependencies pipeline. Represent each sentence as an undirected dependency graph with
  UPOS node labels and a `none` root label. Use a normalized two-iteration Weisfeiler-Lehman vertex
  histogram kernel. Average every cross-poem sentence-pair kernel value and report syntactic
  similarity only. Keep the Stanza model under `.cache/models/stanza`.
- Do not compare homogeneity between generation methods until the user specifies that analysis.
- For the one-poem imageability view, randomly select from normalized basic-poem responses by
  default, retain content lemmas in poem order, predict the selected Glasgow norm with the
  five-member MLP ensemble, and display the plot interactively without saving it. Default to
  imageability and accept optional norm and seed arguments.
- Basic-poem form analysis uses CMU pronunciations through `pronouncing`. Classify variable-length
  rhyme into adjacent `AA`, alternating `AB`, repeating-three-line `ABC`, or
  `Other / non-rhyming` using experiment-local score and matched-line thresholds. Retain row-level
  scores and corpus, topic, and mean-topic distributions.
- Treat automated iambic pentameter as an approximation. Compare pronunciation stress strings to
  `0101010101`, retain pronunciation coverage and alignment scores, and apply the thresholds in
  `poetic_form_analysis_config.json`. Summarize line counts, stanza counts, stanza lengths,
  four-line stanza share, rhyme, and meter in the basic-poetry tables. Combine these summaries
  with the existing pair-similarity results in `homogeneity_comparisons.csv`.

## Text-Image Concept Collapse

- Use `text_image_concept_collapse` as the experiment ID under `experiments/`,
  `.cache/experiments/`, and `results/`.
- Preserve the existing text-generation and monitoring scripts. Make incremental additions instead
  of restructuring this experiment unless the user explicitly requests a restructure.
- Use the 11 approved text-image prompt pairs and request 20 text responses and 20 images per pair.
- Text generation uses `gemini-3.8-flash`; image generation uses `gemini-3-pro-image`. Keep thinking
  enabled for both modalities.
- Include `sample_idx` in both row-level output tables.
- Store images under `results/text_image_concept_collapse/data/images/<query_id>/`, label each file
  by `sample_idx`, and reference the image path from the image-generation CSV.
- Compute every cross-modal metric as all 400 pairings between the 20 items in each modality for
  each query. Store long-form row-level scores under `data/` and 20-by-20 matrices and query means
  under `tables/`.
- Keep analysis artifacts separated into `CLIPS`, `image-caption-embed`, and
  `text-to-image-embed` directories. Retain generated captions, generated-image metadata, and
  generated image files under the corresponding `data/` directory.
- Caption original images with `gemini-3.8-flash`, medium thinking, and the approved instruction:
  `Describe only the visible contents of this image in one paragraph. Do not mention the prompt or
  infer facts that are not visually supported.`
- Generate one image from each original response with `gemini-3-pro-image`, passing the response
  verbatim with no system instruction and keeping thinking enabled.
- Retry only `NO_IMAGE` response-to-image failures in a separate batch. Prefix each failed source
  response with the approved instruction `Create an image depicting the following description:`,
  followed by one blank line, and merge successful retries with successful first attempts.
- Create one 4-by-5 PNG collage for each complete 20-image query group in every image-generation
  setting so the samples can be inspected together.

## ACF Ratings Collapse

- Use `acfr_collapse` as the experiment ID under `experiments/`, `.cache/experiments/`, and
  `results/`.
- Sample 150 words uniformly without replacement from the `Word` column in
  `datasets/ACFnorms_TrochCrutchReilly_2014.csv`. Do not use a fixed random seed.
- Use the three approved text templates and three approved image templates in the experiment-local
  generation configuration, with five responses per word and template and `sample_idx` values 1
  through 5.
- Generate text with `gemini-3.1-flash-lite` and images with
  `gemini-3.1-flash-lite-image` through the Gemini Batch API.
- Use a maximum output length of 2,048 tokens for both modalities. Use medium thinking for text;
  the image model does not support medium, so retain its default minimal thinking level. Request
  returned thoughts for both modalities when available.
- Store all materialized rows in `results/acfr_collapse/data/generations.csv` with exactly the
  columns `word`, `prompt`, `modality`, `sample_idx`, `response`, and `thinking`.
- Store generated images under
  `results/acfr_collapse/data/images/<word>/<prompt_template>/sample_<sample_idx>.<extension>` and
  record their project-root-relative paths in the `response` column.
- Record provider failures in `results/acfr_collapse/data/failed_requests.csv`, omit those rows from
  `generations.csv`, and materialize all successful responses without requiring retries.
- Join every ACFr source column except `Word` onto `generations.csv` by `word` and store the result
  as `results/acfr_collapse/data/generations_augmented.csv`.

## Glasgow Norms Regression

- Use `glasgow_concreteness_regression` as the experiment ID under `experiments/` and `results/`.
- Load `datasets/Glasgow_norms.csv` without modifying it. Exclude parenthetical sense rows for the
  initial word-level experiment.
- Predict all nine continuous mean ratings from frozen `sentence-transformers/all-MiniLM-L6-v2`
  embeddings, using a separately tuned ridge regressor for each target.
- Use one fixed 80/10/10 train/dev/test split, stratified on the composite percentile across all
  targets. Select each ridge alpha on dev MAE, refit on train and dev, and evaluate test once.
- Retain ridge as the linear control. The nonlinear model is one shared nine-output MLP over frozen
  embeddings, with standardized inputs and targets, one GELU hidden layer, dropout, AdamW weight
  decay, and dev early stopping. Refit on train and dev for the selected epoch count.
- The MLP ensemble contains five members with seeds 0 through 4. Select each member's epoch on dev,
  refit each member on train and dev, average predictions equally, and retain prediction standard
  deviation as an uncertainty measure.
- Store embeddings, split assignments, fitted regressors, and row-level test predictions under
  `results/glasgow_concreteness_regression/data`. Store alpha-search and metric tables under
  `results/glasgow_concreteness_regression/tables`.
- Tests must use a fake encoder. Agents must not run the real local embedding stage.

## Code Organization

- Store experiment-specific scripts, configurations, tracked inputs, and tests together under
  `experiments/<experiment_name>` so the complete procedure can be reviewed in one directory.
  Group cross-lingual and intralingual embedding validation under `experiments/embedding_validation`.
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
  outputs only under `results/<experiment_name>/{data,tables,plots}` according to the
  raw-versus-aggregate distinction above. Store combined embedding-model mean tables under
  `results/embedding_validation/tables` in wide format, with one similarity column per model.

## Verification

- When handing off an experiment, always provide the exact repository-root commands for every
  remaining stage in execution order. Assume the user does not know the entry points or command
  syntax. Clearly mark commands that contact a paid API and commands that run local inference.
- Long-running file downloads must display a progress bar with downloaded bytes and total bytes
  when the total is available. Do not leave the terminal without visible progress for multi-minute
  downloads.
- Local analysis pipelines and processing jobs must display a simple progress bar with completed
  and total work so their progress is visible while they run.
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
