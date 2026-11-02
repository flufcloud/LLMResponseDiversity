# Cross-modal analysis

The analysis uses three independent metric directories under both `data/` and `tables/`:

- `CLIPS`: cosine similarity between each original text response and each original image using
  normalized SigLIP 2 text and image embeddings.
- `image-caption-embed`: cosine similarity between each original text response and each generated
  image caption using normalized BGE-M3 sentence embeddings.
- `text-to-image-embed`: cosine similarity between each response-generated image and each original
  image using normalized SigLIP 2 image embeddings.

Each metric writes a long-form `pairwise_similarity.csv`, a wide `similarity_matrix.csv`, and a
`mean_similarity_by_query.csv`. Each query contributes 400 pairwise scores.

SigLIP 2 uses its fixed 64-token text input. Longer responses are truncated by the model processor;
the configured token limit is part of the recorded method and must not vary between runs.

The Gemini stages are separate from local embedding and analysis. Their approved settings are in
`cross_reference_generation_config.json`. Captioning uses Gemini 3.8 Flash with the approved
visible-content-only instruction. Response-to-image generation sends each source response verbatim
to Gemini 3 Pro Image. Thinking is enabled for both.

Image captions are materialized at `data/image-caption-embed/image_captions.csv`.
Response-generated image metadata is materialized at
`data/text-to-image-embed/generated_images.csv`, and the files remain under
`data/text-to-image-embed/generated_images/`.

Prepare each batch locally:

```bash
uv run python -m experiments.text_image_concept_collapse.generate_cross_references captions prepare
uv run python -m experiments.text_image_concept_collapse.generate_cross_references response-images prepare
```

Submission, status, and download commands contact Gemini, can incur charges, and require both
`user_approved_for_api: true` in the configuration and the `--confirm-api` flag. Only the user may
change the approval field.

If the initial response-to-image batch returns `NO_IMAGE`, prepare a separate failed-only retry
batch. Each retry uses the approved instruction followed by the original response verbatim:

```text
Create an image depicting the following description:

{original response}
```

```bash
uv run python -m experiments.text_image_concept_collapse.generate_cross_references response-images-retry prepare
uv run python -m experiments.text_image_concept_collapse.generate_cross_references response-images-retry submit --confirm-api
uv run python -m experiments.text_image_concept_collapse.generate_cross_references response-images-retry status --confirm-api
uv run python -m experiments.text_image_concept_collapse.generate_cross_references response-images-retry download --confirm-api
uv run python -m experiments.text_image_concept_collapse.generate_cross_references response-images-retry finalize
```

Retry finalization merges successful first attempts and successful retries into
`data/text-to-image-embed/generated_images.csv`.

Create 4-by-5 collages for every query in both image-generation settings:

```bash
uv run python -m experiments.text_image_concept_collapse.create_collages
```

Collages are stored under `plots/collages/{prompt-images,response-images}/`.
