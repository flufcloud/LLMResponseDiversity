from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


EXPERIMENT_ID = "text_image_concept_collapse"
RESULTS_ROOT = Path("results") / EXPERIMENT_ID
IMAGE_SETS = {
    "prompt-images": {
        "csv": "data/image_experiment.csv",
        "sample_column": "sample_idx",
    },
    "response-images": {
        "csv": "data/text-to-image-embed/generated_images.csv",
        "sample_column": "response_sample_idx",
    },
}
GRID_COLUMNS = 5
GRID_ROWS = 4
CELL_SIZE = (384, 216)
GAP = 4


def create_collage(
    rows: pd.DataFrame,
    sample_column: str,
    output_path: Path,
) -> None:
    width = GRID_COLUMNS * CELL_SIZE[0] + (GRID_COLUMNS - 1) * GAP
    height = GRID_ROWS * CELL_SIZE[1] + (GRID_ROWS - 1) * GAP
    collage = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(collage)
    for position, row in enumerate(
        rows.sort_values(sample_column).itertuples(index=False)
    ):
        grid_row, grid_column = divmod(position, GRID_COLUMNS)
        x = grid_column * (CELL_SIZE[0] + GAP)
        y = grid_row * (CELL_SIZE[1] + GAP)
        with Image.open(row.image_path) as source:
            tile = ImageOps.contain(
                source.convert("RGB"), CELL_SIZE, Image.Resampling.LANCZOS
            )
        tile_x = x + (CELL_SIZE[0] - tile.width) // 2
        tile_y = y + (CELL_SIZE[1] - tile.height) // 2
        collage.paste(tile, (tile_x, tile_y))
        label = f"{int(getattr(row, sample_column)):03d}"
        draw.rectangle((x + 5, y + 5, x + 35, y + 21), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path, format="PNG", optimize=True)


def create_collages(results_root: Path) -> list[Path]:
    outputs = []
    for setting, options in IMAGE_SETS.items():
        images = pd.read_csv(results_root / options["csv"])
        groups = images.groupby("query_id", sort=True)
        for query_id, rows in tqdm(
            groups,
            total=groups.ngroups,
            desc=f"Creating {setting} collages",
        ):
            output_path = results_root / "plots/collages" / setting / f"{query_id}.png"
            create_collage(rows, options["sample_column"], output_path)
            outputs.append(output_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 4-by-5 image collages")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    for path in create_collages(Path(parse_args().results_root)):
        print(path)
