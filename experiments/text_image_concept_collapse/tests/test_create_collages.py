from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.text_image_concept_collapse.create_collages import (
    CELL_SIZE,
    GAP,
    GRID_COLUMNS,
    GRID_ROWS,
    create_collage,
)


def test_create_collage_writes_four_by_five_grid(tmp_path: Path) -> None:
    rows = []
    for sample_idx in range(1, 21):
        image_path = tmp_path / f"{sample_idx:03d}.png"
        Image.new("RGB", (32, 16), color=(sample_idx, 0, 0)).save(image_path)
        rows.append({"sample_idx": sample_idx, "image_path": str(image_path)})
    output_path = tmp_path / "collage.png"

    create_collage(pd.DataFrame(rows), "sample_idx", output_path)

    with Image.open(output_path) as collage:
        assert collage.size == (
            GRID_COLUMNS * CELL_SIZE[0] + (GRID_COLUMNS - 1) * GAP,
            GRID_ROWS * CELL_SIZE[1] + (GRID_ROWS - 1) * GAP,
        )
        assert collage.format == "PNG"
