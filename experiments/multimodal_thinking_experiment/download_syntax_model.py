from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).with_name("homogeneity_analysis_config.json")


def download_syntax_model(config_path: Path) -> None:
    import stanza

    config = json.loads(config_path.read_text(encoding="utf-8"))["syntax"]
    model_dir = Path(config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    stanza.download(
        lang=config["language"],
        model_dir=str(model_dir),
        processors=config["processors"],
        package=config["package"],
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the local Stanza syntax model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    download_syntax_model(parse_args().config)
