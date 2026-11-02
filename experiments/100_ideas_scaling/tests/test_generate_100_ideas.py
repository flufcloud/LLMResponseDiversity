import json
from argparse import Namespace
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

generation = import_module("experiments.100_ideas_scaling.generate_responses")
finalize_results = generation.finalize_results
load_settings = generation.load_settings
prepare_requests = generation.prepare_requests
submit_requests = generation.submit_requests
validate_requests = generation.validate_requests


def write_config(tmp_path: Path, approved: bool = False) -> Path:
    path = tmp_path / "generation_config.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "100_ideas_scaling",
                "user_approved_for_api": approved,
                "model": "gemini-3.8-flash",
                "responses_per_prompt": 2,
                "system_instruction": None,
                "generation_config": {},
                "diversity_instruction": "List 100 ideas as JSON.",
                "prompts": [
                    {"prompt_id": "prompt-001", "base_prompt": "Write a story."}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def make_args(tmp_path: Path, approved: bool = False) -> Namespace:
    return Namespace(
        config=str(write_config(tmp_path, approved)),
        runtime_root=str(tmp_path / "runtime"),
        results_root=str(tmp_path / "results"),
        confirm_api=False,
    )


def test_prepares_ten_prompts_with_ten_samples(tmp_path: Path) -> None:
    config = Path(
        "experiments/100_ideas_scaling/generation_config.json"
    )
    settings = load_settings(config)
    assert settings.model == "gemini-3.8-flash"
    assert len(settings.prompts) == 10
    assert settings.responses_per_prompt == 10
    args = Namespace(
        config=str(config),
        runtime_root=str(tmp_path / "runtime"),
    )
    prepare_requests(args)
    requests = (tmp_path / "runtime/generation_requests.jsonl").read_text().splitlines()
    manifest = pd.read_csv(tmp_path / "runtime/generation_manifest.csv")
    assert len(requests) == 100
    assert len(manifest) == 100
    assert manifest.groupby("prompt_id")["sample_idx"].apply(list).tolist() == [
        list(range(1, 11))
    ] * 10


def test_prepare_and_validate_requests(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)

    requests = [
        json.loads(line)
        for line in (tmp_path / "runtime/generation_requests.jsonl")
        .read_text()
        .splitlines()
    ]
    manifest = pd.read_csv(tmp_path / "runtime/generation_manifest.csv")
    assert [item["key"] for item in requests] == [
        "generation--prompt-001--001",
        "generation--prompt-001--002",
    ]
    assert manifest["sample_idx"].tolist() == [1, 2]
    assert requests[0]["request"]["contents"][0]["parts"][0]["text"] == (
        "Write a story.\n\nList 100 ideas as JSON."
    )
    validate_requests(Path(args.runtime_root), load_settings(Path(args.config)))


def test_submission_requires_confirmation_and_user_approval(tmp_path: Path) -> None:
    args = make_args(tmp_path)

    with pytest.raises(PermissionError, match="--confirm-api"):
        submit_requests(args)

    args.confirm_api = True
    with pytest.raises(PermissionError, match="user_approved_for_api"):
        submit_requests(args)


def test_finalizes_results_as_flat_csv(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    prepare_requests(args)
    manifest = pd.read_csv(tmp_path / "runtime/generation_manifest.csv")
    result_path = tmp_path / "runtime/generation_results.jsonl"
    result_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "key": row.request_key,
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": f'["idea-{row.sample_idx}"]'}]
                                }
                            }
                        ]
                    },
                }
            )
            for row in manifest.itertuples(index=False)
        )
        + "\n",
        encoding="utf-8",
    )

    finalize_results(args)

    output = pd.read_csv(tmp_path / "results/data/generations.csv")
    assert len(output) == 2
    assert output["sample_idx"].tolist() == [1, 2]
    assert output["response"].tolist() == ['["idea-1"]', '["idea-2"]']
    assert output["system_instruction"].isna().all()
