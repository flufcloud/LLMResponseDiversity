Experiments on multilingual response diversity in LLMs

## Installation

- Install [uv](https://docs.astral.sh/uv/)
- Clone the repo

```bash
git clone https://github.com/flufcloud/LLMResponseDiversity.git
```

- Install libraries with uv sync

```bash
uv sync --frozen
```

# Reproducing experiments

Experiment code is located in the experiments/* folder. Running the code will reproduce the results. Note that this may require an API key for Gemini if you want to reproduce experiments that require model generations.