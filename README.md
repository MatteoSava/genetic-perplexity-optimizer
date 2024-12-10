# Genetic Perplexity Optimizer

Optimization project for Kaggle's
[Santa 2024 - The Perplexity Permutation Puzzle](https://www.kaggle.com/competitions/santa-2024).

The challenge gives short bags of holiday-themed words. The goal is to reorder
each bag into a sequence that receives the lowest possible perplexity from the
competition language model. Lower perplexity means the model considers the text
more natural and more predictable.

## Approach

This repository implements a genetic algorithm for the permutation search:

- represent each candidate sentence as a genome, i.e. one valid permutation of
  the input words
- score candidates by running them through a causal language model and using
  negative loss as fitness
- keep the best candidates with elitism and tournament selection
- generate new candidates with duplicate-safe crossover and swap mutation
- adapt mutation pressure with population diversity and entropy metrics
- run larger experiments on Modal with a Gemma model mounted from a persistent
  volume

The implementation focuses on search quality and iteration speed rather than on
training a model from scratch.

## Project Layout

- `genetic/enhanced_genome.py`: genetic optimizer, genome representation,
  batched perplexity scoring, crossover, mutation, and diversity metrics
- `genetic/main.py`: experiment entrypoint that loads data, model, tokenizer,
  optimizer settings, and writes a Kaggle submission
- `genetic/config.yaml`: Weights & Biases sweep configuration
- `scripts/download_gemma.py`: downloads the Gemma model into a Modal volume
- `scripts/train_modal.py`: launches the optimizer on Modal GPU infrastructure
- `data/sample_submission.csv`: Kaggle sample submission format

## Perplexity

Perplexity is a language-model metric derived from cross-entropy loss. In
practice it answers: "How surprised is the model by this sequence of tokens?"
For this challenge, a good word order is one that makes the holiday text look
more likely under the reference language model, so the optimizer searches for
permutations with lower perplexity.

## Running Locally

Requires Python 3.11 or newer.

Install dependencies with uv:

```bash
uv sync --locked
```

Run the local entrypoint:

```bash
uv run python -m genetic.main
```

The default model path in `genetic/utils.py` is configured for the Modal volume:

```python
MODEL_NAME = "/modal/google/gemma-2-9b/"
```

Change that path before local execution if the model is stored somewhere else.

## Running On Modal

Create the required Modal secrets first:

- `huggingface-secret` with `HF_TOKEN`
- `wandb-secret` for experiment tracking

Download the model into the Modal volume:

```bash
uv run modal run scripts/download_gemma.py
```

Launch the optimization job:

```bash
uv run modal run scripts/train_modal.py
```
