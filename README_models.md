# Tiny Transformer Lab — Model Guide

This guide explains how to work with the **Tiny Transformer Lab** model stack that ships with
the `ttlab` CLI. The A1 milestone introduces a vanilla decoder-only Transformer language model
with a unified registry, Hydra configuration, and a simple MLflow-aware training loop.

## Dataset contract

Training and evaluation jobs expect a directory with JSONL files that contain pre-tokenised
examples. Each line must contain an `input_ids` array and can optionally provide an
`attention_mask`.

```text
dataset/
├── train.jsonl
└── dev.jsonl
```

Example line:

```json
{"input_ids": [101, 1024, 2048, 102, 0], "attention_mask": [1, 1, 1, 1, 0]}
```

## Listing and inspecting models

All models are registered in a central registry and can be discovered via the CLI:

```bash
ttlab model list
ttlab model info --model model=vanilla
```

The `info` command prints the resolved Hydra configuration together with the parameter count.

## Training a model

Launch a training run by pointing the CLI to the dataset directory. The defaults load the
`conf/model/vanilla.yaml` and `conf/trainer/default.yaml` configurations:

```bash
ttlab model train \
  --data-dir /path/to/dataset \
  --model model=vanilla \
  --trainer trainer=default \
  --seed 212
```

Checkpoints are stored in the local `runs/` directory. Training automatically evaluates on
`dev.jsonl` at the configured interval.

## Evaluating checkpoints

To score a saved checkpoint on the development split:

```bash
ttlab model eval \
  --run-path runs/vanilla_step400.pt \
  --data-dir /path/to/dataset \
  --model model=vanilla
```

The command reports the loss and perplexity computed on `dev.jsonl`.

## MLflow logging

Provide an MLflow tracking URI to automatically log metrics, parameters, and artifacts. The run
stores the resolved configs, checkpoints, and a simple training loss plot.

```bash
ttlab model train \
  --data-dir /path/to/dataset \
  --model model=vanilla \
  --trainer trainer=default \
  --mlflow-uri http://127.0.0.1:5000
```

Artifacts include the checkpoint (`checkpoints/*.pt`) and resolved configuration files under the
`configs/` artifact path.

## Exp-lab integration example

Add the following snippet to an exp-lab recipe to orchestrate tokenisation, training, and
evaluation:

```yaml
steps:
  - ttlab.tokenizer.load: {path: ${paths.tokenizer}}
  - ttlab.model.train: {model: model=vanilla, trainer: trainer=default, data_dir: ${paths.dataset}}
  - ttlab.model.eval: {model: model=vanilla, data_dir: ${paths.dataset}}
paths:
  dataset: /absolute/path/to/dataset
  tokenizer: /absolute/path/to/tokenizer
```

This recipe assumes that the dataset already contains JSONL files adhering to the schema above.
