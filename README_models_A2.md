# Tiny Transformer Lab — A2 Models

## Overview

This milestone introduces two efficient attention families designed for long-context language modelling:

- **Linear attention** replaces softmax with non-negative feature maps, enabling attention cost linear in the sequence length.
- **Sparse attention** constrains the receptive field to sliding windows and optional global tokens for improved memory usage.

Both variants integrate with the Hydra configuration system and the `ttlab` CLI. They can be selected using overrides such as `model=linear` or `model=sparse`.

## Linear attention

Linear attention approximates the softmax kernel by mapping queries and keys through a positive feature map \(\phi\):

\[
\operatorname{softmax}(QK^\top) V \approx \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})}.
\]

Two feature maps are provided in `ttlab.utils.feature_maps`:

- `elu_plus_one`: \(\phi(x) = \operatorname{ELU}(x) + 1\)
- `relu`: \(\phi(x) = \operatorname{ReLU}(x) + \varepsilon\)

Set `model.linear.phi` in a Hydra override to switch feature maps. The default uses `n_features = 0`, meaning each head keeps its original dimensionality; specify `n_features` to project queries and keys into a different kernel dimension.

## Sparse attention

Sparse attention layers build masks using utilities from `ttlab.utils.sparse`:

- `sliding_window_mask(T, window, causal)` keeps tokens within a configurable window and optionally enforces causality.
- `local_global_mask(T, window, n_global, causal)` combines the sliding window with global tokens that all positions can attend to.

Configure the pattern via `model.sparse.pattern` (`sliding_window` or `local_global`) and adjust `window` / `n_global` as needed.

## Long-context trainer

Use the long-context trainer for experiments on extended sequences:

```bash
ttlab model train \
  --data-dir <dataset_dir> \
  --model model=linear \
  --trainer trainer=longctx \
  --seed 212
```

Replace `model=linear` with `model=sparse` to train the sparse variant.

## Evaluation

Evaluate a checkpoint with:

```bash
ttlab model eval --ckpt-path runs/linear_step2000.pt --data-dir <dataset_dir>
```

The command reports `dev/loss` and `dev/ppl` using the configuration stored in the checkpoint.

## Benchmarking

Run the long-context benchmark to compare speed and memory:

```bash
ttlab model bench \
  --models vanilla,linear,sparse \
  --seq-lens 512,2048 \
  --d-model 128 \
  --n-layers 2 \
  --n-heads 4 \
  --trials 3
```

Results are printed as JSON and saved to `bench/results.json`. If an MLflow URI is supplied, the benchmark logs metrics and artifacts tagged with `task=lm`, `longctx=true`, and the individual model family.

## Exp-lab recipe snippet

```yaml
steps:
  - ttlab.tokenizer.load: {path: ${paths.tokenizer}}
  - ttlab.model.train: {model: model=linear, trainer: trainer=longctx, data_dir: ${paths.dataset}}
  - ttlab.model.eval: {data_dir: ${paths.dataset}}
paths:
  dataset: /path/to/dataset
  tokenizer: /path/to/tokenizer
```

These additions enable long-context experiments while keeping performance competitive with the vanilla model on shorter sequences.
