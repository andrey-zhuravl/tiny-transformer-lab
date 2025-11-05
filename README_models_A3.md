# Tiny Transformer Lab — A3 Models

## Overview

Milestone A3 introduces two structure-aware Transformer families alongside role-specific evaluation utilities:

- **Hyperbolic Transformer (`model=hyper`)** embeds token representations on the Poincaré ball and performs attention/MLP updates in the tangent space.
- **Graph-biased Transformer (`model=graph`)** injects additive attention biases derived from per-sample graph edges (adjacency or shortest-path distances).
- **Role-wise evaluation** reports accuracies for `who`/`what`/`where` positions in datasets annotated with role indices.

All components integrate with Hydra configs and the `ttlab` CLI so they can be composed in exp-lab recipes.

## Hyperbolic Transformer

The hyperbolic model maps token embeddings to the Poincaré ball of curvature `hyper.curvature_c` via the exponential map:

\[
\exp_0(v) = \tanh(\sqrt{c}\,\lVert v \rVert) \frac{v}{\sqrt{c}\,\lVert v \rVert}.
\]

Transformer blocks operate in the tangent space using the logarithmic map, then project the updated states back to the ball. Important configuration keys:

| Key | Description |
| --- | --- |
| `model.hyper.curvature_c` | Positive curvature parameter \(c > 0\). Larger values reduce the ball radius. |
| `model.hyper.proj_tau` | Pre-scaling factor applied before the exponential map to control the effective norm. |
| `model.hyper.attn_in_tangent` | (A3 default `true`) keeps attention/MLP updates in Euclidean tangent space. |

Select the model with `ttlab model train --model model=hyper`. The LM head automatically applies the logarithmic map to return to Euclidean space before projection to vocabulary logits.

## Graph-biased Transformer

Graph attention biases are constructed from `graph_edges` supplied per sample. Supported bias types:

- `adjacency`: adds `+bias.scale` to scores for listed edges (directional). Optional third element sets a custom weight per edge.
- `shortest_path`: performs an unweighted BFS and subtracts `bias.scale * distance(i, j)` for reachable nodes. The result is clamped to `[-bias.clip, bias.clip]`.

Key configuration fields:

| Key | Description |
| --- | --- |
| `model.graph.bias.type` | `adjacency` or `shortest_path`. |
| `model.graph.bias.scale` | Multiplicative scale applied to the constructed bias. |
| `model.graph.bias.clip` | Absolute clamp for stability. |
| `model.graph.source.format` | `sample` indicates edges are read from each JSONL record. |

Launch training with `ttlab model train --model model=graph`. The model remains causal because PyTorch SDPA receives both the graph bias and the usual causal mask.

### Dataset edge format

Provide a `graph_edges` array in each JSONL sample:

```json
{"input_ids": [11, 22, 33, 44], "graph_edges": [[0, 1], [1, 2, 0.5], [2, 3]]}
```

Each entry is `[src, dst]` or `[src, dst, weight]`. Edges outside the sequence length are ignored.

## Role annotations and evaluation

Dev/test JSONL files may include a `roles` object with index lists per role:

```json
{
  "input_ids": [101, 102, 103, 104],
  "roles": {"who": [0], "what": [2], "where": [3]}
}
```

Run role-wise evaluation on a checkpoint with:

```bash
ttlab model eval-roles \
  --ckpt-path runs/graph_step0040.pt \
  --data-dir /path/to/dataset
```

If role annotations are present, the command prints JSON metrics and writes `eval/roles.json`. When no roles are found, it prints an informational message and exits without creating artifacts.

The evaluation utility supports both vanilla and structure-aware models, automatically loading `graph_edges` so graph-biased checkpoints evaluate correctly.

## MLflow telemetry

When tracking is enabled via `--mlflow-uri`, training runs log:

- Parameters: flattened `model.*`, `trainer.*`, nested `hyper.*` / `graph.*`, `tokenizer.*`, and the `seed` used.
- Tags: `family=hyper|graph|baseline`, `task=lm`, `struct=generalization`.
- Metrics: `train/loss`, `dev/loss`, `dev/ppl`, plus `roles/acc_{who,what,where}` when role evaluation is executed inside an active run.
- Artifacts: resolved configs, checkpoints, and any `eval/roles.json` emitted by the evaluation helper.

## Exp-lab recipe snippet

```yaml
steps:
  - ttlab.tokenizer.load: {path: ${paths.tokenizer}}
  - ttlab.model.train: {model: model=hyper, trainer: trainer=default, data_dir: ${paths.dataset}}
  - ttlab.model.eval: {data_dir: ${paths.dataset}}
  - ttlab.model.eval-roles: {data_dir: ${paths.dataset}}
paths:
  dataset: /path/to/dataset
  tokenizer: /path/to/tokenizer
```

Adjust the `model` override to `model=graph` for graph-biased experiments. Tune `bias.scale` or `hyper.curvature_c` via additional overrides when adapting to new datasets.
