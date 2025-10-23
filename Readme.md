# Tiny Transformer Lab

Tiny Transformer Lab (ttlab) provides a minimal yet production-friendly toolkit for
experimenting with small transformer datasets. The project exposes a Typer-based CLI
with commands for bootstrapping a workspace, validating and converting datasets,
computing statistics, integrating with MLflow, and running smoke tests.

## Features

- Unified CLI `ttlab` with subcommands for init, environment checks, dataset operations,
  MLflow connectivity, and an end-to-end smoke scenario.
- Strict YAML configuration validated via Pydantic models.
- Data validation, conversion, deterministic splitting, and statistics powered by
  pandas and Apache Arrow.
- Metrics emitted to `out/metrics/metrics.jsonl` and optional logging to MLflow runs.
- GardenKeeper manifest for seamless integration with the GardenKeeper ecosystem.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run the smoke test to validate the setup:

```bash
ttlab smoke --config conf/base.yaml
```

## Development

Useful commands:

```bash
ruff check ttlab tests
black ttlab tests
mypy ttlab
pytest -q
```

## Repository Layout

Refer to `specs/A1.json.v2` for the full scope and acceptance criteria of milestone A1.
