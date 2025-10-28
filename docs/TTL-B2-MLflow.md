# TTL B2 — MLflow Tracking

## Quick start

```bash
pip install -e .[mlflow]

python -m ttlab process run \
  --in path/to/data.jsonl \
  --schema conf/data/sample_dataset.yaml \
  --format JSONL \
  --split train=0.7,dev=0.2,test=0.1 \
  --seed 42 \
  --out out/dataset \
  --mlflow --experiment ttlab.dev --mlflow-uri file:./out/mlruns --tag commit=$(git rev-parse HEAD)

# UI
mlflow ui --backend-store-uri out/mlruns
```

## Nested runs

Pass `--parent-run-id <id>` to create a nested run under an existing parent.
