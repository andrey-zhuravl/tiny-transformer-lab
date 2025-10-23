# tiny-transformer-lab

This repository provides a minimal CLI and validation utilities for toy-lang-lab
datasets. Use the `data:validate` command to ensure generated datasets follow
the configured YAML schema.

```bash
python -m ttlab.cli data:validate --in path/to/dataset.jsonl --schema conf/data/sample_dataset.yaml --format jsonl
```
