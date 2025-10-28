# tiny-transformer-lab

This repository provides a minimal CLI utilities for toy-lang-lab
datasets. Use the `ttlab validate run` command to ensure generated datasets follow
the configured YAML schema.

```bash
python -m ttlab validate run --in D:\garden\lab\toy-lang-lab\out\dev\test.jsonl --schema .\conf\data\sample_dataset.yaml --format JSONL --metrics-dir D:\garden\lab\toy-lang-lab\out\dev
python -m ttlab process run --in D:\garden\lab\toy-lang-lab\out\dev\test.jsonl   --schema .\conf\data\sample_dataset.yaml --format JSONL   --out .\out\dataset --split train=0.6,dev=0.2,test=0.2  --seed 42
python -m ttlab tokenizer train --dataset-manifest .\out\dataset\dataset.manifest.json --algo bpe --vocab-size 5000 --norm nfc --lower --punct-policy keep --out .\out\tok\bpe5000 --seed 42
python -m ttlab tokenizer inspect --tokenizer .\out\tok\bpe5000\tokenizer.json --dataset-manifest .\out\dataset\dataset.manifest.json --out out/tokenizer
```
