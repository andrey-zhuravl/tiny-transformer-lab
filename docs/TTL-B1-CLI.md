# CLI — tokenizer train / tokenizer inspect

## tokenizer train
```bash
ttlab tokenizer train --dataset-manifest .\out\dataset\dataset.manifest.json --algo bpe --vocab-size 5000 --norm nfc --lower --punct-policy keep --out .\out\tok\bpe5000 --seed 42
```
**Exit codes:** 0 OK, 2 INVALID_INPUT, 3 IO_ERROR, 4 UNKNOWN

## tokenizer inspect
```bash
ttlab tokenizer inspect --tokenizer .\out\tok\bpe5000\tokenizer.json --dataset-manifest .\out\dataset\dataset.manifest.json --out out/tokenizer
```
```
