# CLI — tok:train / tok:inspect

## tok:train
```bash
ttl tok:train --dataset-manifest path/to/dataset.manifest.json --algo bpe --vocab-size 8000 --norm nfc --lower --seed 13 --out out/tokenizer
```
**Exit codes:** 0 OK, 2 INVALID_INPUT, 3 IO_ERROR, 4 UNKNOWN

## tok:inspect
```bash
ttl tok:inspect --tokenizer out/tokenizer/tokenizer.json --dataset-manifest path/to/dataset.manifest.json --out out/tokenizer
```
