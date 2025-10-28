# Design

Components: CLI (`ttlab/cli/tokenizer.py`), Core (`ttlab/core/tokenizer.py`), Reports (`ttlab/core/tok_report.py`), Registry (`ttlab/registry/index.json`), MLflow utils.
Dataflow: dataset.manifest → tok:train → tokenizer.json → tok:inspect → tokenizer.report.json → MLflow.
Error handling: explicit exit codes; missing files → IO_ERROR; invalid args/data → INVALID_INPUT.
Determinism: fixed seed governs trainer and iteration order.
