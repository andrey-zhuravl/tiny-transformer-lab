# TTL B1 — Overview

**Purpose:** Provide a standardized tokenizer layer (char/BPE/Unigram) for tiny-transformer-lab.
**Integration:** toy-lang-lab (A2) → TTL B1 (tokenizer) → TTL B2 (encoding) → C-phase (training).
**Outputs:** tokenizer.json, tokenizer.manifest.json, tokenizer.report.json (all logged to MLflow/MinIO).
