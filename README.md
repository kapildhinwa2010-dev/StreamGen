# StreamGen

[![CI](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Matrix generation engines: infinite, chunked, memory-safe, parallel and
streamed generation from one seeded recipe. DNA_DYNIMO lineage of the
[Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).

## Features

- Seeded low-rank `GenRecipe` - deterministic matrices
- `generate_infinite` - endless independent matrices
- `generate_chunked` - constant per-chunk memory
- `generate_safe` - byte-budget guard with `MemoryBudget`
- `generate_parallel` - thread-pool block generation
- `generate_stream` / `load_stream` - raw disk streaming round-trip

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```python
from streamgen import GenRecipe, generate, generate_chunked

recipe = GenRecipe(rows=64, cols=128, rank=8, seed=0)
W = generate(recipe)
for chunk in generate_chunked(recipe, chunk_rows=16):
    print(chunk.shape)
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT
