# StreamGen

[![CI](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/StreamGen/actions/workflows/codeql.yml)

Stream recipe matrices to disk chunk by chunk as raw float64, with explicit-shape reload.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

StreamGen owns the STREAM mode of the DNA_DYNIMO generator family:
generate_stream() writes row-chunks of the canonical matrix to a raw
float64 file; load_stream() reads it back with an explicit shape;
stream_bytes() reports the on-disk size before anything is written.
The HONEST LIMITATION (also in the docstring): the format is a bare
float64 binary blob - no header, no dtype metadata, no compression.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from streamgen import GenRecipe, generate_stream, load_stream, stream_bytes

recipe = GenRecipe(rows=64, cols=128, rank=8, seed=0)
print(stream_bytes(64, 128))          # 65536
generate_stream(recipe, "W.bin", chunk_rows=8)
W = load_stream("W.bin", 64, 128)
```

## API

- GenRecipe(rows, cols, rank, seed)
- generate(recipe) - the canonical matrix
- generate_stream(recipe, path, chunk_rows) - write, returns rows
- load_stream(path, rows, cols) - read, raises on size mismatch
- stream_bytes(rows, cols, dtype_bytes=8)

## Honest results and limitations

- No header or metadata: the reader must be told rows and cols.
- No compression: on-disk size is exactly rows*cols*8 bytes.
- generate_stream materializes the full matrix once (see ChunkGen).

## License

MIT - see [LICENSE](LICENSE).
