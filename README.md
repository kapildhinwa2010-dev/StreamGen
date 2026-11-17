# StreamGen

StreamGen is a small Python module for generating seeded, low-rank numeric matrices and writing their values to a raw binary file. The writer emits rows in chunks, but matrix generation currently materializes the complete matrix in memory first. The project metadata identifies version 0.1.0; this is a minimal utility, not a general-purpose streaming or storage system.

## Requirements

- Python 3.9 or newer
- NumPy 1.24 or newer

## Install

From the repository root:

```bash
python -m pip install .
```

To install the test dependencies as well, use `python -m pip install -r requirements.txt`.

## Example

```python
from streamgen import GenRecipe, generate, generate_stream, load_stream, stream_bytes

recipe = GenRecipe(rows=64, cols=128, rank=8, seed=0)

matrix = generate(recipe)  # shape: (64, 128)
print(stream_bytes(64, 128))  # 65536 bytes at the default 8 bytes per value

rows_written = generate_stream(recipe, "matrix.raw", chunk_rows=8)
restored = load_stream("matrix.raw", rows=rows_written, cols=128)
```

`generate` returns the full matrix. `generate_stream` overwrites the specified file and returns the number of rows written. `load_stream` reads a raw stream when given its row and column counts; it raises `ValueError` if the number of stored values does not match that shape.

## Available functions

- `GenRecipe(rows, cols, rank=8, seed=0)` describes a matrix recipe. Rows, columns, and rank must be positive; the effective rank is capped at the smaller matrix dimension.
- `generate(recipe)` returns the deterministic matrix for the recipe's seed. The values are built from seeded standard-normal factors and scaled by the square root of the effective rank.
- `generate_chunked(recipe, chunk_rows)` yields row slices of the generated matrix. It currently calls `generate` first, so it does not avoid holding the full matrix in memory.
- `generate_stream(recipe, path, chunk_rows=8)` writes those row slices as contiguous float64 values and returns the row count.
- `load_stream(path, rows, cols)` reads float64 values and reshapes them to the supplied dimensions.
- `stream_bytes(rows, cols, dtype_bytes=8)` calculates `rows * cols * dtype_bytes` after validating positive inputs. This is a size calculation only: the writer always writes float64, even when a different `dtype_bytes` value is passed here.

## Format and limitations

The file format is a bare sequence of float64 values. It contains no header, shape, dtype marker, byte-order marker, or compression; callers must keep and supply the dimensions when loading. `generate_stream` and `generate_chunked` both generate the complete matrix before slicing it, so they do not provide bounded-memory matrix generation. The module does not define a command-line interface, a self-describing file format, compression, or incremental generation of matrix factors.

## Tests

The test suite uses pytest. On the cloned repository commit `80ffdf611dff03fef7107353728da2647a4d2ab8`, `python3 -m pytest -q` completed locally with **9 passed** (Python 3.12.3). The CI workflow is configured to run pytest and compile/smoke checks on Python 3.10, 3.11, and 3.12; that workflow configuration is not a claim about separate CI run results.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms.
