# StreamGen - Stream matrices to disk chunk by chunk
# Part of the Kapildhinwa2010-dev Toolkit.

"""Raw binary streaming of recipe matrices to disk.

Lineage: DNA_DYNIMO generators. This repo owns the STREAM mode: write
row-chunks of the canonical matrix to a raw float64 file and read them
back with an explicit shape. stream_bytes() reports the on-disk size
before anything is written.

HONEST LIMITATION: the format is a bare float64 binary blob - no
header, no dtype metadata, no compression; the reader must be told the
shape. generate_stream materializes the full matrix once (see
ChunkGen for the memory discussion).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["GenRecipe", "generate", "generate_chunked", "generate_stream", "load_stream", "stream_bytes"]


class GenRecipe:
    """A seeded low-rank recipe: rows x cols output from rank x (rows+cols) latents."""

    def __init__(self, rows: int, cols: int, rank: int = 8, seed: int = 0) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.rows = rows
        self.cols = cols
        self.rank = min(rank, min(rows, cols))
        self.seed = seed

    def _latent(self, seed_offset: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + seed_offset)
        left = rng.standard_normal((self.rows, self.rank))
        right = rng.standard_normal((self.rank, self.cols))
        return (left @ right) / math.sqrt(self.rank)


def generate(recipe: GenRecipe) -> np.ndarray:
    """The canonical matrix of the recipe (seed + 0)."""
    return recipe._latent(0)

def generate_chunked(recipe: GenRecipe, chunk_rows: int):
    """Yield ``generate(recipe)`` in row-chunks; vstack of chunks == full."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    full = generate(recipe)
    for start in range(0, recipe.rows, chunk_rows):
        yield full[start:start + chunk_rows]


def generate_stream(recipe: GenRecipe, path: str, chunk_rows: int = 8) -> int:
    """Stream row-chunks of the matrix into a raw binary file; return rows."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    written = 0
    with open(path, "wb") as fh:
        for chunk in generate_chunked(recipe, chunk_rows):
            fh.write(np.ascontiguousarray(chunk, dtype=np.float64).tobytes())
            written += int(chunk.shape[0])
    return written


def load_stream(path: str, rows: int, cols: int) -> np.ndarray:
    """Load a raw stream written by :func:`generate_stream`."""
    data = np.fromfile(path, dtype=np.float64)
    if data.size != rows * cols:
        raise ValueError("file size mismatch: expected {} values, got {}".format(rows * cols, data.size))
    return data.reshape(rows, cols)


def stream_bytes(rows: int, cols: int, dtype_bytes: int = 8) -> int:
    """On-disk size in bytes of a raw float64 (by default) stream."""
    if rows < 1 or cols < 1 or dtype_bytes < 1:
        raise ValueError("rows, cols and dtype_bytes must be >= 1")
    return rows * cols * dtype_bytes