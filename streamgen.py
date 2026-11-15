# StreamGen - Stream matrices to disk chunk by chunk
# Part of the Kapildhinwa2010-dev Toolkit.

"""GenSuite - matrix generation engines: infinite, chunked, safe, parallel, stream.

Lineage: DNA_DYNIMO's production generators. One seeded recipe, five
execution modes for producing large weight matrices without ever holding
more than one chunk in memory (unless you ask for it).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = [
    "GenRecipe",
    "MemoryBudget",
    "generate",
    "generate_infinite",
    "generate_chunked",
    "generate_safe",
    "generate_parallel",
    "generate_stream",
    "load_stream",
]


class MemoryBudget(Exception):
    """A generation request exceeded the allowed byte budget."""

    def __init__(self, needed: int, allowed: int) -> None:
        super().__init__("need {} bytes, budget allows {}".format(needed, allowed))
        self.needed = needed
        self.allowed = allowed


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


def generate_infinite(recipe: GenRecipe):
    """Yield an endless stream of independent matrices from the recipe."""
    counter = 0
    while True:
        yield recipe._latent(counter)
        counter += 1


def generate_chunked(recipe: GenRecipe, chunk_rows: int):
    """Yield ``generate(recipe)`` in row-chunks; vstack of chunks == full."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    full = generate(recipe)
    for start in range(0, recipe.rows, chunk_rows):
        yield full[start:start + chunk_rows]


def generate_safe(recipe: GenRecipe, max_bytes: int = 8 * 1024 * 1024) -> np.ndarray:
    """Generate only if the float64 result fits in ``max_bytes``."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be >= 1")
    needed = recipe.rows * recipe.cols * 8
    if needed > max_bytes:
        raise MemoryBudget(needed, max_bytes)
    return generate(recipe)


def generate_parallel(recipe: GenRecipe, blocks: int = 4, workers: int = 2) -> list:
    """Generate ``blocks`` independent matrices across worker threads."""
    if blocks < 1:
        raise ValueError("blocks must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(recipe._latent, range(blocks)))


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
