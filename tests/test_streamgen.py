import numpy as np
import pytest

from streamgen import (
    GenRecipe,
    MemoryBudget,
    generate,
    generate_chunked,
    generate_infinite,
    generate_parallel,
    generate_safe,
    generate_stream,
    load_stream,
)


def test_generate_shape():
    recipe = GenRecipe(rows=16, cols=32, rank=4, seed=0)
    assert generate(recipe).shape == (16, 32)


def test_generate_deterministic():
    recipe = GenRecipe(rows=8, cols=8, seed=3)
    assert np.array_equal(generate(recipe), generate(recipe))


def test_recipe_rejects_bad_args():
    with pytest.raises(ValueError):
        GenRecipe(rows=0, cols=8)
    with pytest.raises(ValueError):
        GenRecipe(rows=8, cols=8, rank=0)


def test_infinite_yields_distinct():
    recipe = GenRecipe(rows=4, cols=4, seed=1)
    stream = generate_infinite(recipe)
    first = next(stream)
    second = next(stream)
    third = next(stream)
    assert not np.array_equal(first, second)
    assert not np.array_equal(second, third)


def test_chunked_reassembles_to_full():
    recipe = GenRecipe(rows=10, cols=7, rank=3, seed=2)
    chunks = list(generate_chunked(recipe, chunk_rows=4))
    rebuilt = np.vstack(chunks)
    assert np.array_equal(rebuilt, generate(recipe))
    assert all(c.shape[0] <= 4 for c in chunks)


def test_chunked_rejects_bad_chunk():
    with pytest.raises(ValueError):
        list(generate_chunked(GenRecipe(rows=4, cols=4), chunk_rows=0))


def test_safe_allows_small():
    recipe = GenRecipe(rows=4, cols=4)
    assert generate_safe(recipe, max_bytes=1024).shape == (4, 4)


def test_safe_raises_memory_budget():
    recipe = GenRecipe(rows=1024, cols=1024)
    with pytest.raises(MemoryBudget):
        generate_safe(recipe, max_bytes=1024)


def test_safe_rejects_bad_budget():
    with pytest.raises(ValueError):
        generate_safe(GenRecipe(rows=2, cols=2), max_bytes=0)


def test_parallel_returns_blocks():
    recipe = GenRecipe(rows=6, cols=6, seed=5)
    blocks = generate_parallel(recipe, blocks=3, workers=2)
    assert len(blocks) == 3
    assert all(b.shape == (6, 6) for b in blocks)
    assert not np.array_equal(blocks[0], blocks[1])


def test_parallel_rejects_bad_args():
    with pytest.raises(ValueError):
        generate_parallel(GenRecipe(rows=2, cols=2), blocks=0)


def test_stream_roundtrip(tmp_path):
    recipe = GenRecipe(rows=9, cols=5, rank=3, seed=7)
    path = str(tmp_path / "stream.raw")
    rows = generate_stream(recipe, path, chunk_rows=4)
    assert rows == 9
    rebuilt = load_stream(path, rows=9, cols=5)
    assert np.array_equal(rebuilt, generate(recipe))


def test_load_stream_rejects_mismatch(tmp_path):
    recipe = GenRecipe(rows=4, cols=4)
    path = str(tmp_path / "s.raw")
    generate_stream(recipe, path)
    with pytest.raises(ValueError):
        load_stream(path, rows=5, cols=4)
