import numpy as np
import pytest

from streamgen import GenRecipe, generate, generate_stream, load_stream, stream_bytes


def test_generate_shape():
    recipe = GenRecipe(rows=16, cols=32, rank=4, seed=0)
    assert generate(recipe).shape == (16, 32)


def test_generate_deterministic():
    recipe = GenRecipe(rows=8, cols=8, seed=3)
    assert np.array_equal(generate(recipe), generate(recipe))


def test_recipe_rejects_bad_args():
    with pytest.raises(ValueError):
        GenRecipe(rows=0, cols=8)


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


def test_stream_rejects_bad_chunk(tmp_path):
    with pytest.raises(ValueError):
        generate_stream(GenRecipe(rows=4, cols=4), str(tmp_path / "x.raw"), chunk_rows=0)


# --- size planning (StreamGen-specific) ---


def test_stream_bytes_float64():
    assert stream_bytes(10, 20) == 1600


def test_stream_bytes_custom_dtype():
    assert stream_bytes(10, 20, dtype_bytes=4) == 800


def test_stream_bytes_rejects_bad_args():
    with pytest.raises(ValueError):
        stream_bytes(0, 20)
    with pytest.raises(ValueError):
        stream_bytes(10, 20, dtype_bytes=0)