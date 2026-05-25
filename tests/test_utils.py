# tests/test_utils.py
import pytest
from scripts.utils import genre_from_filename, vector_labels, GENRES


def test_genre_from_filename():
    assert genre_from_filename("blues.00042.wav") == "blues"
    assert genre_from_filename("jazz.00001.wav") == "jazz"
    assert genre_from_filename("classical.00099.wav") == "classical"
    assert genre_from_filename("data/blues.00042.wav") == "blues"  # path prefix


def test_genre_from_filename_empty_raises():
    with pytest.raises(ValueError):
        genre_from_filename("")


def test_vector_labels_length():
    labels = vector_labels()
    assert len(labels) == 41
    assert len(labels) == len(set(labels))  # uniqueness


def test_genres_count():
    expected = {"blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"}
    assert set(GENRES) == expected
