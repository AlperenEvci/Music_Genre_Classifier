# tests/test_utils.py
from scripts.utils import genre_from_filename, vector_labels, GENRES


def test_genre_from_filename():
    assert genre_from_filename("blues.00042.wav") == "blues"
    assert genre_from_filename("jazz.00001.wav") == "jazz"
    assert genre_from_filename("classical.00099.wav") == "classical"


def test_vector_labels_length():
    labels = vector_labels()
    assert len(labels) == 41


def test_genres_count():
    assert len(GENRES) == 10
    assert "blues" in GENRES
    assert "rock" in GENRES
