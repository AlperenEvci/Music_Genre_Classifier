# tests/test_extract.py
from pathlib import Path
import pytest


def test_genre_from_filename_edge_cases():
    from scripts.utils import genre_from_filename
    assert genre_from_filename("rock.00000.wav") == "rock"
    assert genre_from_filename("hiphop.00050.wav") == "hiphop"


def test_scan_dataset_returns_tuples(tmp_path):
    """Boş genres dizini → ([], []) döndürmeli"""
    from scripts.utils import scan_dataset
    genres_dir = tmp_path / "genres"
    genres_dir.mkdir()
    valid, skipped = scan_dataset(tmp_path)
    assert isinstance(valid, list)
    assert isinstance(skipped, list)
    assert len(valid) == 0
    assert len(skipped) == 0


def test_scan_dataset_missing_genres_dir(tmp_path):
    """genres/ alt dizini yoksa boş döndürmeli (rglob silent)"""
    from scripts.utils import scan_dataset
    # genres/ dizini oluşturma — rglob boş iterator döndürmeli
    valid, skipped = scan_dataset(tmp_path)
    assert valid == []
    assert skipped == []
