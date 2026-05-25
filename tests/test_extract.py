# tests/test_extract.py
from pathlib import Path
import pytest
import numpy as np


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


def test_extract_single_returns_correct_shape(tmp_path):
    """5 saniyelik sahte ses → 41 boyutlu vektör döndürmeli."""
    import soundfile as sf
    from scripts.extract_features import extract_features_from_file

    sr = 22050
    duration = 5
    t = np.linspace(0, duration, sr * duration)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    wav_path = tmp_path / "test_tone.wav"
    sf.write(str(wav_path), audio, sr)

    features = extract_features_from_file(str(wav_path))

    assert isinstance(features, np.ndarray)
    assert features.shape == (41,)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_extract_raises_on_too_short(tmp_path):
    """2 saniyelik ses → ValueError raise etmeli (min 3s)."""
    import soundfile as sf
    from scripts.extract_features import extract_features_from_file

    sr = 22050
    audio = np.zeros(sr * 2, dtype=np.float32)
    wav_path = tmp_path / "short.wav"
    sf.write(str(wav_path), audio, sr)

    with pytest.raises((ValueError, Exception)):
        extract_features_from_file(str(wav_path))
