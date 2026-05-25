# tests/test_integration.py
"""
Uçtan uca pipeline testleri.
- GTZAN varsa: pytest tests/test_integration.py -v -m integration
- GTZAN yoksa: skip (pytest skip mekanizması)
- Model yoksa: sahte model oluşturularak test edilir
"""
import numpy as np
import pytest
from pathlib import Path
from scripts.utils import GENRES, DATA_DIR, MODELS_DIR


@pytest.mark.integration
@pytest.mark.parametrize("genre", GENRES)
def test_predict_known_genre(genre, tmp_path_factory):
    """Her türden ilk şarkıyı tahmin et, sonucun geçerli olduğunu doğrula."""
    from scripts.predict import predict_file

    genre_dir = DATA_DIR / "gtzan" / "genres" / genre
    if not genre_dir.exists():
        pytest.skip(f"GTZAN verisi yok: {genre_dir}")

    wav_files = list(genre_dir.glob("*.wav"))
    if not wav_files:
        pytest.skip(f"WAV dosyası yok: {genre_dir}")

    if not (MODELS_DIR / "model.pkl").exists():
        pytest.skip(f"Model dosyaları yok: {MODELS_DIR}")

    result = predict_file(str(wav_files[0]))

    assert result["error"] is None, f"Hata: {result['error']}"
    assert result["genre"] in GENRES, f"Bilinmeyen tür: {result['genre']}"
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) <= 3


@pytest.mark.integration
def test_predict_wrong_format_wav_extension(tmp_path):
    """Metin içeren .wav uzantılı dosya → error döndürmeli."""
    from scripts.predict import predict_file
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Sahte model oluştur
    rng = np.random.default_rng(99)
    X = rng.standard_normal((100, 41)).astype(np.float32)
    y = np.tile(GENRES, 10)

    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(scaler.transform(X), y)

    joblib.dump(clf, tmp_path / "model.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    np.save(str(tmp_path / "feature_db.npy"), X)
    np.save(str(tmp_path / "feature_db_labels.npy"), np.array(y, dtype="U"))
    np.save(str(tmp_path / "feature_db_filenames.npy"), np.array([f"s_{i}.wav" for i in range(100)]))

    fake = tmp_path / "notaudio.wav"
    fake.write_text("bu ses değil")

    result = predict_file(str(fake), models_dir=tmp_path)
    assert result["error"] is not None, "Sahte dosya için error bekleniyor"


@pytest.mark.integration
def test_full_pipeline_with_synthetic_audio(tmp_path):
    """
    Synthetic ses → extract → predict (sahte model) → geçerli JSON döndürür.
    GTZAN veya gerçek model olmadan çalışır.
    """
    import soundfile as sf
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from scripts.predict import predict_file

    # Sahte model
    rng = np.random.default_rng(7)
    X = rng.standard_normal((100, 41)).astype(np.float32)
    y = np.tile(GENRES, 10)

    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(scaler.transform(X), y)

    joblib.dump(clf, tmp_path / "model.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    np.save(str(tmp_path / "feature_db.npy"), X)
    np.save(str(tmp_path / "feature_db_labels.npy"), np.array(y, dtype="U"))
    np.save(str(tmp_path / "feature_db_filenames.npy"), np.array([f"s_{i}.wav" for i in range(100)]))

    # 5 saniyelik kompleks ses (jazz-like)
    sr = 22050
    t = np.linspace(0, 5, sr * 5)
    audio = (
        0.3 * np.sin(2 * np.pi * 261 * t) +  # C4
        0.2 * np.sin(2 * np.pi * 329 * t) +  # E4
        0.1 * np.sin(2 * np.pi * 392 * t)    # G4
    ).astype(np.float32)

    wav_path = tmp_path / "synthetic.wav"
    sf.write(str(wav_path), audio, sr)

    result = predict_file(str(wav_path), models_dir=tmp_path)

    assert result["error"] is None
    assert result["genre"] in GENRES
    assert isinstance(result["confidence"], float)
    assert "tempo" in result["features"]
    assert "zcr_mean" in result["features"]
