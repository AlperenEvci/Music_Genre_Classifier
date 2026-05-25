# tests/test_predict.py
import numpy as np
import pytest
from pathlib import Path


def make_fake_models(tmp_path: Path) -> tuple:
    """Sahte model, scaler, feature_db oluştur."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from scripts.utils import GENRES

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 41)).astype(np.float32)
    y_train = np.tile(GENRES, 10)

    scaler = StandardScaler().fit(X_train)
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(scaler.transform(X_train), y_train)

    joblib.dump(clf, tmp_path / "model.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    np.save(str(tmp_path / "feature_db.npy"), X_train)
    np.save(str(tmp_path / "feature_db_labels.npy"), np.array(y_train, dtype="U"))
    np.save(str(tmp_path / "feature_db_filenames.npy"), np.array([f"song_{i}.wav" for i in range(100)]))

    return clf, scaler


def test_predict_returns_valid_result(tmp_path):
    import soundfile as sf
    from scripts.predict import predict_file

    sr = 22050
    t = np.linspace(0, 5, sr * 5)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, sr)

    make_fake_models(tmp_path)
    result = predict_file(str(wav_path), models_dir=tmp_path)

    assert result["error"] is None
    assert result["genre"] in [g for g in __import__("scripts.utils", fromlist=["GENRES"]).GENRES]
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) <= 3


def test_predict_recommendations_have_score(tmp_path):
    import soundfile as sf
    from scripts.predict import predict_file

    sr = 22050
    audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 5, sr * 5)) * 0.5).astype(np.float32)
    wav_path = tmp_path / "test2.wav"
    sf.write(str(wav_path), audio, sr)

    make_fake_models(tmp_path)
    result = predict_file(str(wav_path), models_dir=tmp_path)

    for rec in result["recommendations"]:
        assert "file" in rec
        assert "genre" in rec
        assert "score" in rec
        assert 0.0 <= rec["score"] <= 1.0


def test_predict_error_on_missing_file(tmp_path):
    from scripts.predict import predict_file

    make_fake_models(tmp_path)
    result = predict_file("/nonexistent/path.wav", models_dir=tmp_path)

    assert result["error"] is not None
    assert result["genre"] is None
    assert result["confidence"] == 0.0


def test_predict_error_on_missing_model(tmp_path):
    from scripts.predict import predict_file

    # Model dosyası yok
    result = predict_file("/some/file.wav", models_dir=tmp_path)

    assert result["error"] is not None
    assert result["genre"] is None
