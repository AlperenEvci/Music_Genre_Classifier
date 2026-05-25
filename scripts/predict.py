# scripts/predict.py
"""
Ana tahmin scripti.
Kullanım (CLI): python -m scripts.predict <dosya_yolu>
Çıktı: JSON → stdout
"""
import sys
import json
import numpy as np
import joblib
from pathlib import Path
from scipy.spatial.distance import cosine
from scripts.utils import MODELS_DIR
from scripts.extract_features import extract_features_from_file

_TEMPO_IDX = 32
_ZCR_MEAN_IDX = 28


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    norm_a = np.linalg.norm(a64)
    norm_b = np.linalg.norm(b64)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(1.0 - cosine(a64, b64))


def predict_file(
    file_path: str,
    models_dir: Path = MODELS_DIR,
    n_recommendations: int = 3,
) -> dict:
    """
    Tek dosya için tür tahmini + cosine similarity öneri.
    Her zaman dict döndürür. Hata durumunda error field dolu, genre=None.
    """
    try:
        model     = joblib.load(models_dir / "model.pkl")
        scaler    = joblib.load(models_dir / "scaler.pkl")
        feat_db   = np.load(str(models_dir / "feature_db.npy"))
        labels    = np.load(str(models_dir / "feature_db_labels.npy"))
        filenames = np.load(str(models_dir / "feature_db_filenames.npy"))
    except FileNotFoundError as e:
        return {
            "error": f"Model dosyası bulunamadı: {e}",
            "genre": None, "confidence": 0.0, "recommendations": [], "features": {}
        }

    try:
        features = extract_features_from_file(file_path)
    except Exception as e:
        return {
            "error": f"Öznitelik çıkarımı başarısız: {e}",
            "genre": None, "confidence": 0.0, "recommendations": [], "features": {}
        }

    features_scaled = scaler.transform(features.reshape(1, -1))[0]

    genre = model.predict(features_scaled.reshape(1, -1))[0]
    proba = model.predict_proba(features_scaled.reshape(1, -1))[0]
    confidence = float(proba.max())

    sims = []
    for i, db_vec in enumerate(feat_db):
        try:
            sim = _cosine_similarity(features, db_vec.astype(np.float32))
            sims.append({
                "idx": i,
                "score": sim,
                "file": str(filenames[i]),
                "genre": str(labels[i]),
            })
        except Exception:
            continue

    sims.sort(key=lambda x: x["score"], reverse=True)
    recommendations = [s for s in sims if s["score"] < 0.9999][:n_recommendations]

    return {
        "genre": str(genre),
        "confidence": round(confidence, 4),
        "recommendations": [
            {"file": r["file"], "genre": r["genre"], "score": round(r["score"], 4)}
            for r in recommendations
        ],
        "features": {
            "tempo":    round(float(features[_TEMPO_IDX]), 2),
            "zcr_mean": round(float(features[_ZCR_MEAN_IDX]), 6),
        },
        "error": None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Kullanım: python -m scripts.predict <dosya_yolu>",
            "genre": None, "confidence": 0.0, "recommendations": [], "features": {}
        }))
        sys.exit(1)

    result = predict_file(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))
