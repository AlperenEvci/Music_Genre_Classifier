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
import librosa
from pathlib import Path
from scipy.spatial.distance import cosine

# PHP'den shell_exec ile çağrıldığında sys.path düzelt
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils import MODELS_DIR, SAMPLE_RATE
from scripts.extract_features import extract_features_from_file

_TEMPO_IDX = 32
_ZCR_MEAN_IDX = 28
_SC_MEAN_IDX = 26
_RMS_MEAN_IDX = 37
_ROLLOFF_MEAN_IDX = 33
_CHROMA_MEAN_IDX = 30


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
    
    # --- Explainable AI (XAI) Logic ---
    classes = model.classes_
    sorted_indices = np.argsort(proba)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]
    top1_genre = classes[top1_idx]
    top1_prob = proba[top1_idx]
    top2_genre = classes[top2_idx]
    top2_prob = proba[top2_idx]
    
    feature_names = {
        _TEMPO_IDX: "Tempo (Hız)",
        _ZCR_MEAN_IDX: "Pürüzlülük (Metalik Tını)",
        _SC_MEAN_IDX: "Parlaklık (Tiz Yoğunluğu)",
        _RMS_MEAN_IDX: "Ses Şiddeti (Enerji)",
        _ROLLOFF_MEAN_IDX: "Frekans Yuvarlaması",
        _CHROMA_MEAN_IDX: "Harmonik Yapı"
    }
    
    max_z_idx = max(feature_names.keys(), key=lambda k: abs(features_scaled[k]))
    max_z_val = features_scaled[max_z_idx]
    dom_feature = feature_names[max_z_idx]
    
    if max_z_val > 1.2:
        durum = "ortalamanın çok üzerinde"
    elif max_z_val > 0.5:
        durum = "ortalamanın üzerinde"
    elif max_z_val < -1.2:
        durum = "ortalamanın çok altında"
    elif max_z_val < -0.5:
        durum = "ortalamanın altında"
    else:
        durum = "standart değerlere yakın"
        
    explanation = (f"Analiz edilen ses dosyasının <strong>{dom_feature}</strong> seviyesi {durum} olduğu için, "
                   f"sistem <strong>%{top1_prob*100:.1f}</strong> ihtimalle <strong>{top1_genre.upper()}</strong> olduğuna karar vermiştir.")
    
    if top2_prob > 0.05:
        explanation += f" Ancak diğer akustik özelliklerdeki sinyaller nedeniyle <strong>%{top2_prob*100:.1f}</strong> ihtimalle <strong>{top2_genre.upper()}</strong> olabileceği de değerlendirilmiştir."
    # ---------------------------------

    feat_db_scaled = scaler.transform(feat_db)

    sims = []
    for i, db_vec_scaled in enumerate(feat_db_scaled):
        try:
            sim = _cosine_similarity(features_scaled, db_vec_scaled.astype(np.float32))
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
            "sc_mean":  round(float(features[_SC_MEAN_IDX]), 2),
            "rms_mean": round(float(features[_RMS_MEAN_IDX]), 6),
            "rolloff_mean": round(float(features[_ROLLOFF_MEAN_IDX]), 2),
            "chroma_mean": round(float(features[_CHROMA_MEAN_IDX]), 6)
        },
        "explanation": explanation,
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
