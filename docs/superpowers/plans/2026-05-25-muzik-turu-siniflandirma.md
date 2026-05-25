# Müzik Türü Sınıflandırma ve Öneri Sistemi — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GTZAN veri seti üzerinde akustik öznitelik çıkarımı yaparak 10 müzik türünü sınıflandıran ve cosine similarity ile içerik tabanlı öneri sunan tam işlevsel bir sistem inşa etmek.

**Architecture:** Python pipeline (librosa → scikit-learn) offline eğitim yapar, `.pkl` modelleri diske kaydeder. PHP web katmanı kullanıcıdan dosya alır, `shell_exec` ile `predict.py`'yi tetikler, JSON stdout'u parse edip HTML render eder. Fallback olarak Flask micro-API mimarisi hazır tutulur.

**Tech Stack:** Python 3.11, librosa ≥0.10, scikit-learn ≥1.3, numpy, pandas, scipy, Flask (fallback), PHP ≥8.1, Bootstrap 5, XAMPP/WAMP, GTZAN Dataset.

---

## Dosya Yapısı

```
proje/
├── web/
│   ├── index.php               # Ana sayfa + upload formu
│   ├── upload.php              # Dosya kabul, Python tetikleme, JSON parse
│   ├── result.php              # Sonuç render (genre + confidence + öneriler)
│   ├── assets/
│   │   ├── css/style.css       # Özel stiller
│   │   └── js/app.js           # AJAX submit + progress bar
│   └── uploads/tmp/            # Geçici dosya dizini (yazılabilir)
│
├── scripts/
│   ├── extract_features.py     # Batch: tüm GTZAN → features.csv + feature_db.npy
│   ├── train.py                # Model eğitimi: RF + KNN + DT, cross-val, kayıt
│   ├── predict.py              # Online: tek dosya → JSON stdout
│   └── utils.py                # Paylaşılan yardımcı fonksiyonlar
│
├── models/
│   ├── model.pkl               # Seçilen en iyi sınıflandırıcı
│   ├── scaler.pkl              # Fit edilmiş StandardScaler
│   ├── pca.pkl                 # (opsiyonel) PCA transformer
│   └── feature_db.npy          # Tüm GTZAN vektörleri (öneri için)
│
├── data/
│   ├── gtzan/                  # Ham .wav dosyaları (genres/blues/... yapısı)
│   └── features.csv            # Batch çıkarım çıktısı
│
├── flask_api/
│   └── app.py                  # Plan C fallback: Flask REST endpoint
│
├── tests/
│   ├── test_utils.py
│   ├── test_extract.py
│   ├── test_train.py
│   └── test_predict.py
│
└── requirements.txt
```

---

## Task 1: Python Ortamı ve requirements.txt

**Files:**
- Create: `requirements.txt`
- Create: `scripts/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: requirements.txt oluştur**

```
librosa>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
soundfile>=0.12.0
flask>=3.0.0
pytest>=7.4.0
```

- [ ] **Step 2: Virtualenv kur ve bağımlılıkları yükle**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

Beklenen çıktı: "Successfully installed librosa-0.10.x scikit-learn-1.3.x ..."

- [ ] **Step 3: utils.py — temel sabitler ve helper'lar yaz**

```python
# scripts/utils.py
import numpy as np
from pathlib import Path

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

SAMPLE_RATE = 22050
DURATION = 30       # saniye — tam dosya
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data"


def genre_from_filename(filename: str) -> str:
    """'blues.00042.wav' → 'blues'"""
    return Path(filename).stem.split(".")[0]


def vector_labels() -> list[str]:
    """Öznitelik vektörünün sütun isimlerini döndürür."""
    cols = []
    for i in range(N_MFCC):
        cols += [f"mfcc_{i}_mean", f"mfcc_{i}_std"]
    cols += [
        "sc_mean", "sc_std",
        "zcr_mean", "zcr_std",
        "chroma_mean", "chroma_std",
        "tempo",
        "rolloff_mean", "rolloff_std",
        "bandwidth_mean", "bandwidth_std",
        "rms_mean", "rms_std",
        "contrast_mean", "contrast_std",
    ]
    return cols
```

- [ ] **Step 4: test_utils.py yaz**

```python
# tests/test_utils.py
from scripts.utils import genre_from_filename, vector_labels, GENRES

def test_genre_from_filename():
    assert genre_from_filename("blues.00042.wav") == "blues"
    assert genre_from_filename("jazz.00001.wav") == "jazz"
    assert genre_from_filename("classical.00099.wav") == "classical"

def test_vector_labels_length():
    labels = vector_labels()
    # 13 MFCC × 2 (mean+std) = 26, diğerleri = 14 → toplam 40
    assert len(labels) == 40

def test_genres_count():
    assert len(GENRES) == 10
    assert "blues" in GENRES
    assert "rock" in GENRES
```

- [ ] **Step 5: Testi çalıştır — geçmeli**

```bash
pytest tests/test_utils.py -v
```

Beklenen: 3 test PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/utils.py tests/test_utils.py
git commit -m "feat: proje iskeleti ve utils modülü"
```

---

## Task 2: Veri Seti Temizleme ve Keşif

**Files:**
- Modify: `scripts/utils.py` — `scan_dataset()` fonksiyonu ekle
- Create: `tests/test_extract.py` (ilk testler)

> **Ön koşul:** GTZAN veri seti `data/gtzan/genres/` altında düzenlenmiş olmalı.
> İndirme: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
> Dizin yapısı: `data/gtzan/genres/blues/blues.00000.wav` ...

- [ ] **Step 1: scan_dataset() fonksiyonunu utils.py'ye ekle**

```python
# scripts/utils.py içine ekle
import librosa
import logging

logger = logging.getLogger(__name__)


def scan_dataset(gtzan_root: Path) -> tuple[list[Path], list[str]]:
    """
    GTZAN dizinini tara, bozuk dosyaları atla.
    Returns: (valid_paths, skipped_paths)
    """
    valid: list[Path] = []
    skipped: list[str] = []

    genres_dir = gtzan_root / "genres"
    for wav_path in sorted(genres_dir.rglob("*.wav")):
        try:
            # Sadece metadata oku — tam yükleme değil
            info = librosa.get_samplerate(str(wav_path))
            if info < 1:
                raise ValueError("Geçersiz sample rate")
            valid.append(wav_path)
        except Exception as e:
            skipped.append(str(wav_path))
            logger.warning("Atlandı: %s — %s", wav_path.name, e)

    logger.info("Geçerli: %d | Atlandı: %d", len(valid), len(skipped))
    return valid, skipped
```

- [ ] **Step 2: test_extract.py — scan_dataset test yaz**

```python
# tests/test_extract.py
from pathlib import Path
import pytest

def test_genre_from_filename_edge_cases():
    from scripts.utils import genre_from_filename
    assert genre_from_filename("rock.00000.wav") == "rock"

def test_scan_dataset_returns_tuples(tmp_path):
    """Boş dizin → ([], []) döndürmeli"""
    from scripts.utils import scan_dataset
    genres_dir = tmp_path / "genres"
    genres_dir.mkdir()
    valid, skipped = scan_dataset(tmp_path)
    assert isinstance(valid, list)
    assert isinstance(skipped, list)
    assert len(valid) == 0
```

- [ ] **Step 3: Testi çalıştır**

```bash
pytest tests/test_extract.py -v
```

Beklenen: 2 test PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/utils.py tests/test_extract.py
git commit -m "feat: veri seti tarama ve temizlik fonksiyonu"
```

---

## Task 3: Öznitelik Çıkarım Pipeline (extract_features.py)

**Files:**
- Create: `scripts/extract_features.py`
- Modify: `tests/test_extract.py` — extraction testleri ekle

- [ ] **Step 1: test — tek dosyadan öznitelik çıkarımı testi yaz**

```python
# tests/test_extract.py içine ekle
import numpy as np

def test_extract_single_returns_correct_length(tmp_path):
    """
    Kısa bir sahte ses dosyası oluştur, 40 boyutlu vektör döndürmeli.
    """
    import soundfile as sf
    from scripts.extract_features import extract_features_from_file

    # 5 saniyelik 440 Hz sinüs dalgası oluştur
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, sr * duration)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    wav_path = tmp_path / "test_tone.wav"
    sf.write(str(wav_path), audio, sr)

    features = extract_features_from_file(str(wav_path))

    assert isinstance(features, np.ndarray)
    assert features.shape == (40,)
    assert not np.any(np.isnan(features))
```

- [ ] **Step 2: Testi çalıştır — FAIL bekleniyor**

```bash
pytest tests/test_extract.py::test_extract_single_returns_correct_length -v
```

Beklenen: ImportError veya ModuleNotFoundError

- [ ] **Step 3: extract_features.py yaz**

```python
# scripts/extract_features.py
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from scripts.utils import (
    SAMPLE_RATE, DURATION, N_MFCC, HOP_LENGTH, N_FFT,
    genre_from_filename, vector_labels, DATA_DIR, scan_dataset
)


def extract_features_from_file(file_path: str) -> np.ndarray:
    """
    Tek .wav dosyasından 40 boyutlu öznitelik vektörü çıkar.
    Raises: Exception — dosya yüklenemezse veya çok kısaysa.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    if len(y) < sr * 3:  # 3 saniyeden kısa → ret
        raise ValueError(f"Ses dosyası çok kısa: {len(y) / sr:.1f}s")

    # MFCC (13 × 2 = 26 boyut)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    # Spectral Centroid (2 boyut)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    sc_features = np.array([sc.mean(), sc.std()])

    # Zero Crossing Rate (2 boyut)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    zcr_features = np.array([zcr.mean(), zcr.std()])

    # Chroma (2 boyut)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    chroma_features = np.array([chroma.mean(), chroma.std()])

    # Tempo (1 boyut)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo_feature = np.array([float(tempo)])

    # Spectral Rolloff (2 boyut)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff_features = np.array([rolloff.mean(), rolloff.std()])

    # Spectral Bandwidth (2 boyut)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    bw_features = np.array([bw.mean(), bw.std()])

    # RMS Energy (2 boyut)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    rms_features = np.array([rms.mean(), rms.std()])

    # Spectral Contrast (2 boyut)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    contrast_features = np.array([contrast.mean(), contrast.std()])

    # Birleştir → 26 + 2 + 2 + 2 + 1 + 2 + 2 + 2 + 2 = 41... düzelt:
    # mfcc(26) + sc(2) + zcr(2) + chroma(2) + tempo(1) + rolloff(2) + bw(2) + rms(2) + contrast(2) = 41
    # NOT: utils.vector_labels() 40 döndürüyor. Bant genişliği eklenince 41 oldu.
    # → vector_labels() güncelle ya da birini çıkar. Burada rolloff+bw birleştiriliyor.
    feature_vector = np.concatenate([
        mfcc_features,      # 26
        sc_features,        # 2
        zcr_features,       # 2
        chroma_features,    # 2
        tempo_feature,      # 1
        rolloff_features,   # 2
        bw_features,        # 2
        rms_features,       # 2
        contrast_features,  # 2  → Toplam: 41
    ])

    return feature_vector.astype(np.float32)


def batch_extract(gtzan_root: Path) -> pd.DataFrame:
    """
    Tüm GTZAN dosyaları için öznitelik çıkar → DataFrame döndür.
    """
    valid_paths, skipped = scan_dataset(gtzan_root)
    print(f"İşlenecek: {len(valid_paths)} dosya | Atlanan: {len(skipped)}")

    records = []
    for i, wav_path in enumerate(valid_paths):
        try:
            features = extract_features_from_file(str(wav_path))
            genre = genre_from_filename(wav_path.name)
            records.append({**dict(zip(range(len(features)), features)), "genre": genre, "filename": wav_path.name})
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(valid_paths)} tamamlandı...")
        except Exception as e:
            print(f"  HATA [{wav_path.name}]: {e}")

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    gtzan_root = DATA_DIR / "gtzan"
    df = batch_extract(gtzan_root)
    output_path = DATA_DIR / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"Kaydedildi: {output_path} ({len(df)} satır)")
```

- [ ] **Step 4: utils.vector_labels() 41'e güncelle**

`scripts/utils.py` içindeki `vector_labels()` sonuna `"bandwidth_mean", "bandwidth_std"` zaten var. Fonksiyon 40 döndürüyor ama extract 41 çıkarıyor. Düzelt:

```python
def vector_labels() -> list[str]:
    cols = []
    for i in range(N_MFCC):
        cols += [f"mfcc_{i}_mean", f"mfcc_{i}_std"]
    cols += [
        "sc_mean", "sc_std",
        "zcr_mean", "zcr_std",
        "chroma_mean", "chroma_std",
        "tempo",
        "rolloff_mean", "rolloff_std",
        "bandwidth_mean", "bandwidth_std",
        "rms_mean", "rms_std",
        "contrast_mean", "contrast_std",
    ]
    return cols  # Toplam: 26 + 14 + 1 = 41
```

`tests/test_utils.py::test_vector_labels_length` → 41'e güncelle:
```python
assert len(labels) == 41
```

- [ ] **Step 5: Testleri çalıştır — hepsi PASS**

```bash
pytest tests/ -v
```

Beklenen: tüm testler PASS

- [ ] **Step 6: Batch extract çalıştır (GTZAN varsa)**

```bash
python scripts/extract_features.py
```

Beklenen çıktı:
```
İşlenecek: ~997 dosya | Atlanan: ~3
100/997 tamamlandı...
...
Kaydedildi: data/features.csv (997 satır)
```

- [ ] **Step 7: Commit**

```bash
git add scripts/extract_features.py scripts/utils.py tests/test_extract.py tests/test_utils.py
git commit -m "feat: batch ve tekli öznitelik çıkarım pipeline'ı"
```

---

## Task 4: Model Eğitimi (train.py)

**Files:**
- Create: `scripts/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: test_train.py yaz — küçük sahte veriye eğit ve kaydet**

```python
# tests/test_train.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def make_dummy_df(n_samples=100, n_features=41) -> pd.DataFrame:
    """10 türde eşit dağılımlı sahte öznitelik DataFrame'i."""
    from scripts.utils import GENRES
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    genres = np.tile(GENRES, n_samples // len(GENRES) + 1)[:n_samples]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["genre"] = genres
    df["filename"] = [f"dummy_{i}.wav" for i in range(n_samples)]
    return df


def test_train_returns_model_and_scaler(tmp_path):
    from scripts.train import train_classifier

    df = make_dummy_df()
    model, scaler, metrics = train_classifier(df, models_dir=tmp_path)

    assert model is not None
    assert scaler is not None
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # Dosyalar kaydedilmiş olmalı
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "scaler.pkl").exists()


def test_feature_db_saved(tmp_path):
    from scripts.train import save_feature_db

    df = make_dummy_df()
    save_feature_db(df, tmp_path)

    db = np.load(str(tmp_path / "feature_db.npy"))
    assert db.shape[0] == len(df)
```

- [ ] **Step 2: Testi çalıştır — FAIL bekleniyor**

```bash
pytest tests/test_train.py -v
```

- [ ] **Step 3: train.py yaz**

```python
# scripts/train.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scripts.utils import GENRES, MODELS_DIR, DATA_DIR


def _get_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame'den X ve y dizileri çıkar."""
    feature_cols = [c for c in df.columns if c not in ("genre", "filename")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["genre"].values
    return X, y


def train_classifier(
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
) -> tuple:
    """
    3 model eğit, en iyisini seç, kaydet.
    Returns: (best_model, scaler, metrics_dict)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y = _get_xy(df)

    # Ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
        "knn":           KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    }

    results = {}
    for name, clf in candidates.items():
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "clf": clf,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }
        print(f"  [{name}] CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f} | Test: {results[name]['test_accuracy']:.3f}")

    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best = results[best_name]
    print(f"\nEn iyi model: {best_name} (accuracy={best['test_accuracy']:.3f})")

    joblib.dump(best["clf"], models_dir / "model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")

    metrics = {
        "model_name": best_name,
        "accuracy": best["test_accuracy"],
        "f1_macro": best["test_f1_macro"],
        "cv_mean": best["cv_mean"],
        "all_models": {k: {kk: vv for kk, vv in v.items() if kk != "clf"} for k, v in results.items()},
    }
    return best["clf"], scaler, metrics


def save_feature_db(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> None:
    """Tüm öznitelik vektörlerini ve etiketleri .npy olarak kaydet (öneri için)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y = _get_xy(df)

    filenames = df["filename"].values if "filename" in df.columns else np.array([""] * len(df))

    np.save(str(models_dir / "feature_db.npy"), X)
    np.save(str(models_dir / "feature_db_labels.npy"), y)
    np.save(str(models_dir / "feature_db_filenames.npy"), filenames)
    print(f"feature_db kaydedildi: {X.shape}")


if __name__ == "__main__":
    features_csv = DATA_DIR / "features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Önce extract_features.py çalıştırın: {features_csv}")

    df = pd.read_csv(features_csv)
    print(f"Veri yüklendi: {df.shape}")

    model, scaler, metrics = train_classifier(df)
    save_feature_db(df)

    print("\n=== Model Karşılaştırma ===")
    for name, m in metrics["all_models"].items():
        print(f"  {name}: accuracy={m['test_accuracy']:.3f}, F1={m['test_f1_macro']:.3f}")
```

- [ ] **Step 4: Testleri çalıştır — PASS**

```bash
pytest tests/test_train.py -v
```

Beklenen: 2 test PASS

- [ ] **Step 5: Gerçek veriyle eğit (GTZAN varsa)**

```bash
python scripts/train.py
```

Beklenen çıktı (referans değerler):
```
  [random_forest] CV: 0.78 ± 0.02 | Test: 0.80
  [knn]           CV: 0.68 ± 0.03 | Test: 0.70
  [decision_tree] CV: 0.58 ± 0.04 | Test: 0.61

En iyi model: random_forest (accuracy=0.80)
feature_db kaydedildi: (997, 41)
```

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py tests/test_train.py
git commit -m "feat: model eğitimi — RF/KNN/DT cross-validation ve kayıt"
```

---

## Task 5: Tahmin ve Öneri Scripti (predict.py)

**Files:**
- Create: `scripts/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: test_predict.py yaz**

```python
# tests/test_predict.py
import numpy as np
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def make_fake_models(tmp_path):
    """Sahte model, scaler, feature_db oluştur."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from scripts.utils import GENRES

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 41))
    y_train = np.tile(GENRES, 10)

    scaler = StandardScaler().fit(X_train)
    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(scaler.transform(X_train), y_train)

    joblib.dump(clf, tmp_path / "model.pkl")
    joblib.dump(scaler, tmp_path / "scaler.pkl")
    np.save(str(tmp_path / "feature_db.npy"), X_train)
    np.save(str(tmp_path / "feature_db_labels.npy"), y_train)
    np.save(str(tmp_path / "feature_db_filenames.npy"), np.array([f"song_{i}.wav" for i in range(100)]))

    return clf, scaler


def test_predict_returns_valid_json(tmp_path):
    import soundfile as sf
    from scripts.predict import predict_file

    # Sahte ses
    sr = 22050
    t = np.linspace(0, 5, sr * 5)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, sr)

    make_fake_models(tmp_path)

    result = predict_file(str(wav_path), models_dir=tmp_path)

    assert "genre" in result
    assert "confidence" in result
    assert "recommendations" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) <= 3


def test_predict_error_on_missing_file(tmp_path):
    from scripts.predict import predict_file

    make_fake_models(tmp_path)
    result = predict_file("/nonexistent/path.wav", models_dir=tmp_path)

    assert "error" in result
    assert result["genre"] is None
```

- [ ] **Step 2: Testi çalıştır — FAIL**

```bash
pytest tests/test_predict.py -v
```

- [ ] **Step 3: predict.py yaz**

```python
# scripts/predict.py
"""
Ana tahmin scripti.
Kullanım: python predict.py <dosya_yolu>
Çıktı:   JSON → stdout
"""
import sys
import json
import numpy as np
import joblib
from pathlib import Path
from scipy.spatial.distance import cosine
from scripts.utils import MODELS_DIR
from scripts.extract_features import extract_features_from_file


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine(a.astype(np.float64), b.astype(np.float64)))


def predict_file(
    file_path: str,
    models_dir: Path = MODELS_DIR,
    n_recommendations: int = 3,
) -> dict:
    """
    Tek dosya için tür tahmini + öneri yap.
    Hata durumunda {"error": ..., "genre": None, ...} döndürür.
    """
    try:
        model     = joblib.load(models_dir / "model.pkl")
        scaler    = joblib.load(models_dir / "scaler.pkl")
        feat_db   = np.load(str(models_dir / "feature_db.npy"))
        labels    = np.load(str(models_dir / "feature_db_labels.npy"))
        filenames = np.load(str(models_dir / "feature_db_filenames.npy"))
    except FileNotFoundError as e:
        return {"error": f"Model dosyası bulunamadı: {e}", "genre": None,
                "confidence": 0.0, "recommendations": []}

    try:
        features = extract_features_from_file(file_path)
    except Exception as e:
        return {"error": f"Öznitelik çıkarımı başarısız: {e}", "genre": None,
                "confidence": 0.0, "recommendations": []}

    # Ölçeklendir
    features_scaled = scaler.transform(features.reshape(1, -1))[0]

    # Tahmin
    genre = model.predict(features_scaled.reshape(1, -1))[0]
    proba = model.predict_proba(features_scaled.reshape(1, -1))[0]
    confidence = float(proba.max())

    # Öneri: cosine similarity ile en yakın 3
    sims = []
    for i, db_vec in enumerate(feat_db):
        try:
            sim = _cosine_similarity(features, db_vec)
            sims.append({"idx": i, "score": sim, "file": str(filenames[i]), "genre": str(labels[i])})
        except Exception:
            continue

    sims.sort(key=lambda x: x["score"], reverse=True)

    # Aynı dosyayı hariç tut (score ~1.0 olanı atla)
    recommendations = [s for s in sims if s["score"] < 0.9999][:n_recommendations]

    return {
        "genre": genre,
        "confidence": round(confidence, 4),
        "recommendations": [
            {"file": r["file"], "genre": r["genre"], "score": round(r["score"], 4)}
            for r in recommendations
        ],
        "features": {
            "tempo": float(features[26]),   # indeks: 13×2=26'dan sonra sc(2)+zcr(2)+chroma(2)=6 → 26. index tempo
            "zcr_mean": float(features[28]), # sc_mean(26), sc_std(27), zcr_mean(28)
        },
        "error": None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Kullanım: python predict.py <dosya_yolu>", "genre": None,
                          "confidence": 0.0, "recommendations": []}))
        sys.exit(1)

    result = predict_file(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))
```

> **Not:** `features[26]` ile tempo erişimi: MFCC 0-25 (13×2), sc_mean=26, sc_std=27, zcr_mean=28, zcr_std=29, chroma_mean=30, chroma_std=31, tempo=32. Yukarıdaki yorumları düzelt:
> ```python
> "tempo":    float(features[32]),
> "zcr_mean": float(features[28]),
> ```

- [ ] **Step 4: Testleri çalıştır — PASS**

```bash
pytest tests/test_predict.py -v
```

- [ ] **Step 5: El ile test**

```bash
# GTZAN'dan gerçek bir dosyayla dene
python scripts/predict.py data/gtzan/genres/jazz/jazz.00042.wav
```

Beklenen:
```json
{"genre": "jazz", "confidence": 0.82, "recommendations": [...], "features": {...}, "error": null}
```

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py tests/test_predict.py
git commit -m "feat: tahmin ve cosine similarity öneri sistemi"
```

---

## Task 6: PHP Web Katmanı — Dosya Yükleme ve Python Tetikleme

**Files:**
- Create: `web/index.php`
- Create: `web/upload.php`
- Create: `web/assets/js/app.js`

> **Ön koşul:** XAMPP kurulu, `web/` klasörü XAMPP `htdocs/proje/` altına symlink veya kopyalanmış.

- [ ] **Step 1: uploads/tmp dizini ve .gitkeep**

```bash
mkdir web/uploads/tmp
echo "" > web/uploads/tmp/.gitkeep
```

- [ ] **Step 2: index.php yaz**

```php
<?php
// web/index.php
$maxSizeMB = 10;
$allowedExts = ['wav', 'mp3'];
?>
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Müzik Türü Sınıflandırma</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="assets/css/style.css" rel="stylesheet">
</head>
<body class="bg-dark text-light">
<div class="container py-5">
    <h1 class="text-center mb-2">🎵 Müzik Türü Sınıflandırıcı</h1>
    <p class="text-center text-muted mb-5">Ses dosyanızı yükleyin, tür tahmini ve benzer şarkı önerileri alın.</p>

    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card bg-secondary text-light p-4">
                <form id="uploadForm" action="upload.php" method="POST" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="audioFile" class="form-label">Ses Dosyası (.wav veya .mp3)</label>
                        <input class="form-control" type="file" id="audioFile" name="audioFile"
                               accept=".wav,.mp3" required>
                        <div class="form-text text-muted">Maksimum <?= $maxSizeMB ?> MB</div>
                    </div>
                    <button type="submit" class="btn btn-primary w-100" id="submitBtn">
                        Sınıflandır
                    </button>
                </form>

                <div id="progressSection" class="mt-3 d-none">
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated w-100"
                             role="progressbar">Analiz ediliyor...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="resultSection" class="row justify-content-center mt-4"></div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="assets/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 3: upload.php yaz**

```php
<?php
// web/upload.php
header('Content-Type: application/json; charset=utf-8');
set_time_limit(120);

$allowedExts  = ['wav', 'mp3'];
$maxSizeBytes = 10 * 1024 * 1024; // 10 MB
$uploadDir    = __DIR__ . '/uploads/tmp/';
$pythonBin    = 'C:\\Python311\\python.exe'; // Windows — gerekirse düzenle
$predictScript = realpath(__DIR__ . '/../scripts/predict.py');

// Dosya gelmedi mi?
if (!isset($_FILES['audioFile']) || $_FILES['audioFile']['error'] !== UPLOAD_ERR_OK) {
    http_response_code(400);
    echo json_encode(['error' => 'Dosya yüklenemedi.', 'genre' => null]);
    exit;
}

$file     = $_FILES['audioFile'];
$ext      = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
$mimeType = mime_content_type($file['tmp_name']);

// Doğrulama
if (!in_array($ext, $allowedExts, true)) {
    http_response_code(422);
    echo json_encode(['error' => 'Geçersiz dosya uzantısı. Yalnızca .wav veya .mp3 kabul edilir.', 'genre' => null]);
    exit;
}

if ($file['size'] > $maxSizeBytes) {
    http_response_code(422);
    echo json_encode(['error' => 'Dosya çok büyük. Maksimum 10 MB.', 'genre' => null]);
    exit;
}

// Güvenli geçici dosya adı
$tmpFile = $uploadDir . bin2hex(random_bytes(8)) . '.' . $ext;

if (!move_uploaded_file($file['tmp_name'], $tmpFile)) {
    http_response_code(500);
    echo json_encode(['error' => 'Dosya kaydedilemedi.', 'genre' => null]);
    exit;
}

// Python çalıştır
$cmd    = escapeshellcmd($pythonBin) . ' ' . escapeshellarg($predictScript) . ' ' . escapeshellarg($tmpFile) . ' 2>&1';
$output = shell_exec($cmd);

// Geçici dosyayı sil
@unlink($tmpFile);

if ($output === null) {
    http_response_code(500);
    echo json_encode(['error' => 'Python scripti çalıştırılamadı.', 'genre' => null]);
    exit;
}

// JSON parse
$data = json_decode(trim($output), true);

if ($data === null) {
    http_response_code(500);
    echo json_encode(['error' => 'Python çıktısı parse edilemedi: ' . substr($output, 0, 200), 'genre' => null]);
    exit;
}

echo json_encode($data);
```

- [ ] **Step 4: app.js yaz**

```javascript
// web/assets/js/app.js
document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const submitBtn      = document.getElementById('submitBtn');
    const progressSection = document.getElementById('progressSection');
    const resultSection  = document.getElementById('resultSection');

    submitBtn.disabled = true;
    progressSection.classList.remove('d-none');
    resultSection.innerHTML = '';

    const formData = new FormData(this);

    try {
        const response = await fetch('upload.php', { method: 'POST', body: formData });
        const data = await response.json();
        renderResult(data);
    } catch (err) {
        resultSection.innerHTML = `<div class="col-12"><div class="alert alert-danger">İstek başarısız: ${err.message}</div></div>`;
    } finally {
        submitBtn.disabled = false;
        progressSection.classList.add('d-none');
    }
});

function renderResult(data) {
    const section = document.getElementById('resultSection');

    if (data.error) {
        section.innerHTML = `<div class="col-12"><div class="alert alert-danger">${data.error}</div></div>`;
        return;
    }

    const confidence = (data.confidence * 100).toFixed(1);
    const recsHtml = (data.recommendations || []).map(r =>
        `<li class="list-group-item bg-dark text-light border-secondary">
            <strong>${r.genre}</strong> — ${r.file}
            <span class="badge bg-info float-end">${(r.score * 100).toFixed(1)}% benzer</span>
         </li>`
    ).join('');

    section.innerHTML = `
        <div class="col-md-8">
            <div class="card bg-secondary text-light p-4">
                <h3 class="text-center mb-3">
                    Tür: <span class="badge bg-success fs-4">${data.genre.toUpperCase()}</span>
                </h3>
                <div class="mb-3">
                    <label class="form-label">Güven Skoru</label>
                    <div class="progress" style="height:24px;">
                        <div class="progress-bar bg-success" style="width:${confidence}%">${confidence}%</div>
                    </div>
                </div>
                ${recsHtml ? `
                <h5 class="mt-3">Benzer Şarkılar</h5>
                <ul class="list-group">${recsHtml}</ul>` : ''}
            </div>
        </div>`;
}
```

- [ ] **Step 5: style.css oluştur**

```css
/* web/assets/css/style.css */
body { min-height: 100vh; }
.card { border-radius: 12px; }
.badge { border-radius: 6px; }
```

- [ ] **Step 6: XAMPP'te test et**

1. XAMPP Apache başlat
2. `web/` → `htdocs/muzik-sinif/` olarak kopyala
3. `http://localhost/muzik-sinif/` aç
4. Gerçek bir .wav yükle, sonucu gözlemle

- [ ] **Step 7: Commit**

```bash
git add web/
git commit -m "feat: PHP upload + AJAX frontend + sonuç kartları"
```

---

## Task 7: Flask Fallback API (Plan C)

**Files:**
- Create: `flask_api/app.py`

> Bu task yalnızca `shell_exec` çalışmazsa aktive edilir.

- [ ] **Step 1: flask_api/app.py yaz**

```python
# flask_api/app.py
import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from scripts.predict import predict_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio alanı gerekli', 'genre': None}), 400

    audio_file = request.files['audio']
    ext = Path(audio_file.filename).suffix.lower()

    if ext not in ('.wav', '.mp3'):
        return jsonify({'error': 'Yalnızca .wav/.mp3 kabul edilir', 'genre': None}), 422

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
```

- [ ] **Step 2: Flask'ı başlat ve test et**

```bash
python flask_api/app.py
```

```bash
# Ayrı terminalde:
curl -X POST http://localhost:5000/predict \
  -F "audio=@data/gtzan/genres/jazz/jazz.00042.wav"
```

Beklenen: `{"genre": "jazz", "confidence": ..., "recommendations": [...]}`

- [ ] **Step 3: upload.php'yi Flask moduna geç (gerekirse)**

`upload.php` içindeki `shell_exec` bloğunu şununla değiştir:

```php
// Flask API çağrısı — Plan C
$ch = curl_init('http://127.0.0.1:5000/predict');
curl_setopt_array($ch, [
    CURLOPT_POST           => true,
    CURLOPT_POSTFIELDS     => ['audio' => new CURLFile($tmpFile)],
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_TIMEOUT        => 60,
]);
$output = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);
@unlink($tmpFile);

if ($httpCode !== 200) {
    http_response_code(500);
    echo json_encode(['error' => 'Flask API hatası', 'genre' => null]);
    exit;
}
$data = json_decode($output, true);
echo json_encode($data);
```

- [ ] **Step 4: Commit**

```bash
git add flask_api/app.py web/upload.php
git commit -m "feat: Flask fallback API (Plan C) ve cURL entegrasyonu"
```

---

## Task 8: Uçtan Uca Entegrasyon Testi

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: test_integration.py yaz**

```python
# tests/test_integration.py
"""
Her 10 türden birer şarkıyla uçtan uca pipeline test et.
GTZAN mevcut olduğunda çalıştır: pytest tests/test_integration.py -v -m integration
"""
import pytest
import json
from pathlib import Path
from scripts.utils import GENRES, DATA_DIR, MODELS_DIR


@pytest.mark.integration
@pytest.mark.parametrize("genre", GENRES)
def test_predict_known_genre(genre):
    """Her türden ilk şarkıyı tahmin et, sonucun geçerli olduğunu doğrula."""
    from scripts.predict import predict_file

    genre_dir = DATA_DIR / "gtzan" / "genres" / genre
    if not genre_dir.exists():
        pytest.skip(f"GTZAN verisi yok: {genre_dir}")

    wav_files = list(genre_dir.glob("*.wav"))
    if not wav_files:
        pytest.skip(f"Dosya yok: {genre_dir}")

    result = predict_file(str(wav_files[0]))

    assert result["error"] is None, f"Hata: {result['error']}"
    assert result["genre"] in GENRES
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["recommendations"], list)


@pytest.mark.integration
def test_predict_wrong_format(tmp_path):
    """Sahte .txt dosyası → error döndürmeli"""
    from scripts.predict import predict_file

    fake = tmp_path / "notaudio.wav"
    fake.write_text("bu ses değil")

    result = predict_file(str(fake))
    assert result["error"] is not None
```

- [ ] **Step 2: Entegrasyon testlerini çalıştır**

```bash
pytest tests/test_integration.py -v -m integration
```

Beklenen: 10 tür için PASS (GTZAN varsa), geçersiz format testi PASS

- [ ] **Step 3: Tüm testleri çalıştır**

```bash
pytest tests/ -v --ignore=tests/test_integration.py
```

Beklenen: tüm birim testler PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: uçtan uca entegrasyon testleri"
```

---

## Task 9: PCA Karşılaştırması (Akademik Rapor için)

**Files:**
- Modify: `scripts/train.py` — PCA dallı eğitim ekle
- Create: `scripts/visualize.py` — confusion matrix ve görseller

- [ ] **Step 1: train.py'e PCA karşılaştırması ekle**

`train.py` sonuna şunu ekle:

```python
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def train_with_pca(df: pd.DataFrame, n_components: int = 20, models_dir: Path = MODELS_DIR) -> dict:
    """PCA ile boyut indirgeme yaparak model eğit, karşılaştır."""
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y = _get_xy(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    joblib.dump(pca, models_dir / "pca.pkl")
    print(f"PCA ({n_components} bileşen): accuracy={acc:.3f}, F1={f1:.3f}")
    print(f"Açıklanan varyans: {pca.explained_variance_ratio_.sum():.3f}")

    return {"accuracy": acc, "f1_macro": f1, "explained_variance": float(pca.explained_variance_ratio_.sum())}
```

- [ ] **Step 2: visualize.py yaz**

```python
# scripts/visualize.py
"""
Akademik rapor için confusion matrix ve öznitelik dağılımı görselleri üret.
Çalıştır: python scripts/visualize.py
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scripts.utils import GENRES, DATA_DIR, MODELS_DIR


def plot_confusion_matrix(output_path: str = "confusion_matrix.png") -> None:
    df = pd.read_csv(DATA_DIR / "features.csv")
    feature_cols = [c for c in df.columns if c not in ("genre", "filename")]
    X = df[feature_cols].values
    y = df["genre"].values

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    model  = joblib.load(MODELS_DIR / "model.pkl")

    X_scaled = scaler.transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=GENRES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Random Forest (GTZAN Test Set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Kaydedildi: {output_path}")


if __name__ == "__main__":
    plot_confusion_matrix("confusion_matrix.png")
```

- [ ] **Step 3: Görselleri üret**

```bash
python scripts/visualize.py
```

Beklenen: `confusion_matrix.png` üretilir.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py scripts/visualize.py
git commit -m "feat: PCA karşılaştırması ve confusion matrix görselleştirme"
```

---

## Task 10: README ve Kurulum Dokümanı

**Files:**
- Create: `README.md`

- [ ] **Step 1: README.md yaz**

```markdown
# Müzik Türü Sınıflandırma ve Öneri Sistemi

Pattern Recognition dersi projesi. GTZAN veri seti üzerinde akustik öznitelik çıkarımı (librosa) ile Random Forest sınıflandırıcısı ve cosine similarity öneri sistemi.

## Kurulum

### 1. Python Ortamı
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. GTZAN Veri Seti
Kaggle'dan indirin → `data/gtzan/genres/` altına yerleştirin.
Dizin yapısı: `data/gtzan/genres/blues/blues.00000.wav`

### 3. Eğitim Pipeline
```bash
python scripts/extract_features.py   # ~10 dk, features.csv üretir
python scripts/train.py              # model.pkl, scaler.pkl, feature_db.npy üretir
```

### 4. Web Arayüzü (XAMPP)
- XAMPP kur, Apache başlat
- `web/` → `htdocs/muzik-sinif/` kopyala
- `upload.php` içinde `$pythonBin` yolunu doğrula
- `http://localhost/muzik-sinif/` aç

### 5. Flask Fallback (opsiyonel)
```bash
python flask_api/app.py   # http://localhost:5000
```

## Test
```bash
pytest tests/ -v                               # birim testler
pytest tests/test_integration.py -m integration  # GTZAN gerekli
```

## Beklenen Doğruluk
| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Random Forest | ~0.78 | ~0.80 |
| K-NN | ~0.68 | ~0.70 |
| Decision Tree | ~0.58 | ~0.61 |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: kurulum ve kullanım kılavuzu"
```

---

## Özet — Haftalık Task Dağılımı

| Hafta | Tasklar | Çıktı |
|---|---|---|
| Hafta 1 | Task 1, Task 2 | Ortam, utils, veri tarama |
| Hafta 2 | Task 3 | features.csv, feature_db.npy |
| Hafta 3 | Task 4, Task 5 | model.pkl, predict.py |
| Hafta 4 | Task 6, Task 7, Task 8 | Web arayüzü, entegrasyon |
| Hafta 5 | Task 9, Task 10 | PCA karşılaştırma, rapor görselleri, README |

## Kişi Rolleri

| Task | K1 (Veri & ML) | K2 (Python Backend) | K3 (PHP & Frontend) |
|---|---|---|---|
| 1 | Ortam kur | Ortam kur | Ortam kur |
| 2 | scan_dataset, veri analizi | — | — |
| 3 | batch_extract | extract_features_from_file | — |
| 4 | train.py tüm modeller | save_feature_db | — |
| 5 | — | predict.py | — |
| 6 | — | — | index.php, upload.php, app.js |
| 7 | — | flask_api/app.py | PHP cURL entegrasyonu |
| 8 | — | Entegrasyon testleri | — |
| 9 | visualize.py, rapor | — | — |
| 10 | Akademik rapor | README | Sunum |
