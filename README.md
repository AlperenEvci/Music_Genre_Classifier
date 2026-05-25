# Müzik Türü Sınıflandırma ve Öneri Sistemi

Pattern Recognition dersi projesi. GTZAN veri seti üzerinde akustik öznitelik çıkarımı ile 10 müzik türünü sınıflandıran ve cosine similarity tabanlı içerik öneri sistemi sunan web uygulaması.

**Ekip:** 3 kişi | **Süre:** 5 hafta | **Ders:** Örüntü Tanıma

---

## Sistem Mimarisi

```
[Tarayıcı] → [PHP: upload.php] → [Python: predict.py] → [JSON] → [PHP: render] → [Tarayıcı]
```

- **Python Pipeline:** librosa → 41 boyutlu öznitelik vektörü → scikit-learn RF sınıflandırıcı → cosine similarity öneri
- **Web Katmanı:** PHP 8.1 + Bootstrap 5 + Vanilla JS (AJAX)
- **Fallback:** Flask REST API (shell_exec çalışmazsa)

---

## Kurulum

### 1. Python Ortamı

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. GTZAN Veri Seti

Kaggle'dan indirin: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

`data/gtzan/genres/` altına yerleştirin:
```
data/gtzan/genres/blues/blues.00000.wav
data/gtzan/genres/jazz/jazz.00000.wav
...
```

### 3. Eğitim Pipeline (sırayla çalıştır)

```bash
# Adım 1: Öznitelik çıkarımı (~10 dk, 1000 dosya)
python scripts/extract_features.py
# Çıktı: data/features.csv

# Adım 2: Model eğitimi
python scripts/train.py
# Çıktı: models/model.pkl, models/scaler.pkl, models/feature_db.npy
```

### 4. Web Arayüzü (XAMPP)

1. XAMPP kur, Apache başlat
2. `web/` → `C:\xampp\htdocs\muzik-sinif\` kopyala
3. `web/upload.php` içindeki `$pythonBin` yolunu kontrol et:
   ```php
   $pythonBin = realpath(__DIR__ . '/../venv/Scripts/python.exe');
   ```
4. `http://localhost/muzik-sinif/` aç

### 5. Flask Fallback (opsiyonel — shell_exec çalışmazsa)

```bash
python flask_api/app.py   # http://localhost:5000
```

`web/upload.php` içindeki Plan C yorumunu uncomment et.

---

## Test

```bash
# Birim testler
python -m pytest tests/ -v --ignore=tests/test_integration.py

# Entegrasyon testleri (GTZAN + eğitilmiş model gerekli)
python -m pytest tests/test_integration.py -v -m integration
```

---

## Öznitelik Vektörü (41 boyut)

| Öznitelik | Boyut | Açıklama |
|---|---|---|
| MFCC (mean+std) | 26 | Mel-frekans kepstral katsayıları |
| Spectral Centroid | 2 | Spektrum ağırlık merkezi |
| Zero Crossing Rate | 2 | Sıfır geçiş hızı |
| Chroma | 2 | 12 yarım ton enerji dağılımı |
| Tempo | 1 | BPM tahmini |
| Spectral Rolloff | 2 | %85 enerji sınır frekansı |
| Spectral Bandwidth | 2 | Spektral yayılım genişliği |
| RMS Energy | 2 | Ortalama sinyal enerjisi |
| Spectral Contrast | 2 | Band tepe-çukur enerji farkı |

---

## Beklenen Model Doğruluğu (GTZAN)

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Random Forest | ~%78 | ~%80 |
| K-NN | ~%68 | ~%70 |
| Decision Tree | ~%58 | ~%61 |

---

## Dizin Yapısı

```
proje/
├── scripts/          # Python ML pipeline
├── web/              # PHP + Bootstrap5 arayüz
├── flask_api/        # Flask fallback API
├── tests/            # pytest testleri
├── models/           # Eğitilmiş modeller (git-ignored)
├── data/             # GTZAN + features.csv (git-ignored)
└── requirements.txt
```

---

## Akademik Sorular

**MFCC nasıl hesaplanır?**
Ham ses → Pre-emphasis → Framing + Windowing → FFT → Mel Filterbank → Log → DCT → MFCC katsayıları.

**Neden Cosine Similarity, Euclidean değil?**
Cosine similarity vektör büyüklüğünden bağımsız olarak yönleri karşılaştırır. Normalize edilmemiş öznitelik vektörlerinde magnitude farkı benzerlik skorunu bozar; cosine bu etkiyi ortadan kaldırır.
