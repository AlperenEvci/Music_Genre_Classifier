# Müzik Türü Sınıflandırma ve Öneri Sistemi

**Ders:** Örüntü Tanıma (Pattern Recognition)  
**Veri Seti:** GTZAN (1000 × 30 saniyelik WAV, 10 tür)  
**Durum:** Tam çalışır

---

## Proje Amacı

GTZAN veri seti üzerinde akustik öznitelik çıkarımıyla 10 müzik türünü sınıflandıran, cosine similarity ile benzer şarkı öneren, PHP web arayüzüyle kullanıcıya sunan uçtan uca sistem.

**10 tür:** blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

---

## Mimari

```
[Tarayıcı] → [PHP: upload.php] → [Python: predict.py] → [JSON stdout] → [PHP render] → [Tarayıcı]
```

3 katman:
- **Python ML Pipeline** — offline eğitim, online tahmin
- **PHP Web Katmanı** — dosya yükleme, shell_exec tetikleme
- **Flask Fallback** — shell_exec çalışmazsa Plan C (localhost:5000)

---

## Teknoloji Stack

| Katman | Teknoloji |
|---|---|
| Öznitelik çıkarımı | Python 3.14, librosa ≥0.10 |
| ML modelleri | scikit-learn ≥1.3 (RF, KNN, DT) |
| Veri işleme | numpy, pandas, scipy |
| Web backend | PHP 8.1, XAMPP/Apache |
| Frontend | Bootstrap 5, Vanilla JS (AJAX) |
| Fallback API | Flask 3.0 |
| Paket yönetimi | uv 0.11.15 |

---

## Dosya Yapısı

```
pattern-recogtion-project/
├── scripts/
│   ├── utils.py            # Sabitler, yardımcı fonksiyonlar
│   ├── extract_features.py # 41 boyutlu öznitelik çıkarımı
│   ├── train.py            # RF/KNN/DT eğitimi, cross-validation
│   ├── predict.py          # Tahmin + cosine similarity öneri
│   └── visualize.py        # Confusion matrix PNG üretimi
├── web/
│   ├── index.php           # Bootstrap 5 upload formu
│   ├── upload.php          # Dosya validasyon + shell_exec tetikleyici
│   └── assets/
│       ├── js/app.js       # AJAX submit + XSS korumalı render
│       └── css/style.css   # Dark theme overrides
├── flask_api/
│   └── app.py              # Plan C: /predict POST, /health GET
├── tests/
│   ├── test_utils.py
│   ├── test_extract.py
│   ├── test_train.py
│   ├── test_predict.py
│   ├── test_visualize.py
│   └── test_integration.py
├── data/
│   ├── gtzan/              # GTZAN veri seti buraya gelecek
│   └── features.csv        # Çıkarılmış öznitelikler (eğitim sonrası)
├── models/
│   ├── model.pkl           # Eğitilmiş en iyi model (RF)
│   ├── scaler.pkl          # StandardScaler
│   └── feature_db.npy      # Cosine similarity için öznitelik veritabanı
├── requirements.txt
└── pytest.ini
```

---

## Öznitelik Vektörü (41 boyut)

| Öznitelik | Boyut | Açıklama |
|---|---|---|
| MFCC mean | 13 | Mel-frequency cepstral coefficients ortalaması |
| MFCC std | 13 | MFCC standart sapması |
| Spectral Centroid | 2 | Frekans ağırlık merkezi (mean+std) |
| Zero Crossing Rate | 2 | Sıfır geçiş oranı (mean+std) |
| Chroma | 2 | 12 nota sınıfı enerji dağılımı (mean+std) |
| Tempo | 1 | BPM |
| Spectral Rolloff | 2 | %85 enerji frekans eşiği (mean+std) |
| Spectral Bandwidth | 2 | Spektral genişlik (mean+std) |
| RMS Energy | 2 | Kök ortalama kare enerji (mean+std) |
| Spectral Contrast | 2 | Peak-valley fark (mean+std) |
| **TOPLAM** | **41** | |

---

## Model Sonuçları

| Model | CV Accuracy | Test Accuracy | F1 Macro |
|---|---|---|---|
| **Random Forest** | 68.8% ± 4.0% | **69.5%** | 69.4% |
| K-NN | 64.2% ± 2.7% | 65.5% | 65.8% |
| Decision Tree | 49.1% ± 1.8% | 46.5% | 46.8% |

**Not:** GTZAN literatür normu RF için %70-80. Bizim %69.5 beklenen aralıkta.  
Eğitim verisi: 999/1000 WAV (1 corrupt dosya otomatik atlandı).

---

## Tahmin Çıktısı (JSON)

```json
{
  "genre": "jazz",
  "confidence": 0.805,
  "recommendations": [
    {"file": "jazz.00068.wav", "genre": "jazz", "score": 0.9998},
    {"file": "jazz.00031.wav", "genre": "jazz", "score": 0.9997},
    {"file": "blues.00012.wav", "genre": "blues", "score": 0.9991}
  ],
  "features": {
    "tempo": 73.83,
    "zcr_mean": 0.061212
  },
  "error": null
}
```

---

## Test Sonuçları

18/18 test PASS.

```
tests/test_utils.py         4 test  — genre parse, vector_labels (41 boyut, unique), genres set
tests/test_extract.py       5 test  — scan_dataset, shape=41, kısa ses exception
tests/test_train.py         4 test  — model/scaler/metrics dönüşü, feature_db shape, PCA
tests/test_predict.py       4 test  — valid result, recommendation scores, missing file, missing model
tests/test_visualize.py     1 test  — FileNotFoundError without data
tests/test_integration.py   2 test  — synthetic audio sınıflandırma (GTZAN skip)
```

Çalıştırma:
```bash
pytest tests/ -v
```

---

## Kurulum

```bash
# 1. Python ortamı
uv venv
.venv\Scripts\activate      # Windows
uv pip install -r requirements.txt

# 2. GTZAN veri seti
# Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
# → data/gtzan/genres/ altına koy (genres/blues/, genres/jazz/, ...)

# 3. Eğitim pipeline (~15 dk)
python -m scripts.extract_features   # → data/features.csv
python -m scripts.train              # → models/model.pkl, scaler.pkl, feature_db.npy

# 4. CLI test
python -m scripts.predict data/gtzan/genres/jazz/jazz.00042.wav

# 5. Web arayüzü (XAMPP)
# web/ → C:\xampp\htdocs\muzik-sinif\ kopyala
# upload.php içinde Python path güncelle:
#   $pythonBin = 'C:\\...\\pattern-recogtion-project\\.venv\\Scripts\\python.exe';
# Apache başlat → http://localhost/muzik-sinif/

# 6. Flask fallback (opsiyonel)
python flask_api/app.py     # → http://localhost:5000
```

---

## Güvenlik Önlemleri

- `escapeshellarg()` — shell injection koruması
- MIME type hard-block — magic byte kontrolü (WAV: `RIFF`, MP3: `ID3`/`\xff\xfb`)
- Dosya uzantı + boyut validasyonu (max 80MB)
- `htmlspecialchars()` / `escapeHtml()` — XSS koruması
- Path traversal koruması — basename() + temp dir
- Error bilgisi maskeleme — PHP kullanıcıya Python stack trace göstermiyor

---

## Karşılaşılan Önemli Hatalar ve Çözümler

| Hata | Neden | Çözüm |
|---|---|---|
| `ModuleNotFoundError: No module named 'scripts'` | `python scripts/x.py` yerine `-m` gerekir | `python -m scripts.extract_features` |
| `only 0-dimensional arrays can be converted` | librosa 0.10+ `beat_track()` array döndürüyor | `float(np.squeeze(tempo))` |
| PHP JSON yerine HTML dönüyor | 64MB dosya php.ini limitini aştı | `upload_max_filesize=80M`, `post_max_size=85M` |
| XAMPP'tan predict.py çalışmıyor | Working directory proje root değil | `predict.py` içine `sys.path.insert(0, PROJECT_ROOT)` |
| `vector_labels()` MFCC sırası yanlış | interleaved vs all-means-then-stds uyumsuzluğu | `vector_labels()` sıralama düzeltildi |
| `@unlink($tmpFile)` undefined variable | MIME check bloğunda `$tmpFile` henüz atanmamış | `@unlink` kaldırıldı |

---

## Gerçek Şarkı Test Sonuçları

| Şarkı | Beklenen | Model | Confidence | Not |
|---|---|---|---|---|
| jazz.00042.wav (GTZAN) | jazz | **jazz ✓** | 80.5% | Doğru |
| Manifest - Hileli | hiphop | **hiphop ✓** | 45% | Hibrit trap, düşük confidence normal |
| Amy Winehouse - Back to Black | soul/r&b | disco | 23% | GTZAN'da soul yok → doğru davranış |
| Metallica - Nothing Else Matters | metal | classical | 38% | Akustik intro 30 sn dominant |
| Metallica - Enter Sandman | metal | disco | 24% | Tempo tespiti kayması |

**Önemli:** Model sadece ilk 30 saniyeye bakıyor. GTZAN'da olmayan türler (soul, trap) düşük confidence üretir — bu beklenen davranış, "bilmiyorum" sinyali.

---

## Akademik Sorular

**MFCC nasıl hesaplanır?**  
Ses → Pre-emphasis → Framing + Windowing → FFT → Mel Filterbank → Log → DCT → 13 katsayı

**Neden Cosine Similarity, Euclidean değil?**  
Cosine vektör büyüklüğünden bağımsız, sadece yönü karşılaştırır. Normalize edilmemiş özniteliklerde magnitude farkı Euclidean'ı yanıltır; cosine bunu ortadan kaldırır.

**Neden %70 altında?**  
GTZAN literatür normu %70-80. %69.5 bu aralığın alt sınırında. Rock/Metal ve Jazz/Blues çiftleri en çok karışıyor — akustik olarak benzer türler.

**Model domain dışı girdiyi nasıl ele alıyor?**  
Confidence düşüyor (Amy Winehouse %23, soul GTZAN'da yok). Eşik belirlenmemiş, ama düşük confidence kullanıcıya gösteriliyor.

---

## Commit Geçmişi

| Commit | İçerik |
|---|---|
| `e8ccdb0` | fix: resolve runtime bugs for local XAMPP deployment |
| `1690109` | fix: vector_labels sıralama, upload.php undefined var, PCA uyarısı, Flask magic byte check |
| `4d496e2` | docs: kurulum ve kullanım kılavuzu |
| `125f2be` | feat: PCA karşılaştırması ve confusion matrix görselleştirme |
| `3231971` | test: uçtan uca entegrasyon testleri |
| `869ef30` | feat: Flask fallback API (Plan C) ve cURL entegrasyonu |
| `2c3485b` | fix: MIME hard-block ve Python error information disclosure |
| `743b0a6` | feat: PHP upload + AJAX frontend + Bootstrap sonuç kartları |
| `27ef9a3` | feat: tahmin ve cosine similarity öneri sistemi |
| `3712036` | feat: model eğitimi RF/KNN/DT cross-validation ve kayıt |
| `7c0f892` | feat: öznitelik çıkarım pipeline (41 boyut, TDD) |
| `b99c30c` | feat: veri seti tarama ve temizlik fonksiyonu |
| `60f200b` | fix: utils immutability, validation, test coverage |
| `4354a93` | feat: proje iskeleti ve utils modülü |
