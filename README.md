# Profesyonel Müzik Türü Sınıflandırma ve Analiz Sistemi

Bu proje, bir ses dosyasının akustik özelliklerini çıkararak **Makine Öğrenmesi (Random Forest)** algoritmalarıyla müzik türünü tahmin eden ve benzer şarkılar öneren modern bir web uygulamasıdır.

Uygulama son güncellemelerle birlikte; **Kural Tabanlı XAI (Karar Özeti)**, **Radar Grafikleri**, **Şarkı Dinleme (Playback)** ve **Glassmorphism UI** ile tam teşekküllü bir portfolyo / veri bilimi projesine dönüştürülmüştür.

---

## 🏗️ Sistem Mimarisi (Ayrık Yapı)

Sistem profesyonel bir ayrık (decoupled) mimari kullanır:
1. **Frontend (Önyüz):** PHP 8.1 ve Vanilla JS (Docker Container içinde çalışır). Müzik yükleme, arayüz yönetimi ve veri görselleştirme (Chart.js) işlemlerini yapar.
2. **Backend (Yapay Zeka):** Python Flask REST API (Makinenizde Local olarak çalışır). Model tahmini, Kosinüs Benzerliği ve XAI açıklamalarını üretir.

```text
[Kullanıcı / Tarayıcı] → (Port 8080) → [PHP Docker Container] → (cURL) → [Python Flask API] → (Tahmin & JSON)
```

---

## 🚀 Hızlı Kurulum Rehberi

Projeyi başka bir bilgisayarda sıfırdan çalıştırmak için aşağıdaki adımları sırasıyla uygulayın.

### Adım 1: Python Yapay Zeka Sunucusunu Kurmak
Sistemdeki tüm matematiksel hesaplamaları Python yapacaktır.

```bash
# 1. Proje dizinine gidin
cd Music_Genre_Classifier

# 2. Sanal ortam (Virtual Environment) oluşturun
python -m venv .venv

# 3. Sanal ortamı aktif edin
.venv\Scripts\activate      # Windows için
# source .venv/bin/activate  # Mac/Linux için

# 4. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 5. Yapay Zeka Modelini Eğitin (Kısa sürer)
# Bu işlem GitHub'dan gelen features.csv dosyasını kullanarak model.pkl ve scaler.pkl dosyalarını oluşturur.
python scripts/train.py

# 6. Flask API sunucusunu başlatın
python -m flask_api.app
```
*(Sunucu `http://0.0.0.0:5000` adresinde çalışmaya başlayacak ve arkada açık kalmalıdır).*

### Adım 2: Veri Setini (Şarkıları) Yüklemek
Önerilen şarkıları dinleyebilmek için orijinal ses dosyalarına ihtiyacınız var:
1. Kaggle'dan [GTZAN Veri Setini (1.2 GB)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) indirin.
2. Dosyaları şu dizine yerleştirin: `data/gtzan/genres/`
*(Örnek tam yol: `data/gtzan/genres/blues/blues.00000.wav`)*

### Adım 3: Web Arayüzünü Docker ile Çalıştırmak
Bilgisayarınıza XAMPP vb. kurmanıza gerek kalmadan arayüzü doğrudan Docker ile ayağa kaldırabilirsiniz.

Yeni bir terminal açın ve şu komutu çalıştırın:
```bash
docker run -d -p 8080:80 --name music_web -v "%cd%\web":/var/www/html php:8.1-apache
# Mac/Linux için "%cd%" yerine "$PWD" kullanın.
```
*(Bu komut `web` klasörünüzü sanal bir Apache sunucusuna bağlar ve `http://localhost:8080` adresinden yayına açar).*

---

## 🧠 Yeni Özellikler Neler?

* **Kural Tabanlı XAI (Açıklanabilir Yapay Zeka):** Model artık sadece "Bu Rock" demekle kalmıyor; *Z-Score* ve *StandardScaler* kullanarak şarkının ortalamadan sapan en belirgin özelliklerini matematiksel olarak tespit edip Türkçe bir karar özeti üretiyor.
* **Akustik Karakteristik Radar Grafiği:** Tempo, Parlaklık, Enerji, Pürüzlülük ve Harmonik Yapı özellikleri `Chart.js` ile örümcek ağı grafiğinde görselleştirildi.
* **Benzer Şarkı Dinleme (Playback):** Kosinüs benzerliği (Cosine Similarity) formülü hatalı hesaplamalara karşı düzeltildi. Çıkan en benzer 3 şarkı artık doğrudan web arayüzündeki `Play` butonlarına basılarak dinlenebilir.
* **Modern UI:** "Glassmorphism" (Buzlu cam) tasarımı, sürükle-bırak desteği ve dinamik animasyonlarla donatıldı.

---

## 📂 Dizin Yapısı

```
Music_Genre_Classifier/
├── data/             # GTZAN klasörü ve features.csv burada yer alır.
├── flask_api/        # Yapay zekayı dışa açan Flask Web Sunucusu
├── models/           # Eğitilmiş Model (model.pkl, scaler.pkl, feature_db.npy)
├── scripts/          # Model Eğitimi (train.py) ve Tahmin Algoritmaları (predict.py)
├── web/              # PHP, JS ve CSS'den oluşan Modern Arayüz (Docker volume)
├── requirements.txt  # Gerekli Python kütüphaneleri listesi
└── README.md
```

Tebrikler! Sistem kullanıma hazır. Tarayıcınızdan `http://localhost:8080` adresine girerek test edebilirsiniz.
