# scripts/extract_features.py
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from scripts.utils import (
    SAMPLE_RATE, DURATION, N_MFCC, HOP_LENGTH, N_FFT,
    genre_from_filename, DATA_DIR, scan_dataset
)

MIN_DURATION_SEC = 3.0


def extract_features_from_file(file_path: str) -> np.ndarray:
    """
    Tek .wav/.mp3 dosyasından 41 boyutlu öznitelik vektörü çıkar.
    Raises: ValueError — dosya çok kısaysa
            Exception — librosa yükleyemezse
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    if len(y) < sr * MIN_DURATION_SEC:
        raise ValueError(f"Ses dosyası çok kısa: {len(y) / sr:.1f}s (min {MIN_DURATION_SEC}s)")

    # MFCC: 13 × 2 = 26
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])  # 26

    # Spectral Centroid: 2
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    sc_feat = np.array([sc.mean(), sc.std()])

    # ZCR: 2
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    zcr_feat = np.array([zcr.mean(), zcr.std()])

    # Chroma: 2
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    chroma_feat = np.array([chroma.mean(), chroma.std()])

    # Tempo: 1
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo_feat = np.array([float(tempo)])

    # Spectral Rolloff: 2
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff_feat = np.array([rolloff.mean(), rolloff.std()])

    # Spectral Bandwidth: 2
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    bw_feat = np.array([bw.mean(), bw.std()])

    # RMS Energy: 2
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    rms_feat = np.array([rms.mean(), rms.std()])

    # Spectral Contrast: 2
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH)
    contrast_feat = np.array([contrast.mean(), contrast.std()])

    vector = np.concatenate([
        mfcc_feat,      # 26
        sc_feat,        # 2
        zcr_feat,       # 2
        chroma_feat,    # 2
        tempo_feat,     # 1
        rolloff_feat,   # 2
        bw_feat,        # 2
        rms_feat,       # 2
        contrast_feat,  # 2
    ])  # Toplam: 41

    assert vector.shape == (41,), f"Beklenen 41, alınan {vector.shape}"
    return vector.astype(np.float32)


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
            row = {f"f{j}": float(features[j]) for j in range(len(features))}
            row["genre"] = genre
            row["filename"] = wav_path.name
            records.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(valid_paths)} tamamlandı...")
        except Exception as e:
            print(f"  HATA [{wav_path.name}]: {e}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    gtzan_root = DATA_DIR / "gtzan"
    df = batch_extract(gtzan_root)
    output_path = DATA_DIR / "features.csv"
    df.to_csv(output_path, index=False)
    print(f"Kaydedildi: {output_path} ({len(df)} satır)")
