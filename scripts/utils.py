# scripts/utils.py
import functools
import numpy as np
from pathlib import Path
import librosa
import logging

logger = logging.getLogger(__name__)

GENRES = [
    "blues", "classical", "country", "disco",
    "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

SAMPLE_RATE = 22050
DURATION = 30
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR   = Path(__file__).parent.parent / "data"


def genre_from_filename(filename: str) -> str:
    """'blues.00042.wav' → 'blues'"""
    if not filename:
        raise ValueError(f"Empty filename: {filename!r}")
    return Path(filename).stem.split(".")[0]


@functools.lru_cache(maxsize=None)
def vector_labels() -> tuple[str, ...]:
    """Öznitelik vektörünün sütun isimlerini döndürür. Toplam 41 eleman."""
    cols  = [f"mfcc_{i}_mean" for i in range(N_MFCC)]
    cols += [f"mfcc_{i}_std"  for i in range(N_MFCC)]
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
    return tuple(cols)  # 13×2=26 + 15 = 41


def scan_dataset(gtzan_root: Path) -> tuple[list[Path], list[str]]:
    """
    GTZAN dizinini tara, bozuk dosyaları atla.
    Returns: (valid_paths, skipped_paths)
    """
    valid: list[Path] = []
    skipped: list[str] = []

    genres_dir = gtzan_root / "genres"
    if not genres_dir.exists():
        return [], []
    for wav_path in sorted(genres_dir.rglob("*.wav")):
        try:
            info = librosa.get_samplerate(str(wav_path))
            if info < 1:
                raise ValueError("Geçersiz sample rate")
            valid.append(wav_path)
        except Exception as e:
            skipped.append(str(wav_path))
            logger.warning("Atlandı: %s — %s", wav_path.name, e)

    logger.info("Geçerli: %d | Atlandı: %d", len(valid), len(skipped))
    return valid, skipped
