# scripts/visualize.py
"""
Akademik rapor için confusion matrix ve görselleştirme.
Çalıştır: python scripts/visualize.py
Çıktı: confusion_matrix.png (proje root'unda)
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from scripts.utils import GENRES, DATA_DIR, MODELS_DIR


def plot_confusion_matrix(output_path: str = "confusion_matrix.png") -> None:
    """
    features.csv + model.pkl + scaler.pkl kullanarak test seti confusion matrix çiz.
    Raises: FileNotFoundError — gerekli dosyalar yoksa
    """
    features_csv = DATA_DIR / "features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Önce extract_features.py çalıştırın: {features_csv}")

    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c not in ("genre", "filename")]
    X = df[feature_cols].values
    y = df["genre"].values

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    model  = joblib.load(MODELS_DIR / "model.pkl")

    X_scaled = scaler.transform(X)
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=GENRES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Random Forest (GTZAN Test Set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Kaydedildi: {output_path}")


if __name__ == "__main__":
    plot_confusion_matrix("confusion_matrix.png")
