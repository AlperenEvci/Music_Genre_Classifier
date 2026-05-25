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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # GUI olmadan PNG kaydet
import matplotlib.pyplot as plt
from scripts.utils import MODELS_DIR, DATA_DIR


def _get_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame'den öznitelik matrisi X ve etiket vektörü y çıkar."""
    feature_cols = [c for c in df.columns if c not in ("genre", "filename")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["genre"].values
    return X, y


def train_classifier(
    df: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
) -> tuple:
    """
    RF, KNN, DT eğit; 5-fold CV ile en iyiyi seç; model.pkl + scaler.pkl kaydet.
    Returns: (best_model, scaler, metrics_dict)
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y = _get_xy(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
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
            "cv_std":  float(cv_scores.std()),
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
        "accuracy":   best["test_accuracy"],
        "f1_macro":   best["test_f1_macro"],
        "cv_mean":    best["cv_mean"],
        "all_models": {
            k: {kk: vv for kk, vv in v.items() if kk != "clf"}
            for k, v in results.items()
        },
    }
    return best["clf"], scaler, metrics


def save_feature_db(df: pd.DataFrame, models_dir: Path = MODELS_DIR) -> None:
    """Tüm öznitelik vektörlerini öneri sistemi için .npy olarak kaydet."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y = _get_xy(df)
    filenames = df["filename"].values if "filename" in df.columns else np.array([""] * len(df))

    np.save(str(models_dir / "feature_db.npy"), X)
    np.save(str(models_dir / "feature_db_labels.npy"), np.array(y, dtype="U"))
    np.save(str(models_dir / "feature_db_filenames.npy"), np.array(filenames, dtype="U"))
    print(f"feature_db kaydedildi: {X.shape}")


def train_with_pca(
    df: pd.DataFrame,
    n_components: int = 20,
    models_dir: Path = MODELS_DIR,
) -> dict:
    """
    PCA ile boyut indirgeme yaparak RF eğit; scaler+pca kaydet.
    Returns: {"accuracy": float, "f1_macro": float, "explained_variance": float}
    """
    models_dir = Path(models_dir)
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

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


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
