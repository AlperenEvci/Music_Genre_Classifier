import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def make_dummy_df(n_samples: int = 100, n_features: int = 41) -> pd.DataFrame:
    """10 türde eşit dağılımlı sahte öznitelik DataFrame'i."""
    from scripts.utils import GENRES
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    genres = np.tile(GENRES, n_samples // len(GENRES) + 1)[:n_samples]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["genre"] = genres
    df["filename"] = [f"dummy_{i}.wav" for i in range(n_samples)]
    return df


def test_train_returns_model_scaler_metrics(tmp_path):
    from scripts.train import train_classifier

    df = make_dummy_df()
    model, scaler, metrics = train_classifier(df, models_dir=tmp_path)

    assert model is not None
    assert scaler is not None
    assert "accuracy" in metrics
    assert "model_name" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "scaler.pkl").exists()


def test_feature_db_saved(tmp_path):
    from scripts.train import save_feature_db

    df = make_dummy_df()
    save_feature_db(df, tmp_path)

    db = np.load(str(tmp_path / "feature_db.npy"))
    labels = np.load(str(tmp_path / "feature_db_labels.npy"))
    filenames = np.load(str(tmp_path / "feature_db_filenames.npy"))

    assert db.shape[0] == len(df)
    assert db.shape[1] == 41
    assert len(labels) == len(df)
    assert len(filenames) == len(df)


def test_get_xy_excludes_meta_columns(tmp_path):
    from scripts.train import _get_xy

    df = make_dummy_df()
    X, y = _get_xy(df)

    assert X.shape == (100, 41)
    assert len(y) == 100
    assert "genre" not in X.tolist()


def test_train_with_pca(tmp_path):
    from scripts.train import train_with_pca

    df = make_dummy_df()
    result = train_with_pca(df, n_components=10, models_dir=tmp_path)

    assert "accuracy" in result
    assert "f1_macro" in result
    assert "explained_variance" in result
    assert 0.0 < result["explained_variance"] <= 1.0
    assert (tmp_path / "pca_analysis.pkl").exists()
