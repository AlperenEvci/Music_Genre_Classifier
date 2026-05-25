# tests/test_visualize.py
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def test_plot_confusion_matrix_raises_without_data(tmp_path):
    """features.csv yoksa FileNotFoundError raise etmeli."""
    from scripts.visualize import plot_confusion_matrix
    import scripts.utils as utils_mod

    # DATA_DIR'ı tmp_path'e yönlendir
    original_dir = utils_mod.DATA_DIR
    utils_mod.DATA_DIR = tmp_path

    try:
        with pytest.raises(FileNotFoundError):
            plot_confusion_matrix(str(tmp_path / "test.png"))
    finally:
        utils_mod.DATA_DIR = original_dir
