import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_predictions import identify_columns, point_metrics


def metrics(path):
    frame = pd.read_csv(path)
    label, _, folds = identify_columns(frame)
    scores = frame[folds].to_numpy(dtype=float).mean(axis=1)
    return point_metrics(frame[label].to_numpy(dtype=int), scores, threshold=0.5)


def test_tcga_reference_metrics():
    observed = metrics(ROOT / "reference" / "tcga" / "predictions.csv")
    expected = np.asarray([0.7827190827, 0.7596124313, 0.7260273973])
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=0)


def test_ispy2_reference_metrics():
    observed = metrics(ROOT / "reference" / "ispy2" / "predictions.csv")
    expected = np.asarray([0.6743679163, 0.3432093780, 0.0])
    np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=0)
