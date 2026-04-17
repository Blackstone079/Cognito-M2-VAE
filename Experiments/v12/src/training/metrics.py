# src/training/metrics.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, average: str = "macro") -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
