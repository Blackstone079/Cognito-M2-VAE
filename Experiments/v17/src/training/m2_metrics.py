from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float('nan')


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {'n': 0, 'acc': float('nan'), 'bal_acc': float('nan'), 'f1_macro': float('nan'), 'f1_weighted': float('nan')}
    pred = np.argmax(probs, axis=1) + 1
    return {
        'n': int(len(y_true)),
        'acc': float(accuracy_score(y_true, pred)),
        'bal_acc': float(balanced_accuracy_score(y_true, pred)),
        'f1_macro': float(f1_score(y_true, pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, pred, average='weighted')),
    }


def _binary_score(y_true: np.ndarray, pred: np.ndarray, metric: str) -> float:
    if metric == 'bal_acc':
        return float(balanced_accuracy_score(y_true, pred))
    if metric == 'f1':
        return float(f1_score(y_true, pred, zero_division=0))
    if metric == 'acc':
        return float(accuracy_score(y_true, pred))
    raise ValueError(f'Unsupported threshold metric: {metric}')


def select_binary_threshold(y_true: np.ndarray, probs_pos: np.ndarray, metric: str = 'bal_acc') -> float:
    if len(y_true) == 0:
        return 0.5
    uniq = np.unique(y_true)
    if len(uniq) < 2:
        return 0.5
    grid = np.linspace(0.01, 0.99, 199)
    best_thr = 0.5
    best_score = -np.inf
    for thr in grid:
        pred = (probs_pos >= thr).astype(int)
        score = _binary_score(y_true, pred, metric)
        if score > best_score or (np.isclose(score, best_score) and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score = score
            best_thr = float(thr)
    return float(best_thr)


def _binary_ece(y_true: np.ndarray, probs_pos: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float('nan')
    y_true = np.asarray(y_true).astype(int)
    probs_pos = np.asarray(probs_pos).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi >= 1.0:
            mask = (probs_pos >= lo) & (probs_pos <= hi)
        else:
            mask = (probs_pos >= lo) & (probs_pos < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs_pos[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true: np.ndarray, probs_pos: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            'n': 0,
            'threshold': float(threshold),
            'acc': float('nan'),
            'bal_acc': float('nan'),
            'f1': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'specificity': float('nan'),
            'auroc': float('nan'),
            'auprc': float('nan'),
            'brier': float('nan'),
            'ece10': float('nan'),
        }
    pred = (probs_pos >= threshold).astype(int)
    out = {
        'n': int(len(y_true)),
        'threshold': float(threshold),
        'acc': float(accuracy_score(y_true, pred)),
        'bal_acc': float(balanced_accuracy_score(y_true, pred)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'specificity': float('nan'),
        'auroc': float('nan'),
        'auprc': float('nan'),
        'brier': float(np.mean((probs_pos - y_true) ** 2)),
        'ece10': _binary_ece(y_true, probs_pos, n_bins=10),
    }
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    denom = tn + fp
    if denom > 0:
        out['specificity'] = float(tn / denom)
    uniq = np.unique(y_true)
    if len(uniq) > 1:
        out['auroc'] = _safe_float(roc_auc_score(y_true, probs_pos))
        out['auprc'] = _safe_float(average_precision_score(y_true, probs_pos))
    return out


def confusion_from_probs(y_true: np.ndarray, probs: np.ndarray, *, task: str, threshold: float = 0.5) -> np.ndarray:
    if task == 'protox':
        pred = np.argmax(probs, axis=1) + 1
        labels = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(pred).ravel()]))
        return confusion_matrix(y_true, pred, labels=labels)
    pred = (probs >= threshold).astype(int)
    return confusion_matrix(y_true, pred, labels=[0, 1])
