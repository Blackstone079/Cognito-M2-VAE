# src/features/scaling.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .featurize_rdkit import DescScaler


def fit_scaler_on_train(X_desc_train: np.ndarray) -> DescScaler:
    mu = X_desc_train.mean(axis=0)
    sd = X_desc_train.std(axis=0) + 1e-8
    return DescScaler(mean=mu.astype(np.float32), std=sd.astype(np.float32))


def apply_scaler(X_desc: np.ndarray, scaler: DescScaler) -> np.ndarray:
    return ((X_desc - scaler.mean) / scaler.std).astype(np.float32)


def scaler_to_json(scaler: DescScaler) -> Dict:
    return {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()}


def scaler_from_json(d: Dict) -> DescScaler:
    return DescScaler(mean=np.array(d["mean"], dtype=np.float32), std=np.array(d["std"], dtype=np.float32))
