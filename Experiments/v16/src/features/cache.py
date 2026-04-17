# src/features/cache.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def save_features_npz(
    path_npz: Path,
    *,
    X_fp: np.ndarray,
    X_desc: np.ndarray,
    inchi_key: np.ndarray,
    scaffolds: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
) -> None:
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_npz,
        X_fp=X_fp.astype(np.float32),
        X_desc=X_desc.astype(np.float32),
        inchi_key=inchi_key.astype(object),
        scaffolds=scaffolds.astype(object),
        y=y.astype(np.int64),
        split=split.astype(object),
    )


def load_features_npz(path_npz: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path_npz, allow_pickle=True))


def save_meta_json(path_json: Path, meta: Dict) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_meta_json(path_json: Path) -> Dict:
    return json.loads(path_json.read_text(encoding="utf-8"))
