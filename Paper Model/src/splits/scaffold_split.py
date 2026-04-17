# src/splits/scaffold_split.py
from __future__ import annotations

from typing import Tuple

import numpy as np


def scaffold_train_val_test_split(
    scaffolds: np.ndarray,
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy scaffold split: assign largest scaffolds first to train, then val, then test.

    This is deterministic given `seed` (ties are shuffled).
    """
    rng = np.random.RandomState(seed)

    scaffolds = scaffolds.astype(object)
    uniq = np.unique(scaffolds)
    rng.shuffle(uniq)

    groups = {sc: np.where(scaffolds == sc)[0] for sc in uniq}
    sizes = [(sc, len(groups[sc])) for sc in uniq]
    sizes.sort(key=lambda x: x[1], reverse=True)

    n = len(scaffolds)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    train_idx, val_idx, test_idx = [], [], []

    for sc, _ in sizes:
        g = groups[sc].tolist()
        if len(train_idx) + len(g) <= n_train:
            train_idx.extend(g)
        elif len(val_idx) + len(g) <= n_val:
            val_idx.extend(g)
        else:
            test_idx.extend(g)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)
