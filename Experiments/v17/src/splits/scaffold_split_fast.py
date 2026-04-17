from __future__ import annotations

from typing import Tuple

import numpy as np


def scaffold_train_val_test_split_fast(
    scaffold_ids: np.ndarray,
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy scaffold split that scales to large datasets.

    The crucial detail is that allocation is based on the number of molecules
    already assigned, not on the number of scaffold groups already assigned.
    """
    sc = np.asarray(scaffold_ids)
    if sc.ndim != 1:
        raise ValueError("scaffold_ids must be 1D")

    n = int(len(sc))
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)

    order = np.argsort(sc, kind="mergesort")
    sc_sorted = sc[order]
    uniq, start, counts = np.unique(sc_sorted, return_index=True, return_counts=True)

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(uniq))
    start_shuf = start[perm]
    counts_shuf = counts[perm]

    sort_idx = np.argsort(-counts_shuf, kind="mergesort")

    train_idx = []
    val_idx = []
    test_idx = []
    n_train_now = 0
    n_val_now = 0

    for k in sort_idx:
        s0 = int(start_shuf[k])
        c = int(counts_shuf[k])
        idx = order[s0 : s0 + c]
        if n_train_now + c <= n_train:
            train_idx.append(idx)
            n_train_now += c
        elif n_val_now + c <= n_val:
            val_idx.append(idx)
            n_val_now += c
        else:
            test_idx.append(idx)

    tr = np.concatenate(train_idx, axis=0) if train_idx else np.array([], dtype=int)
    va = np.concatenate(val_idx, axis=0) if val_idx else np.array([], dtype=int)
    te = np.concatenate(test_idx, axis=0) if test_idx else np.array([], dtype=int)
    return tr.astype(int), va.astype(int), te.astype(int)
