# src/labels/build_multiclass.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from .schemas import LabelSpec


def _to_int_or_none(v):
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        x = int(float(v))
        return x
    except Exception:
        return None


def build_multiclass_labels(df: pd.DataFrame, spec: LabelSpec) -> Tuple[np.ndarray, Dict, pd.DataFrame]:
    """Create y (int) for M2.

    Output:
      - y: shape (N,), values in {0..K-1} or spec.unknown_value
      - label_map: json-serializable mapping (classes, policy, counts)
      - summary_df: class counts table
    """
    if spec.column not in df.columns:
        raise KeyError(f"Label column '{spec.column}' not found in dataframe.")

    raw = df[spec.column].tolist()
    raw_int = np.array([_to_int_or_none(v) for v in raw], dtype=object)

    # accept only 1..6 for protox_toxclass
    valid_mask = np.array([(v is not None) and (1 <= v <= 6) for v in raw_int], dtype=bool)
    valid_vals = np.array([v for v in raw_int[valid_mask]], dtype=int)

    # base mapping 1..6 -> 0..5
    base_classes = [1, 2, 3, 4, 5, 6]

    policy = spec.policy.name.strip().lower()

    # policy transforms (in raw label space first)
    if policy == "strict_6_class":
        kept = base_classes
        raw_to_model = {c: (c - 1) for c in kept}

    elif policy == "merge_i_ii":
        # merge 1 and 2 into a single class "I_II" -> id 0
        # map 3..6 -> ids 1..4
        kept = [1, 2, 3, 4, 5, 6]
        raw_to_model = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}

    elif policy == "drop_min_count":
        # drop classes with < min_count (turn them into unknown)
        min_count = int(spec.policy.min_count)
        counts = {c: int(np.sum(valid_vals == c)) for c in base_classes}
        kept = [c for c in base_classes if counts.get(c, 0) >= min_count]
        raw_to_model = {c: i for i, c in enumerate(sorted(kept))}

    else:
        raise ValueError(f"Unknown LabelPolicy: {spec.policy.name}")

    K = len(set(raw_to_model.values()))
    y = np.full((len(df),), spec.unknown_value, dtype=int)

    for i, v in enumerate(raw_int):
        if v is None:
            continue
        if v in raw_to_model:
            y[i] = raw_to_model[v]
        else:
            # valid raw label but dropped by policy
            y[i] = spec.unknown_value

    # build summary
    counts_model = []
    for k in range(K):
        counts_model.append(int(np.sum(y == k)))
    n_unknown = int(np.sum(y == spec.unknown_value))

    class_rows = []
    # stable names
    if policy == "merge_i_ii":
        names = {0: "I_II", 1: "III", 2: "IV", 3: "V", 4: "VI"}
        # raw sources per class
        raw_sources = {0: [1, 2], 1: [3], 2: [4], 3: [5], 4: [6]}
        for k in range(K):
            class_rows.append(
                dict(model_id=k, name=names.get(k, str(k)), source_values=raw_sources.get(k, []), count=counts_model[k])
            )
    else:
        # strict or dropped: order by model_id
        inv = {}
        for rv, mid in raw_to_model.items():
            inv.setdefault(mid, []).append(rv)
        for k in range(K):
            sv = sorted(inv.get(k, []))
            nm = "_".join(["I" if x == 1 else "II" if x == 2 else "III" if x == 3 else "IV" if x == 4 else "V" if x == 5 else "VI" if x == 6 else str(x) for x in sv]) or str(k)
            class_rows.append(dict(model_id=k, name=nm, source_values=sv, count=counts_model[k]))

    summary_df = pd.DataFrame(class_rows).sort_values("model_id").reset_index(drop=True)

    label_map = {
        "column": spec.column,
        "unknown_value": spec.unknown_value,
        "policy": asdict(spec.policy),
        "num_classes": K,
        "classes": class_rows,
        "unknown_count": n_unknown,
        "total": int(len(df)),
    }

    return y, label_map, summary_df
