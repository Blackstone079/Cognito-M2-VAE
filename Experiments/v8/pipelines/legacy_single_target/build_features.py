"""Step 2 (memmap): build features.

Streams `data/interim/drug_table.csv` and writes disk-backed feature arrays (memmap `.npy`).

Inputs:
 - data/interim/drug_table.csv
 - data/processed/splits/split_scaffold_v1.csv

Writes:
 - data/processed/features/features_<feature_id>_memmap/
     - X_fp.npy (float32, shape N x fp_bits)
     - X_desc.npy (float32, shape N x 8; scaled using TRAIN only)
     - y.npy (int64, shape N; -1 for unlabeled)
     - split_code.npy (uint8, 0=train,1=val,2=test)
     - inchi_key.npy ('S27')
 - data/processed/features/features_<feature_id>_memmap_meta.json
 - data/processed/features/label_map_<feature_id>.json
 - data/processed/features/label_summary_<feature_id>.csv
 - data/processed/features/desc_scaler_<feature_id>.json

This keeps model maths identical; it changes storage so you don't load all features into RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.features.featurize_rdkit import mol_from_smiles, morgan_fp, desc8, DescScaler  # noqa: E402

DESC_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHBD",
    "NumHBA",
    "NumRotatableBonds",
    "NumRings",
    "FractionCSP3",
]

SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _to_int_or_none(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, str) and not v.strip():
            return None
        x = int(float(v))
        return x
    except Exception:
        return None


def _build_raw_to_model(policy: str, min_count: int, raw_counts: Dict[int, int]) -> Dict[int, int]:
    p = policy.strip().lower()
    base = [1, 2, 3, 4, 5, 6]

    if p == "strict_6_class":
        return {c: (c - 1) for c in base}

    if p == "merge_i_ii" or p == "merge_i_ii".lower():
        return {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}

    if p == "drop_min_count":
        kept = [c for c in base if int(raw_counts.get(c, 0)) >= int(min_count)]
        kept = sorted(kept)
        return {c: i for i, c in enumerate(kept)}

    raise ValueError(f"Unknown label policy: {policy}")


def _label_map_and_summary(spec_column: str, unknown_value: int, policy: str, min_count: int, raw_to_model: Dict[int, int], counts_model: Dict[int, int], n_unknown: int, total: int) -> tuple[Dict, list[Dict]]:
    K = len(set(raw_to_model.values()))

    class_rows = []
    p = policy.strip().lower()
    if p == "merge_i_ii":
        names = {0: "I_II", 1: "III", 2: "IV", 3: "V", 4: "VI"}
        raw_sources = {0: [1, 2], 1: [3], 2: [4], 3: [5], 4: [6]}
        for k in range(K):
            class_rows.append(
                dict(
                    model_id=k,
                    name=names.get(k, str(k)),
                    source_values=raw_sources.get(k, []),
                    count=int(counts_model.get(k, 0)),
                )
            )
    else:
        inv: Dict[int, list[int]] = {}
        for rv, mid in raw_to_model.items():
            inv.setdefault(int(mid), []).append(int(rv))

        def _roman(x: int) -> str:
            return "I" if x == 1 else "II" if x == 2 else "III" if x == 3 else "IV" if x == 4 else "V" if x == 5 else "VI" if x == 6 else str(x)

        for k in range(K):
            sv = sorted(inv.get(k, []))
            nm = "_".join(_roman(x) for x in sv) or str(k)
            class_rows.append(dict(model_id=k, name=nm, source_values=sv, count=int(counts_model.get(k, 0))))

    label_map = {
        "column": spec_column,
        "unknown_value": int(unknown_value),
        "policy": {"name": policy, "min_count": int(min_count)},
        "num_classes": int(K),
        "classes": class_rows,
        "unknown_count": int(n_unknown),
        "total": int(total),
    }
    return label_map, class_rows


def _welford_update(n: int, mean: np.ndarray, M2: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n1 = n + 1
    delta = x - mean
    mean = mean + delta / n1
    delta2 = x - mean
    M2 = M2 + delta * delta2
    return n1, mean, M2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="pipelines/legacy_single_target/config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    paths = Paths()

    in_csv = paths.INTERIM / "drug_table.csv"
    split_path = paths.SPLITS / "split_scaffold_v1.csv"

    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}. Run 00_extract_dataset_stream.py first.")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run 01_make_split_stream.py first.")

    # Load split mapping (valid smiles subset)
    split_map: Dict[str, int] = {}
    with split_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get("inchi_key") or "").strip()
            sp = (row.get("split") or "").strip().lower()
            if ik and sp in SPLIT_TO_CODE:
                split_map[ik] = int(SPLIT_TO_CODE[sp])

    N = len(split_map)
    if N == 0:
        raise RuntimeError("Split file contained 0 valid rows.")

    feature_id = str(cfg["featurizer"].get("feature_id", "v1"))
    fp_bits = int(cfg["featurizer"]["fp_bits"])
    fp_radius = int(cfg["featurizer"]["fp_radius"])

    out_dir = paths.FEATURES / f"features_{feature_id}_memmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Label policy setup
    label_col = str(cfg["label"]["column"])
    lp = cfg["label"]["policy"]
    policy_name = str(lp["name"])
    min_count = int(lp.get("min_count", 0))
    unknown_value = int(cfg["label"].get("unknown_value", -1))

    raw_counts = {i: 0 for i in range(1, 7)}
    if policy_name.strip().lower() == "drop_min_count":
        # pass0: count raw labels among valid subset
        with in_csv.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ik = (row.get("inchi_key") or "").strip()
                if ik not in split_map:
                    continue
                v = _to_int_or_none(row.get(label_col))
                if v is not None and 1 <= v <= 6:
                    raw_counts[int(v)] += 1

    raw_to_model = _build_raw_to_model(policy_name, min_count, raw_counts)
    K = len(set(raw_to_model.values()))

    # Allocate memmaps
    X_fp = np.lib.format.open_memmap(out_dir / "X_fp.npy", mode="w+", dtype=np.float32, shape=(N, fp_bits))
    X_desc = np.lib.format.open_memmap(out_dir / "X_desc.npy", mode="w+", dtype=np.float32, shape=(N, 8))
    y_mm = np.lib.format.open_memmap(out_dir / "y.npy", mode="w+", dtype=np.int64, shape=(N,))
    split_mm = np.lib.format.open_memmap(out_dir / "split_code.npy", mode="w+", dtype=np.uint8, shape=(N,))
    ik_mm = np.lib.format.open_memmap(out_dir / "inchi_key.npy", mode="w+", dtype="S27", shape=(N,))

    # pass1: descriptors + labels + split + inchi_key
    n_train = 0
    mean = np.zeros((8,), dtype=np.float64)
    M2 = np.zeros((8,), dtype=np.float64)

    counts_model = {i: 0 for i in range(K)}
    n_unknown = 0

    idx = 0
    with in_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get("inchi_key") or "").strip()
            code = split_map.get(ik, None)
            if code is None:
                continue

            smi = row.get("smiles")
            mol = mol_from_smiles(smi)
            if mol is None:
                # should not happen if split step was built on the same CSV, but remain robust
                continue

            # descriptors
            d = desc8(mol).astype(np.float32)
            X_desc[idx, :] = d

            # label
            raw_v = _to_int_or_none(row.get(label_col))
            if raw_v is None:
                yv = unknown_value
            else:
                rv = int(raw_v)
                if rv in raw_to_model:
                    yv = int(raw_to_model[rv])
                else:
                    yv = unknown_value

            y_mm[idx] = int(yv)
            if yv == unknown_value:
                n_unknown += 1
            else:
                counts_model[int(yv)] = counts_model.get(int(yv), 0) + 1

            split_mm[idx] = int(code)
            ik_mm[idx] = ik.encode("ascii", errors="ignore")[:27]

            if int(code) == 0:  # train
                n_train, mean, M2 = _welford_update(n_train, mean, M2, d.astype(np.float64))

            idx += 1

    if idx != N:
        # If this happens, the split file and drug_table.csv are inconsistent.
        raise RuntimeError(f"Expected N={N} valid rows from split file, but wrote idx={idx}.")

    # fit scaler (population std, matching np.std default ddof=0)
    if n_train <= 0:
        raise RuntimeError("No TRAIN rows found; cannot fit descriptor scaler.")

    var = (M2 / max(n_train, 1)).astype(np.float32)
    sd = (np.sqrt(var) + 1e-8).astype(np.float32)
    mu = mean.astype(np.float32)
    scaler = DescScaler(mean=mu, std=sd)

    # scale descriptors in-place (chunked)
    chunk = 100_000
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        X_desc[s:e, :] = ((X_desc[s:e, :] - scaler.mean) / scaler.std).astype(np.float32)

    # pass2: fingerprints
    idx = 0
    with in_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ik = (row.get("inchi_key") or "").strip()
            if ik not in split_map:
                continue
            smi = row.get("smiles")
            mol = mol_from_smiles(smi)
            if mol is None:
                continue
            X_fp[idx, :] = morgan_fp(mol, n_bits=fp_bits, radius=fp_radius)
            idx += 1

    if idx != N:
        raise RuntimeError(f"Fingerprint pass mismatch: expected N={N}, got idx={idx}.")

    # write meta + mappings
    meta = {
        "feature_id": feature_id,
        "fp_bits": fp_bits,
        "fp_radius": fp_radius,
        "desc_names": DESC_NAMES,
        "desc_scaled": True,
        "d_fp": int(fp_bits),
        "d_desc": 8,
        "num_rows": int(N),
        "format": "memmap",
    }

    (paths.FEATURES / f"features_{feature_id}_memmap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    label_map, class_rows = _label_map_and_summary(
        label_col,
        unknown_value,
        policy_name,
        min_count,
        raw_to_model,
        counts_model,
        n_unknown,
        N,
    )

    (paths.FEATURES / f"label_map_{feature_id}.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    # label summary CSV
    out_sum = paths.FEATURES / f"label_summary_{feature_id}.csv"
    with out_sum.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "name", "source_values", "count"])
        w.writeheader()
        for row in class_rows:
            w.writerow(
                {
                    "model_id": row["model_id"],
                    "name": row["name"],
                    "source_values": json.dumps(row["source_values"], ensure_ascii=False),
                    "count": row["count"],
                }
            )

    (paths.FEATURES / f"desc_scaler_{feature_id}.json").write_text(
        json.dumps({"mean": scaler.mean.tolist(), "std": scaler.std.tolist()}, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] wrote memmap features to {out_dir}")


if __name__ == "__main__":
    main()
