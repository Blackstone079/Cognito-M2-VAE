"""Step 2: build model-ready features.

Featurizes molecules (Morgan fingerprints + a small descriptor set), builds
labels, applies train-only scaling for continuous descriptors, and writes a
single .npz bundle consumed by training/evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.labels.schemas import LabelPolicy, LabelSpec  # noqa: E402
from src.labels.build_multiclass import build_multiclass_labels  # noqa: E402
from src.features.featurize_rdkit import featurize_df  # noqa: E402
from src.features.scaling import apply_scaler, fit_scaler_on_train, scaler_to_json  # noqa: E402
from src.features.cache import save_features_npz, save_meta_json  # noqa: E402


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


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="experiments/config_m2.yaml")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    paths = Paths()

    df = pd.read_csv(paths.INTERIM / "drug_table.csv")

    split_path = paths.SPLITS / "split_scaffold_v1.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}. Run 01_make_split.py first.")
    split_df = pd.read_csv(split_path)

    # merge split onto df (inner join ensures valid smiles subset)
    df = df.merge(split_df, on="inchi_key", how="inner")

    # labels
    lp = cfg["label"]["policy"]
    spec = LabelSpec(
        column=str(cfg["label"]["column"]),
        policy=LabelPolicy(name=str(lp["name"]), min_count=int(lp.get("min_count", 0))),
        unknown_value=int(cfg["label"].get("unknown_value", -1)),
    )
    y, label_map, label_summary = build_multiclass_labels(df, spec)
    df["y"] = y

    # featurize
    df_f, X_fp, X_desc, scaffolds, invalid_df = featurize_df(
        df[["inchi_key", "smiles", "split", "y"]].copy(),
        n_bits=int(cfg["featurizer"]["fp_bits"]),
        radius=int(cfg["featurizer"]["fp_radius"]),
    )

    if len(invalid_df) > 0:
        # Should be empty because split step already filtered invalid smiles, but keep robust.
        invalid_df.to_csv(paths.INTERIM / "invalid_smiles_extra.csv", index=False)

    # align split + y after filtering
    split_arr = df_f["split"].astype(object).values
    y_arr = df_f["y"].astype(int).values
    inchi_key = df_f["inchi_key"].astype(object).values

    # scale descriptors using TRAIN only
    train_mask = split_arr.astype(str) == "train"
    scaler = fit_scaler_on_train(X_desc[train_mask])
    X_desc_s = apply_scaler(X_desc, scaler)

    feature_id = str(cfg["featurizer"].get("feature_id", "v1"))
    out_npz = paths.FEATURES / f"features_{feature_id}.npz"
    out_meta = paths.FEATURES / f"features_{feature_id}_meta.json"
    out_label_map = paths.FEATURES / f"label_map_{feature_id}.json"
    out_label_summary = paths.FEATURES / f"label_summary_{feature_id}.csv"
    out_rows = paths.FEATURES / f"dataset_rows_{feature_id}.csv"
    out_scaler = paths.FEATURES / f"desc_scaler_{feature_id}.json"

    # store split array inside npz as object array
    save_features_npz(
        out_npz,
        X_fp=X_fp,
        X_desc=X_desc_s,
        inchi_key=inchi_key,
        scaffolds=scaffolds,
        y=y_arr,
        split=split_arr,
    )

    meta = {
        "feature_id": feature_id,
        "fp_bits": int(cfg["featurizer"]["fp_bits"]),
        "fp_radius": int(cfg["featurizer"]["fp_radius"]),
        "desc_names": DESC_NAMES,
        "desc_scaled": True,
        "d_fp": int(X_fp.shape[1]),
        "d_desc": int(X_desc_s.shape[1]),
        "num_rows": int(len(df_f)),
    }
    save_meta_json(out_meta, meta)

    Path(out_label_map).write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    label_summary.to_csv(out_label_summary, index=False)

    Path(out_scaler).write_text(json.dumps(scaler_to_json(scaler), indent=2), encoding="utf-8")

    df_f.to_csv(out_rows, index=False)

    print(f"[ok] wrote {out_npz}")
    print(f"[ok] wrote {out_meta}")
    print(f"[ok] wrote {out_label_map} (K={label_map['num_classes']})")


if __name__ == "__main__":
    main()
