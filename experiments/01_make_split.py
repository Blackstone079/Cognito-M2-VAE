"""Step 1: scaffold split.

Creates a train/val/test split based on Bemis–Murcko scaffolds to reduce
information leakage due to close structural analogs.

Writes:
  - data/splits/split_scaffold_v1.csv
  - data/interim/invalid_smiles.csv
"""
from __future__ import annotations

import argparse
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
from src.features.featurize_rdkit import featurize_df  # noqa: E402
from src.splits.scaffold_split import scaffold_train_val_test_split  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="experiments/config_m2.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    paths = Paths()

    in_csv = paths.INTERIM / "drug_table.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}. Run 00_extract_dataset.py first.")

    df = pd.read_csv(in_csv)

    # featurize just to validate SMILES and compute scaffolds
    df_f, _X_fp, _X_desc, scaffolds, invalid_df = featurize_df(
        df[["inchi_key", "smiles"]].copy(),
        n_bits=int(cfg["featurizer"]["fp_bits"]),
        radius=int(cfg["featurizer"]["fp_radius"]),
    )

    invalid_path = paths.INTERIM / "invalid_smiles.csv"
    invalid_df.to_csv(invalid_path, index=False)

    tr, va, te = scaffold_train_val_test_split(
        scaffolds,
        frac_train=float(cfg["split"]["frac_train"]),
        frac_val=float(cfg["split"]["frac_val"]),
        seed=int(cfg["split"]["seed"]),
    )

    split = np.array(["test"] * len(df_f), dtype=object)
    split[tr] = "train"
    split[va] = "val"
    split[te] = "test"

    out_split = paths.SPLITS / "split_scaffold_v1.csv"
    out_split.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(
        {
            "inchi_key": df_f["inchi_key"].values,
            "split": split,
        }
    )
    out_df.to_csv(out_split, index=False)

    (paths.SPLITS / "valid_inchi_keys_v1.csv").write_text(
        "\n".join(out_df["inchi_key"].astype(str).tolist()) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {out_split}")
    print(f"[ok] wrote {invalid_path} (n_invalid={len(invalid_df)})")


if __name__ == "__main__":
    main()
