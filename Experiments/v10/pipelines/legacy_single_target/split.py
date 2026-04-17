"""Step 1 (streaming): scaffold split.

Streams `data/interim/drug_table.csv`, validates SMILES, computes Bemis–Murcko scaffold IDs,
then performs a greedy scaffold split (train/val/test).

Writes:
 - data/processed/splits/split_scaffold_v1.csv
 - data/interim/invalid_smiles.csv
 - data/processed/splits/valid_inchi_keys_v1.csv

This avoids loading the full drug table into pandas and avoids the O(N*#scaffolds) grouping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.features.featurize_rdkit import mol_from_smiles, murcko_scaffold_smiles  # noqa: E402
from src.splits.scaffold_split_fast import scaffold_train_val_test_split_fast  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _hash64(s: str) -> np.uint64:
    # Stable 64-bit hash (blake2b-64)
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(h, dtype=np.uint64)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="pipelines/legacy_single_target/config.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    paths = Paths()

    in_csv = paths.INTERIM / "drug_table.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing {in_csv}. Run 00_extract_dataset_stream.py first.")

    invalid_path = paths.INTERIM / "invalid_smiles.csv"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)

    inchi_keys: list[str] = []
    scaff_hash: list[np.uint64] = []
    n_invalid = 0

    with in_csv.open("r", newline="", encoding="utf-8") as f_in, invalid_path.open(
        "w", newline="", encoding="utf-8"
    ) as f_bad:
        r = csv.DictReader(f_in)
        w_bad = csv.DictWriter(f_bad, fieldnames=["inchi_key", "smiles"])
        w_bad.writeheader()

        for row in r:
            ik = (row.get("inchi_key") or "").strip()
            smi = row.get("smiles")
            mol = mol_from_smiles(smi)
            if mol is None:
                w_bad.writerow({"inchi_key": ik, "smiles": smi if smi is not None else ""})
                n_invalid += 1
                continue
            scaff = murcko_scaffold_smiles(mol)
            inchi_keys.append(ik)
            scaff_hash.append(_hash64(scaff))

    if len(inchi_keys) == 0:
        raise RuntimeError("No valid SMILES found. Check the input CSV.")

    sc = np.asarray(scaff_hash, dtype=np.uint64)

    tr, va, te = scaffold_train_val_test_split_fast(
        sc,
        frac_train=float(cfg["split"]["frac_train"]),
        frac_val=float(cfg["split"]["frac_val"]),
        seed=int(cfg["split"]["seed"]),
    )

    split = np.array(["test"] * len(inchi_keys), dtype=object)
    split[tr] = "train"
    split[va] = "val"
    split[te] = "test"

    out_split = paths.SPLITS / "split_scaffold_v1.csv"
    out_split.parent.mkdir(parents=True, exist_ok=True)

    with out_split.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=["inchi_key", "split"])
        w.writeheader()
        for ik, sp in zip(inchi_keys, split.tolist()):
            w.writerow({"inchi_key": ik, "split": sp})

    valid_keys_path = paths.SPLITS / "valid_inchi_keys_v1.csv"
    valid_keys_path.write_text("\n".join(inchi_keys) + "\n", encoding="utf-8")

    print(f"[ok] wrote {out_split}")
    print(f"[ok] wrote {invalid_path} (n_invalid={n_invalid})")


if __name__ == "__main__":
    main()
