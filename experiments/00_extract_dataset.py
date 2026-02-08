"""Step 0: extract the drug table from the project SQLite database.

Writes:
  - data/interim/drug_table.csv
  - data/interim/extract_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.db.extract import ExtractConfig, extract_drug_table  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the drug table from the SQLite DB.")
    ap.add_argument("--config", type=str, default="experiments/config_m2.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))

    paths = Paths()
    db_path = Path(cfg["db"]["path"])
    if not db_path.is_absolute():
        db_path = paths.ROOT / db_path

    out_csv = paths.INTERIM / "drug_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = extract_drug_table(
        str(db_path),
        cfg=ExtractConfig(include_drugbank_text=bool(cfg["db"].get("include_drugbank_text", False))),
    )
    df.to_csv(out_csv, index=False)

    # quick summary
    summary = {
        "n_rows": int(len(df)),
        "n_smiles_nonnull": int(df["smiles"].notna().sum()),
        "n_protox_labeled": int(df["protox_toxclass"].notna().sum()),
    }
    (paths.INTERIM / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {paths.INTERIM / 'extract_summary.json'}")


if __name__ == "__main__":
    main()
