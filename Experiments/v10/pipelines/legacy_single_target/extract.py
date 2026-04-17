"""Step 0 (streaming): extract the drug table from the project SQLite database.

Writes:
 - data/interim/drug_table.csv
 - data/interim/extract_summary.json

This version streams the join (ORDER BY inchi_key) to avoid `pd.read_sql` over the full join.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_on_path()

from src.config import Paths  # noqa: E402
from src.db.extract_stream import ExtractConfig, extract_drug_table_stream  # noqa: E402


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the drug table from the SQLite DB (streaming).")
    ap.add_argument("--config", type=str, default="pipelines/legacy_single_target/config.yaml", help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.config))
    paths = Paths()

    db_path = Path(cfg["db"]["path"])
    if not db_path.is_absolute():
        db_path = paths.ROOT / db_path

    out_csv = paths.INTERIM / "drug_table.csv"

    summary = extract_drug_table_stream(
        db_path,
        out_csv,
        cfg=ExtractConfig(include_drugbank_text=bool(cfg["db"].get("include_drugbank_text", False))),
    )

    (paths.INTERIM / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {paths.INTERIM / 'extract_summary.json'}")


if __name__ == "__main__":
    main()
