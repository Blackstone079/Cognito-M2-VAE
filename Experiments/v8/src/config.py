# src/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    # src/config.py -> src -> repo root
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    ROOT: Path = project_root()
    DATA: Path = ROOT / "data"
    RAW: Path = DATA / "raw"
    INTERIM: Path = DATA / "interim"
    PROCESSED: Path = DATA / "processed"
    SPLITS: Path = PROCESSED / "splits"
    FEATURES: Path = PROCESSED / "features"
    RESULTS: Path = ROOT / "results"
    RUNS: Path = RESULTS / "runs"


DEFAULT_DB_FILENAME = "epshteins_list.db"


def default_db_path() -> Path:
    return Paths().RAW / DEFAULT_DB_FILENAME
