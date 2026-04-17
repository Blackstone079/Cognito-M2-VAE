# src/utils/fingerprint.py
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_data_fingerprint(db_path: Path, featurizer_cfg: Dict, label_cfg: Dict, split_cfg: Dict) -> Dict:
    return {
        "db_path": str(db_path),
        "db_sha256": sha256_file(db_path) if db_path.exists() else None,
        "featurizer_cfg_sha256": sha256_text(str(featurizer_cfg)),
        "label_cfg_sha256": sha256_text(str(label_cfg)),
        "split_cfg_sha256": sha256_text(str(split_cfg)),
    }
