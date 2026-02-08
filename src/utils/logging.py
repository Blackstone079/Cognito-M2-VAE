# src/utils/logging.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RunLogger:
    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run.log"
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def log(self, msg: str) -> None:
        line = msg.strip()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_metrics(self, d: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")
