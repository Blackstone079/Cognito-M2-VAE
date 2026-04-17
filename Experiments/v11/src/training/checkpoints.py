# src/training/checkpoints.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch


def save_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "opt_state" in payload:
        optimizer.load_state_dict(payload["opt_state"])
    return payload
